"""Compare experiments on the public tasks with the official harness.

Run through notebooks/colab_validate.ipynb (one A100) or notebooks/kaggle/validate (4x L4,
the scoring hardware; vLLM shards the model over every GPU as scoring does).
Each sampled task is first run with its reference
patch, and tasks that fail even then (the local sandbox lacks something the tests need) are
replaced by the next ones. One vLLM server then serves the official QAT model and every
adapter, and `swegemma eval` runs each experiment on the same tasks.
Finished tasks are kept on Drive (or in the Kaggle output, when that is attached to the next
run), so running again continues where it stopped.
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from kaggle_ws import find_kaggle_inputs, get_paths, is_kaggle, kaggle_input

COMPETITION_DIR = Path(__file__).resolve().parent
COMPETITION = COMPETITION_DIR.name
WHEELHOUSE = "metric/gemma-4-developer-agent-wheelhouse"
# Packages without a pin resolve to what was current when the scoring wheels were published.
EXCLUDE_NEWER = "2026-10-13T00:00:00Z"
SERVER_PACKAGES = ("vllm==0.19.1", "transformers==5.13.1", "compressed-tensors==0.15.0.1")
EVAL_PACKAGES = (
    "google-adk==1.36.1",
    "google-genai==2.11.0",
    "litellm",
    "pandas",
    "networkx",
    "cachetools",
    "python-dotenv",
    # installed in the official sandbox image (docker/Dockerfile.sandbox)
    "pytest",
    "pytest-timeout==2.1.0",
    "typer",
    "pdm-backend",
    "setuptools",
    "wheel",
    "poetry-core",
    "hatchling",
    "flit-core",
    "editables",
    # fastapi tests import these, but the published wheels do not include them
    "inline-snapshot",
    "dirty-equals",
    "orjson",
)
# requests tests need pytest-httpbin, which the published wheels do not include.
DEFAULT_REPOS = ("fastapi/fastapi", "Textualize/rich")
EVAL_FLAGS = {
    "timeout_seconds": "--timeout-seconds",
    "max_tool_calls": "--max-tool-calls",
    "max_time_minutes": "--max-time-minutes",
    "max_turns": "--max-turns",
}
MIN_GPU_GIB = 30  # in total; the model is sharded over every GPU
MIN_DISK_GIB = 60
WORK = Path("/tmp/validate") if is_kaggle() else Path("/content")
PORT = 8000
SERVER_TIMEOUT_SECONDS = 1800
POLL_SECONDS = 30
GOLD_BATCH = 8


def run(*command) -> None:
    subprocess.run([str(part) for part in command], check=True)


def kaggle_file(name: str, data_dir: Path) -> None:
    """Download one competition file; the CLI keeps its path under data_dir."""
    if not (data_dir / name).exists():
        run("kaggle", "competitions", "download", COMPETITION, "-f", name, "-p", data_dir, "-q")


def competition_files() -> list[str]:
    names: list[str] = []
    token = None
    while True:
        command = ["kaggle", "competitions", "files", COMPETITION, "--csv", "--page-size", "200"]
        command += ["--page-token", token] if token else []
        listing = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        names += [line.split(",")[0] for line in listing.splitlines() if "," in line]
        match = re.search(r"Next Page Token = (\S+)", listing)
        if not match:
            return names
        token = match.group(1)


def download(names: list[str], data_dir: Path) -> None:
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda name: kaggle_file(name, data_dir), names))


def prepare_data(data_dir: Path) -> None:
    if is_kaggle():
        return  # the competition is attached as an input
    names = [name for name in competition_files() if name.startswith("wheels/")]
    download([*names, "sandbox/setup.py"], data_dir)


def gpu_count() -> int:
    """Check the machine; returns how many GPUs vLLM shards the model over."""
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    gibs = [int(line) / 1024 for line in memory.splitlines() if line.strip()]
    print(f"GPU {len(gibs)}x {min(gibs):.0f} GiB", flush=True)
    if sum(gibs) < MIN_GPU_GIB:
        raise SystemExit("The GPUs are too small: select a Colab A100 or a Kaggle L4x4 machine.")
    return len(gibs)


def prepare_machine(model: str, data_dir: Path) -> tuple[Path, Path, Path]:
    """Build both environments and download the model."""
    WORK.mkdir(parents=True, exist_ok=True)
    disk_gib = shutil.disk_usage(WORK).free / 2**30
    print(f"free disk {disk_gib:.0f} GiB", flush=True)
    if disk_gib < MIN_DISK_GIB:
        raise SystemExit(f"About {MIN_DISK_GIB} GiB of free disk is needed.")

    run("apt-get", "update", "-qq")
    run("apt-get", "install", "-y", "-qq", "git", "patch", "pigz", "protobuf-compiler")
    run("git", "config", "--global", "user.email", "agent@eval")
    run("git", "config", "--global", "user.name", "Agent")
    run("git", "config", "--global", "init.defaultBranch", "main")
    run(sys.executable, "-m", "pip", "install", "-q", "uv", "huggingface_hub")

    wheels = data_dir / "wheelhouse"
    if is_kaggle():
        wheels = next(kaggle_input(WHEELHOUSE.split("/")[1]).rglob("swegemma-*.whl")).parent
    elif not any(wheels.glob("swegemma-*.whl")):
        run("kaggle", "datasets", "download", WHEELHOUSE, "-p", wheels, "--unzip", "-q")
    pins = ("uv", "pip", "install", "-q", "--exclude-newer", EXCLUDE_NEWER, "--find-links", wheels)

    root = WORK / "validate_env"
    server_python, eval_python = root / "server/bin/python", root / "eval/bin/python"
    if not server_python.exists():
        # The scoring wheels are built for Python 3.12.
        run("uv", "venv", "-q", "--python", "3.12", server_python.parents[1])
    run(*pins, "--python", server_python, *SERVER_PACKAGES)
    if not eval_python.exists():
        # The official sandbox runs Python 3.13, and tasks run with this interpreter.
        run("uv", "venv", "-q", "--python", "3.13", eval_python.parents[1])
    harness = [next(wheels.glob(f"{name}-*.whl")) for name in ("adk_submission", "adk_eval_core")]
    run(*pins, "--python", eval_python, *harness, *EVAL_PACKAGES)
    run("uv", "pip", "install", "-q", "--python", eval_python, "--no-deps",
        next(wheels.glob("swegemma-*.whl")))  # fmt: skip
    site = next((root / "eval/lib").glob("python*/site-packages"))
    for name in ("imp.py", "telnetlib.py"):
        kaggle_file(f"docker/{name}", data_dir)
        shutil.copy(data_dir / "docker" / name, site / name)

    from huggingface_hub import snapshot_download

    model_dir = WORK / "model_qat"
    snapshot_download(
        f"google/{model}",
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"],
        local_dir=model_dir,
    )
    return server_python, eval_python, model_dir


def declared(submission: Path, key: str) -> set[str]:
    pattern = re.compile(rf"^\s*{key}:\s*([\w.-]+)\s*$", re.MULTILINE)
    found: set[str] = set()
    for path in submission.rglob("*.yaml"):
        found.update(pattern.findall(path.read_text()))
    return found


def install_adapter(sources: list[Path], target: Path) -> bool:
    """Copy the first source that exists: a folder, or a zip holding the adapter files."""
    for source in sources:
        if (source / "adapter_model.safetensors").exists():
            shutil.copytree(source, target, dirs_exist_ok=True)
            return True
        if source.with_suffix(".zip").is_file():
            unpacked = target.with_name(f"{target.name}.unzipped")
            shutil.rmtree(unpacked, ignore_errors=True)
            zipfile.ZipFile(source.with_suffix(".zip")).extractall(unpacked)
            weights = next(unpacked.rglob("adapter_model.safetensors"), None)
            if weights is None:
                raise SystemExit(f"{source.with_suffix('.zip')} has no adapter_model.safetensors.")
            shutil.copytree(weights.parent, target, dirs_exist_ok=True)
            shutil.rmtree(unpacked)
            return True
    return False


def collect_adapters(exps: list[str], output: Path) -> dict[str, Path]:
    """Adapters are not in git: take each declared one from <output>/<exp>/ on Drive.

    A training run writes the adapter folder there; an uploaded <adapter>.zip also works.
    """
    adapters: dict[str, Path] = {}
    for exp in exps:
        submission = COMPETITION_DIR / "exp" / exp / "submission"
        for name in declared(submission, "adapter"):
            target = submission / "adapters" / name
            source = output / exp / name
            # on Kaggle, the training kernel's output is attached as an input
            sources = [source, *find_kaggle_inputs(f"{exp}/{name}")] if is_kaggle() else [source]
            ready = (target / "adapter_model.safetensors").exists()
            if not ready and not install_adapter(sources, target):
                raise SystemExit(f"{exp} declares adapter {name}: upload {source}/ or {source}.zip")
            print(f"{exp}: adapter {name} ready", flush=True)
            if name in adapters:
                raise SystemExit(f"Adapter name {name} is used by more than one experiment.")
            adapters[name] = target
    return adapters


def adapter_rank(path: Path) -> int:
    return json.loads((path / "adapter_config.json").read_text())["r"]


def start_server(
    server_python: Path, model_dir: Path, model: str, adapters: dict[str, Path], log: Path, gpus
) -> subprocess.Popen:
    """Start vLLM with the scoring flags from HARNESS_README (scoring uses 4 GPUs)."""
    command = [
        server_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--served-model-name", model,
        "--host", "127.0.0.1",
        "--port", PORT,
        "--tensor-parallel-size", gpus,
        "--max-model-len", 32768,
        "--gpu-memory-utilization", 0.90,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "gemma4",
        "--reasoning-parser", "gemma4",
        "--reasoning-config",
        json.dumps({"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}),
        "--no-scheduler-reserve-full-isl",
    ]  # fmt: skip
    if adapters:
        rank = max(adapter_rank(path) for path in adapters.values())
        command += ["--enable-lora", "--max-loras", len(adapters), "--max-lora-rank", rank]
        command += ["--lora-modules", *(f"{name}={path}" for name, path in adapters.items())]
    print(f"starting vLLM; log: {log}", flush=True)
    command = [str(part) for part in command]
    server = subprocess.Popen(command, stdout=log.open("w"), stderr=subprocess.STDOUT)
    deadline = time.time() + SERVER_TIMEOUT_SECONDS
    while time.time() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"vLLM exited with {server.returncode}:\n{log.read_text()[-3000:]}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            print("vLLM is ready", flush=True)
            return server
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    server.terminate()
    raise SystemExit(f"vLLM was not ready in time:\n{log.read_text()[-3000:]}")


def sample_tasks(tasks_path: Path, repos: list[str], seed: int) -> list[dict]:
    """A fixed order, so that a larger count extends a smaller one."""
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    rows = [row for row in rows if row["repo"] in repos]
    rows.sort(key=lambda row: row["instance_id"])
    random.Random(seed).shuffle(rows)
    return rows


def read_results(results_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for path in sorted(results_dir.glob("run_*/task_results.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            results[row["instance_id"]] = row
    return results


def eval_command(
    eval_python: Path, tasks_path: Path, data_dir: Path, submission: Path, run_dir: Path, workers
) -> list[str]:
    limits = yaml.safe_load((submission / "eval_config.yaml").read_text())["evaluation"]
    command = [
        eval_python, COMPETITION_DIR / "swegemma_eval.py",
        "--tasks", tasks_path,
        # the harness looks for wheels and sandbox/setup.py next to the snapshots
        "--snapshots-dir", data_dir / "snapshots",
        "--submission-dir", submission,
        "--results-dir", run_dir,
        "--sandbox", "subprocess",
        "--concurrency", workers,
        "--display", "quiet",
    ]  # fmt: skip
    for key, flag in EVAL_FLAGS.items():
        if key in limits:
            command += [flag, limits[key]]
    return [str(part) for part in command]


def eval_env(port: int) -> dict[str, str]:
    env = {**os.environ, "LOCAL_INFERENCE_URL": f"http://127.0.0.1:{port}/v1"}
    env.pop("PYTHONPATH", None)
    return env


def gold_server(patch: str) -> ThreadingHTTPServer:
    """A stand-in model that applies the reference patch, submits it, and stops."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            turn = sum(message["role"] == "assistant" for message in body["messages"])
            apply = f"cd /workspace && git apply <<'GOLD_PATCH'\n{patch}\nGOLD_PATCH"
            calls = [("run_command", {"command": apply}), ("submit_patch", {})]
            if turn < len(calls):
                name, arguments = calls[turn]
                function = {"name": name, "arguments": json.dumps(arguments)}
                call = {"id": f"call{turn}", "type": "function", "function": function}
                message, finish = (
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    "tool_calls",
                )
            else:
                message, finish = {"role": "assistant", "content": "Done."}, "stop"
            reply = {
                "id": "gold",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def check_gold(eval_python: Path, task: dict, data_dir: Path, gold_dir: Path, submission) -> None:
    run_dir = gold_dir / f"run_{task['instance_id']}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    tasks_path = run_dir / "tasks.jsonl"
    tasks_path.write_text(json.dumps(task) + "\n")
    server = gold_server(task["patch"])
    try:
        command = eval_command(eval_python, tasks_path, data_dir, submission, run_dir, 1)
        with (run_dir / "eval.log").open("w") as log:
            env = eval_env(server.server_address[1])
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    finally:
        server.shutdown()


def select_tasks(eval_python, candidates, count, data_dir, gold_dir, submission, workers) -> list:
    """Take candidates in order, keeping those whose reference patch passes locally."""
    selected: list[dict] = []
    excluded: list[str] = []
    for start in range(0, len(candidates), GOLD_BATCH):
        if len(selected) >= count:
            break
        batch = candidates[start : start + GOLD_BATCH]
        download([f"snapshots/{t['instance_id']}.tgz" for t in batch], data_dir)
        todo = [t for t in batch if t["instance_id"] not in read_results(gold_dir)]
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            jobs = [
                pool.submit(check_gold, eval_python, t, data_dir, gold_dir, submission)
                for t in todo
            ]
            for job in jobs:
                job.result()
        gold = read_results(gold_dir)
        for task in batch:
            row = gold.get(task["instance_id"])
            if row and row["resolved"]:
                selected.append(task)
            else:
                excluded.append(task["instance_id"])
        print(f"reference patches: {len(selected)} pass, {len(excluded)} fail", flush=True)
    if excluded:
        print(f"left out because the reference patch fails locally: {excluded}", flush=True)
    return selected[:count]


def evaluate_experiment(
    eval_python: Path, exp: str, sample: list[dict], data_dir: Path, results_dir: Path, args
) -> None:
    todo = [t for t in sample if t["instance_id"] not in read_results(results_dir)]
    if not todo:
        print(f"{exp}: every task is finished", flush=True)
        return
    run_number = len(list(results_dir.glob("run_*"))) + 1
    run_dir = results_dir / f"run_{run_number:02d}"
    run_dir.mkdir(parents=True)
    tasks_path = run_dir / "tasks.jsonl"
    tasks_path.write_text("".join(json.dumps(t) + "\n" for t in todo))

    submission = COMPETITION_DIR / "exp" / exp / "submission"
    command = eval_command(eval_python, tasks_path, data_dir, submission, run_dir, args.concurrency)
    env = eval_env(PORT)
    log = run_dir / "eval.log"
    print(f"{exp}: {len(todo)} tasks, concurrency {args.concurrency}; log: {log}", flush=True)
    began = time.time()
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        seen = 0
        while process.poll() is None:
            time.sleep(POLL_SECONDS)
            seen = report_progress(exp, run_dir, seen, len(todo), began)
    report_progress(exp, run_dir, seen, len(todo), began)
    if process.returncode != 0:
        raise SystemExit(f"{exp} failed:\n{log.read_text()[-3000:]}")


def report_progress(exp: str, run_dir: Path, seen: int, total: int, began: float) -> int:
    path = run_dir / "task_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    for row in rows[seen:]:
        state = "PASS" if row["resolved"] else "FAIL"
        note = f" ({row['error'][:60]})" if row.get("error") else ""
        print(
            f"{exp} {len(rows)}/{total} {row['instance_id']} {state} "
            f"{row['duration_seconds']:.0f}s elapsed={time.time() - began:.0f}s{note}",
            flush=True,
        )
    return max(seen, len(rows))


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided p-value for the tasks that only one of the two experiments solved."""
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(wins, losses) + 1)) / 2**total
    return min(1.0, 2 * tail)


def compare(exps: list[str], root: Path) -> dict:
    results = {exp: read_results(root / exp) for exp in exps}
    common = sorted(set.intersection(*(set(results[exp]) for exp in exps)))
    base = exps[0]
    summary = {"tasks": len(common), "experiments": {}}
    print(f"\ncompared on {len(common)} tasks present in every experiment")
    for exp in exps:
        rows = results[exp]
        resolved = sum(rows[t]["resolved"] for t in common)
        wins = [t for t in common if rows[t]["resolved"] and not results[base][t]["resolved"]]
        losses = [t for t in common if results[base][t]["resolved"] and not rows[t]["resolved"]]
        errors = sum(rows[t].get("error") is not None for t in common)
        seconds = sum(rows[t]["duration_seconds"] for t in common) / max(len(common), 1)
        p_value = sign_test(len(wins), len(losses))
        summary["experiments"][exp] = {
            "resolved": resolved,
            "errors": errors,
            "mean_seconds": seconds,
            "only_this": wins,
            "only_base": losses,
            "p_value": p_value,
        }
        line = f"{exp}: {resolved}/{len(common)} resolved, {errors} stopped early"
        line += f", {seconds:.0f}s/task"
        if exp != base:
            line += f"; vs {base}: +{len(wins)} -{len(losses)} (p={p_value:.3f})"
        print(line)
    (root / "comparison.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", required=True, help="Comma-separated; the first is the baseline.")
    parser.add_argument("--num-tasks", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repos", nargs="+", default=list(DEFAULT_REPOS))
    args = parser.parse_args()

    exps = args.exps.split(",")
    models = set()
    for exp in exps:
        models |= declared(COMPETITION_DIR / "exp" / exp / "submission", "model")
    if len(models) != 1:
        raise SystemExit(f"Experiments must share one model, got {sorted(models)}.")
    model = models.pop()

    paths = get_paths(COMPETITION)
    # v3: reference-patch check and extra test packages; v2 compacted sessions as scoring does
    root = paths.output / "validate" / f"v3_seed{args.seed}"
    if is_kaggle() and not root.exists():
        for previous in find_kaggle_inputs(f"validate/{root.name}")[:1]:
            print(f"continuing the results in {previous}", flush=True)
            shutil.copytree(previous, root)
    adapters = collect_adapters(exps, paths.output)

    kaggle_file("tasks.jsonl", paths.data)
    candidates = sample_tasks(paths.data / "tasks.jsonl", args.repos, args.seed)
    gpus = gpu_count()
    prepare_data(paths.data)
    server_python, eval_python, model_dir = prepare_machine(model, paths.data)
    submission = COMPETITION_DIR / "exp" / exps[0] / "submission"
    sample = select_tasks(
        eval_python, candidates, args.num_tasks, paths.data, root / "gold", submission,
        args.concurrency,
    )  # fmt: skip
    print(f"{len(sample)} tasks: {[t['instance_id'] for t in sample]}", flush=True)

    for exp in exps:
        (root / exp).mkdir(parents=True, exist_ok=True)
    server = start_server(server_python, model_dir, model, adapters, WORK / "vllm.log", gpus)
    try:
        for exp in exps:
            evaluate_experiment(eval_python, exp, sample, paths.data, root / exp, args)
    finally:
        server.terminate()
    compare(exps, root)
    print(f"results: {root}", flush=True)


if __name__ == "__main__":
    main()
