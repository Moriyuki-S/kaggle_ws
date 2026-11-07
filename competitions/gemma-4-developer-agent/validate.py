"""Compare experiments on the public tasks with the official harness on one Colab GPU.

Run through notebooks/colab_validate.ipynb. One vLLM server serves the official QAT model and
every adapter, then `swegemma eval` runs each experiment on the same sample of tasks.
Finished tasks are kept on Drive, so running again continues where it stopped.
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
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from kaggle_ws import get_paths

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
)
# requests tests need pytest-httpbin, which the published wheels do not include.
DEFAULT_REPOS = ("fastapi/fastapi", "Textualize/rich")
EVAL_FLAGS = {
    "timeout_seconds": "--timeout-seconds",
    "max_tool_calls": "--max-tool-calls",
    "max_time_minutes": "--max-time-minutes",
    "max_turns": "--max-turns",
}
MIN_GPU_GIB = 30
MIN_DISK_GIB = 60
PORT = 8000
SERVER_TIMEOUT_SECONDS = 1800
POLL_SECONDS = 30


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


def prepare_data(data_dir: Path, task_ids: list[str]) -> None:
    names = [name for name in competition_files() if name.startswith("wheels/")]
    names += ["sandbox/setup.py"]
    names += [f"snapshots/{task_id}.tgz" for task_id in task_ids]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda name: kaggle_file(name, data_dir), names))


def prepare_machine(model: str, data_dir: Path) -> tuple[Path, Path, Path]:
    """Check the machine, then build both environments and download the model."""
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    gpu_gib = int(memory.splitlines()[0]) / 1024
    disk_gib = shutil.disk_usage("/content").free / 2**30
    print(f"GPU {gpu_gib:.0f} GiB; free disk {disk_gib:.0f} GiB", flush=True)
    if gpu_gib < MIN_GPU_GIB:
        raise SystemExit("This GPU is too small: select an A100 runtime.")
    if disk_gib < MIN_DISK_GIB:
        raise SystemExit(f"About {MIN_DISK_GIB} GiB of free disk is needed.")

    run("apt-get", "update", "-qq")
    run("apt-get", "install", "-y", "-qq", "git", "patch", "pigz", "protobuf-compiler")
    run("git", "config", "--global", "user.email", "agent@eval")
    run("git", "config", "--global", "user.name", "Agent")
    run("git", "config", "--global", "init.defaultBranch", "main")
    run(sys.executable, "-m", "pip", "install", "-q", "uv", "huggingface_hub")

    wheels = data_dir / "wheelhouse"
    if not any(wheels.glob("swegemma-*.whl")):
        run("kaggle", "datasets", "download", WHEELHOUSE, "-p", wheels, "--unzip", "-q")
    pins = ("uv", "pip", "install", "-q", "--exclude-newer", EXCLUDE_NEWER, "--find-links", wheels)

    root = Path("/content/validate_env")
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

    model_dir = Path("/content/model_qat")
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


def collect_adapters(exps: list[str], output: Path) -> dict[str, Path]:
    """Adapters are not in git: copy each declared one from the experiment's output on Drive."""
    adapters: dict[str, Path] = {}
    for exp in exps:
        submission = COMPETITION_DIR / "exp" / exp / "submission"
        for name in declared(submission, "adapter"):
            target = submission / "adapters" / name
            if not (target / "adapter_model.safetensors").exists():
                source = output / exp / name
                if not (source / "adapter_model.safetensors").exists():
                    raise SystemExit(f"{exp} declares adapter {name}, but {source} has no weights.")
                shutil.copytree(source, target, dirs_exist_ok=True)
            if name in adapters:
                raise SystemExit(f"Adapter name {name} is used by more than one experiment.")
            adapters[name] = target
    return adapters


def adapter_rank(path: Path) -> int:
    return json.loads((path / "adapter_config.json").read_text())["r"]


def start_server(
    server_python: Path, model_dir: Path, model: str, adapters: dict[str, Path], log: Path
) -> subprocess.Popen:
    """Start vLLM with the scoring flags from HARNESS_README, on one GPU instead of four."""
    command = [
        server_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--served-model-name", model,
        "--host", "127.0.0.1",
        "--port", PORT,
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


def sample_tasks(tasks_path: Path, repos: list[str], count: int, seed: int) -> list[dict]:
    """A larger count extends a smaller one, so earlier results stay in the sample."""
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    rows = [row for row in rows if row["repo"] in repos]
    rows.sort(key=lambda row: row["instance_id"])
    random.Random(seed).shuffle(rows)
    return rows[:count]


def read_results(results_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for path in sorted(results_dir.glob("run_*/task_results.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            results[row["instance_id"]] = row
    return results


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
    limits = yaml.safe_load((submission / "eval_config.yaml").read_text())["evaluation"]
    command = [
        eval_python, "-m", "swegemma", "eval",
        "--tasks", tasks_path,
        # the harness looks for wheels and sandbox/setup.py next to the snapshots
        "--snapshots-dir", data_dir / "snapshots",
        "--submission-dir", submission,
        "--results-dir", run_dir,
        "--sandbox", "subprocess",
        "--concurrency", args.concurrency,
        "--display", "quiet",
    ]  # fmt: skip
    for key, flag in EVAL_FLAGS.items():
        if key in limits:
            command += [flag, limits[key]]

    env = {**os.environ, "LOCAL_INFERENCE_URL": f"http://127.0.0.1:{PORT}/v1"}
    env.pop("PYTHONPATH", None)
    log = run_dir / "eval.log"
    print(f"{exp}: {len(todo)} tasks, concurrency {args.concurrency}; log: {log}", flush=True)
    began = time.time()
    with log.open("w") as stream:
        process = subprocess.Popen(
            [str(part) for part in command], stdout=stream, stderr=subprocess.STDOUT, env=env
        )
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
    root = paths.output / "validate" / f"seed{args.seed}"
    adapters = collect_adapters(exps, paths.output)

    kaggle_file("tasks.jsonl", paths.data)
    sample = sample_tasks(paths.data / "tasks.jsonl", args.repos, args.num_tasks, args.seed)
    print(f"{len(sample)} tasks: {[t['instance_id'] for t in sample]}", flush=True)
    prepare_data(paths.data, [t["instance_id"] for t in sample])
    server_python, eval_python, model_dir = prepare_machine(model, paths.data)

    for exp in exps:
        (root / exp).mkdir(parents=True, exist_ok=True)
    server = start_server(server_python, model_dir, model, adapters, Path("/content/vllm.log"))
    try:
        for exp in exps:
            evaluate_experiment(eval_python, exp, sample, paths.data, root / exp, args)
    finally:
        server.terminate()
    compare(exps, root)
    print(f"results: {root}", flush=True)


if __name__ == "__main__":
    main()
