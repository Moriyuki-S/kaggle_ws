"""Train a LoRA adapter for Gemma 4 31B on a single Colab GPU from captured agent trajectories.

The checkpoint is loaded in nf4 so that it fits one GPU; the adapter is served later on the
official QAT checkpoint through vLLM. Progress is checkpointed so a new session can resume.

Run through notebooks/colab_runner.ipynb with DOWNLOAD and REQUIRE_SUBMISSION off.

Data: dmitriigluzdov/gemma-4-serving-and-lora-verification-kit (CC BY 4.0).
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

SEED = 42
R = 16
ALPHA = 16
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
TIERS = ("native", "external_A")
LEARNING_RATE = 1e-4
GRAD_ACCUM = 8
WARMUP_STEPS = 5
CHECKPOINT_EVERY = 5
EVAL_EVERY = 25
VAL_REPO_FRACTION = 0.1
MAX_VAL_STEPS = 32
SMOKE_LENGTHS = (2048, 4096, 6144, 8192)

SERVED_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
KEY_PREFIX = "base_model.model.model.language_model."
LAYER_RE = re.compile(r"layers\.(\d+)\.self_attn\.([qkvo]_proj)$")


EXP_DIR = Path(__file__).resolve().parent
EXP = EXP_DIR.name
COMPETITION = EXP_DIR.parents[1].name
KIT = "dmitriigluzdov/gemma-4-serving-and-lora-verification-kit"
KIT_FILES = ("lora_data.py", "lora_native_resolved.jsonl", "lora_external_A.jsonl")
SAMPLE_ADAPTER = "sample_submission/adapters/main_lora/adapter_model.safetensors"
PACKAGES = ("transformers==5.13.1", "bitsandbytes", "accelerate", "huggingface_hub", "six")
MIN_GPU_GIB = 20
MIN_DISK_GIB = 75


def prepare_colab(data_root: Path) -> tuple[Path, Path]:
    """Check the machine, then install packages and download only what training needs."""
    if not torch.cuda.is_available():
        raise SystemExit("No GPU: select an A100 runtime.")
    gpu = torch.cuda.get_device_properties(0)
    gpu_gib = gpu.total_memory / 2**30
    disk_gib = shutil.disk_usage("/content").free / 2**30
    print(f"GPU: {gpu.name} {gpu_gib:.0f} GiB; free disk: {disk_gib:.0f} GiB", flush=True)
    if gpu_gib < MIN_GPU_GIB:
        raise SystemExit(f"{gpu.name} is too small for the nf4 model: select an A100 runtime.")
    model_dir = Path("/content/model")
    if not (model_dir / "config.json").exists() and disk_gib < MIN_DISK_GIB:
        raise SystemExit(f"The checkpoint needs about 63 GB of disk; {disk_gib:.0f} GiB is free.")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        "google/gemma-4-31b-it",
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
        local_dir=model_dir,
    )
    data_dir = data_root / "lora"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in KIT_FILES:
        if not (data_dir / name).exists():
            command = ["kaggle", "datasets", "download", KIT, "-f", name, "-p", data_dir, "-q"]
            subprocess.run([str(part) for part in command], check=True)
    if not any(data_dir.rglob("adapter_model.safetensors")):
        command = ["kaggle", "competitions", "download", COMPETITION]
        command += ["-f", SAMPLE_ADAPTER, "-p", str(data_dir), "-q"]
        subprocess.run(command, check=True)
    return model_dir, data_dir


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(R, base.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, R, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = ALPHA / R

    def forward(self, x):
        delta = x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T
        return self.base(x) + delta * self.scaling


def load_model(model_dir: Path):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=quantization, device_map={"": 0}
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def load_steps(model_dir: Path, data_dir: Path, max_context: int):
    from transformers import AutoTokenizer

    sys.path.insert(0, str(next(data_dir.rglob("lora_data.py")).parent))
    from lora_data import iter_steps

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    root = next(data_dir.rglob("lora_data.py")).parent
    steps = []
    for step in iter_steps(root, tiers=TIERS, max_context_tokens=max_context):
        target = step["completion_token_ids"]
        if not target:
            continue
        prompt = step.get("prompt_token_ids")
        if prompt is None:
            prompt = tokenizer(step["prompt"], add_special_tokens=False)["input_ids"]
        steps.append(
            {
                "prompt": prompt,
                "target": target,
                "repo": step["issue"].rsplit("-", 1)[0],
                "tier": step["tier"],
            }
        )
    return steps


def split_by_repo(steps):
    repos = sorted({s["repo"] for s in steps if s["tier"] == "native"})
    random.Random(SEED).shuffle(repos)
    held_out = set(repos[: max(1, int(len(repos) * VAL_REPO_FRACTION))])
    train = [s for s in steps if s["repo"] not in held_out]
    val = [s for s in steps if s["repo"] in held_out]
    random.Random(SEED).shuffle(train)
    random.Random(SEED).shuffle(val)
    return train, val[:MAX_VAL_STEPS]


def wrap_modules(model) -> dict[str, LoraLinear]:
    wrapped: dict[str, LoraLinear] = {}
    for name, module in list(model.named_modules()):
        match = LAYER_RE.search(name)
        if not match or "vision" in name or "audio" in name or not isinstance(module, nn.Linear):
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        lora = LoraLinear(module)
        setattr(parent, match.group(2), lora)
        wrapped[f"layers.{match.group(1)}.self_attn.{match.group(2)}"] = lora
    if not wrapped:
        raise RuntimeError("no target modules found")
    return wrapped


def check_against_sample(wrapped: dict[str, LoraLinear], data_dir: Path) -> None:
    """The official sample adapter fixes tensor names and feature sizes that vLLM expects."""
    path = next(data_dir.rglob("adapter_model.safetensors"))
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    for key, info in header.items():
        if key == "__metadata__":
            continue
        layer, kind = key.removeprefix(KEY_PREFIX).rsplit(".", 2)[:2]
        axis = 0 if kind == "lora_B" else 1
        actual = getattr(wrapped[layer], kind).shape[axis]
        if actual != info["shape"][axis]:
            raise RuntimeError(f"{key}: sample {info['shape']}, model has size {actual}")


def save_adapter(wrapped: dict[str, LoraLinear], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer, lora in wrapped.items():
        for kind in ("lora_A", "lora_B"):
            value = getattr(lora, kind).detach().to("cpu", torch.bfloat16).clone()
            tensors[f"{KEY_PREFIX}{layer}.{kind}.weight"] = value
    save_file(tensors, directory / "adapter_model.safetensors")
    config = {
        "base_model_name_or_path": f"google/{SERVED_MODEL}",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": ALPHA,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": R,
        "target_modules": list(TARGET_MODULES),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2))


def peak_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2**30


def step_loss(model, step):
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([step["prompt"] + step["target"]], device=device)
    n = len(step["target"])
    logits = model(input_ids=input_ids, logits_to_keep=n + 1).logits[0, :-1]
    labels = torch.tensor(step["target"], device=logits.device)
    return F.cross_entropy(logits.float(), labels), n


@torch.no_grad()
def evaluate(model, subset) -> float:
    total, count = 0.0, 0
    for step in subset:
        loss, n = step_loss(model, step)
        total += loss.item() * n
        count += n
    return total / max(count, 1)


def smoke(model, train, params, optimizer) -> None:
    by_length = sorted(train, key=lambda s: len(s["prompt"]) + len(s["target"]))
    for goal in SMOKE_LENGTHS:
        step = min(by_length, key=lambda s: abs(len(s["prompt"]) + len(s["target"]) - goal))
        size = len(step["prompt"]) + len(step["target"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        begin = time.time()
        loss, _ = step_loss(model, step)
        loss.backward()
        print(
            f"smoke length={size} loss={loss.item():.4f} "
            f"seconds={time.time() - begin:.1f} peak_gib={peak_gib():.1f}",
            flush=True,
        )
        optimizer.zero_grad(set_to_none=True)
    print("smoke finished; no adapter was trained", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, help="Skips setup when given with --data-dir.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-context", type=int, default=6144)
    parser.add_argument("--time-limit-hours", type=float, default=10.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None or args.model_dir is None:
        from kaggle_ws import get_paths

        paths = get_paths(COMPETITION)
        args.output_dir = args.output_dir or paths.output / EXP
        if args.model_dir is None:
            args.model_dir, args.data_dir = prepare_colab(paths.data)

    random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    deadline = started + args.time_limit_hours * 3600
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.pt"

    context = 8192 if args.smoke else args.max_context
    steps = load_steps(args.model_dir, args.data_dir, context)
    train, val = split_by_repo(steps)
    print(f"steps: {len(steps)} train={len(train)} val={len(val)}", flush=True)

    model = load_model(args.model_dir)
    model.requires_grad_(False)
    model.config.use_cache = False
    model.train()
    print(f"{type(model).__name__} loaded at {time.time() - started:.0f}s", flush=True)
    print(f"peak after load: {peak_gib():.1f} GiB", flush=True)

    wrapped = wrap_modules(model)
    check_against_sample(wrapped, args.data_dir)
    print(f"wrapped {len(wrapped)} modules", flush=True)
    params = [p for lora in wrapped.values() for p in (lora.lora_A, lora.lora_B)]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.0)

    if args.smoke:
        smoke(model, train, params, optimizer)
        return

    total_updates = max(1, len(train) // GRAD_ACCUM)

    def lr_at(update: int) -> float:
        if update < WARMUP_STEPS:
            return (update + 1) / WARMUP_STEPS
        progress = (update - WARMUP_STEPS) / max(1, total_updates - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    update, skipped, history, best = 0, 0, [], None
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state["max_context"] != args.max_context:
            raise ValueError(f"checkpoint used max_context={state['max_context']}")
        for param, saved in zip(params, state["params"], strict=True):
            param.data.copy_(saved.to(param.device))
        optimizer.load_state_dict(state["optimizer"])
        update, skipped = state["update"], state["skipped"]
        history, best = state["history"], state["best"]
        print(f"resumed at update {update}/{total_updates}", flush=True)

    def write_checkpoint() -> None:
        state = {
            "max_context": args.max_context,
            "params": [p.detach().cpu() for p in params],
            "optimizer": optimizer.state_dict(),
            "update": update,
            "skipped": skipped,
            "history": history,
            "best": best,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)

    def write_metrics() -> None:
        metrics = {
            "max_context": args.max_context,
            "train_steps": len(train),
            "val_steps": len(val),
            "total_updates": total_updates,
            "updates": update,
            "skipped_steps": skipped,
            "best_val_nll": best,
            "history": history,
            "config": {"r": R, "alpha": ALPHA, "lr": LEARNING_RATE, "tiers": list(TIERS)},
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if best is None:
        best = evaluate(model, val)
        history.append({"update": 0, "val_nll": best})
        print(f"baseline val_nll={best:.4f} ({time.time() - started:.0f}s)", flush=True)

    while update < total_updates and time.time() < deadline:
        batch = train[update * GRAD_ACCUM : (update + 1) * GRAD_ACCUM]
        begin = time.time()
        running, seen = 0.0, 0
        for step in batch:
            try:
                loss, _ = step_loss(model, step)
                (loss / len(batch)).backward()
                running += loss.item()
                seen += 1
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                torch.cuda.empty_cache()
                print(f"skipped a step after OOM (total {skipped})", flush=True)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * lr_at(update)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        print(
            f"update {update}/{total_updates} loss={running / max(seen, 1):.4f} grad={norm:.3f} "
            f"s/update={time.time() - begin:.0f} peak_gib={peak_gib():.1f} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
        if update % EVAL_EVERY == 0 or update == total_updates:
            nll = evaluate(model, val)
            history.append({"update": update, "val_nll": nll})
            print(f"val_nll={nll:.4f} (best {best:.4f})", flush=True)
            if nll < best:
                best = nll
                save_adapter(wrapped, args.output_dir / "main_lora")
        if update % CHECKPOINT_EVERY == 0 or update == total_updates:
            write_checkpoint()
            write_metrics()

    write_checkpoint()
    write_metrics()
    save_adapter(wrapped, args.output_dir / "main_lora_last")
    state = (
        "finished" if update == total_updates else "stopped at the time limit; run again to resume"
    )
    print(f"{state}: update {update}/{total_updates}, best val_nll={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
