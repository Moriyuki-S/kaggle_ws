"""Train a LoRA adapter for Gemma 4 31B on Kaggle L4x4 from captured agent trajectories.

Runs as a private Kaggle script (see kernel-metadata.json). Weights are loaded in bf16
because PEFT-style layers cannot attach to the packed w4a16 serving checkpoint; the adapter
is served on the official QAT checkpoint through vLLM.

Data: dmitriigluzdov/gemma-4-serving-and-lora-verification-kit (CC BY 4.0).
"""

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

SMOKE = True  # True: time one training step per context length, to check memory and speed.

SEED = 42
R = 16
ALPHA = 16
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
TIERS = ("native", "external_A")
MAX_CONTEXT_TOKENS = 16384
LEARNING_RATE = 1e-4
GRAD_ACCUM = 8
EPOCHS = 1
WARMUP_STEPS = 10
EVAL_EVERY = 50
VAL_REPO_FRACTION = 0.1
MAX_VAL_STEPS = 64
TIME_LIMIT_HOURS = 8.0
MAX_MEMORY = "17GiB"
SMOKE_LENGTHS = (2048, 4096, 8192, 12288, 16384)

SERVED_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
LAYER_RE = re.compile(r"layers\.(\d+)\.self_attn\.([qkvo]_proj)$")


def install_wheelhouse() -> None:
    """Install the competition's pinned transformers stack, as the official starter does."""
    source = next(INPUT.rglob("adk_submission-*.whl")).parent
    staging = Path("/tmp/wheelhouse")
    staging.mkdir(parents=True, exist_ok=True)
    wheels = []
    for wheel in source.glob("*.whl"):
        if "cutlass" in wheel.name.lower():
            continue
        name = wheel.name
        if "cu128" in name and "+" not in name:
            name = name.replace("cu128", "+cu128")
        target = staging / name
        if not target.exists():
            shutil.copyfile(wheel, target)
        wheels.append(str(target))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels],
        check=True,
    )


def find_model_dir() -> Path:
    candidates = [p.parent for p in INPUT.rglob("config.json") if "gemma-4-31b-it" in str(p)]
    candidates = [p for p in candidates if "qat" not in str(p)]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one bf16 31B checkpoint, found {candidates}")
    return candidates[0]


def load_steps(tokenizer):
    root = next(INPUT.rglob("lora_data.py")).parent
    sys.path.insert(0, str(root))
    from lora_data import iter_steps

    steps = []
    for step in iter_steps(root, tiers=TIERS, max_context_tokens=MAX_CONTEXT_TOKENS):
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
    random.Random(SEED).shuffle(val)
    return train, val[:MAX_VAL_STEPS]


def read_sample_shapes() -> dict[str, list[int]]:
    """Tensor shapes of the official sample adapter, used to check compatibility with vLLM."""
    path = next(INPUT.rglob("sample_submission/adapters/main_lora/adapter_model.safetensors"))
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}


def main() -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if SMOKE:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    install_wheelhouse()

    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    model_dir = find_model_dir()
    print(f"model: {model_dir}; gpus: {torch.cuda.device_count()}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    steps = load_steps(tokenizer)
    train, val = split_by_repo(steps)
    print(f"steps: {len(steps)} train={len(train)} val={len(val)}", flush=True)

    memory = {i: MAX_MEMORY for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map="auto", max_memory=memory
    )
    print(f"{type(model).__name__} loaded in {time.time() - started:.0f}s", flush=True)
    model.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

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
    print(f"wrapped {len(wrapped)} modules", flush=True)

    prefix = "base_model.model.model.language_model."
    for key, shape in read_sample_shapes().items():
        layer = key.removeprefix(prefix).rsplit(".", 2)[0]
        kind = key.rsplit(".", 2)[1]
        actual = list(getattr(wrapped[layer], kind).shape)
        if actual[0 if kind == "lora_B" else 1] != shape[0 if kind == "lora_B" else 1]:
            raise RuntimeError(f"{key}: sample {shape}, model {actual}")

    def save_adapter(directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for layer, lora in wrapped.items():
            for kind in ("lora_A", "lora_B"):
                value = getattr(lora, kind).detach().to("cpu", torch.bfloat16).clone()
                tensors[f"{prefix}{layer}.{kind}.weight"] = value
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

    def step_loss(step):
        ids = step["prompt"] + step["target"]
        device = model.get_input_embeddings().weight.device
        input_ids = torch.tensor([ids], device=device)
        n = len(step["target"])
        logits = model(input_ids=input_ids, logits_to_keep=n + 1).logits[0, :-1]
        labels = torch.tensor(step["target"], device=logits.device)
        return F.cross_entropy(logits.float(), labels), n

    @torch.no_grad()
    def evaluate(subset) -> float:
        total, count = 0.0, 0
        for step in subset:
            loss, n = step_loss(step)
            total += loss.item() * n
            count += n
        return total / max(count, 1)

    if SMOKE:
        val = val[:2]
    else:
        random.Random(SEED).shuffle(train)

    params = [p for lora in wrapped.values() for p in (lora.lora_A, lora.lora_B)]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.0)
    total_updates = max(1, EPOCHS * len(train) // GRAD_ACCUM)

    def lr_at(update: int) -> float:
        if update < WARMUP_STEPS:
            return (update + 1) / WARMUP_STEPS
        progress = (update - WARMUP_STEPS) / max(1, total_updates - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    best = evaluate(val)
    baseline = best
    history = [{"update": 0, "val_nll": best}]
    print(f"baseline val_nll={best:.4f} elapsed={time.time() - started:.0f}s", flush=True)
    save_adapter(WORKING / "main_lora")

    if SMOKE:
        lengths = sorted(train, key=lambda s: len(s["prompt"]) + len(s["target"]))
        for goal in SMOKE_LENGTHS:
            step = min(lengths, key=lambda s: abs(len(s["prompt"]) + len(s["target"]) - goal))
            size = len(step["prompt"]) + len(step["target"])
            print(f"smoke start length={size} target={len(step['target'])}", flush=True)
            for i in memory:
                torch.cuda.reset_peak_memory_stats(i)
            begin = time.time()
            try:
                loss, _ = step_loss(step)
                loss.backward()
            except RuntimeError:
                subprocess.run(["nvidia-smi"], check=False)
                raise
            peak = max(torch.cuda.max_memory_allocated(i) / 2**30 for i in memory)
            print(
                f"smoke done length={size} loss={loss.item():.4f} "
                f"seconds={time.time() - begin:.1f} peak_gib={peak:.1f}",
                flush=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        counts = [len(s["prompt"]) + len(s["target"]) for s in train]
        for goal in SMOKE_LENGTHS:
            print(f"train steps <= {goal}: {sum(c <= goal for c in counts)}", flush=True)
        save_adapter(WORKING / "main_lora_final")
        return

    update, running, seen, step_started = 0, 0.0, 0, time.time()
    deadline = started + TIME_LIMIT_HOURS * 3600
    stop = False
    for epoch in range(EPOCHS):
        for index, step in enumerate(train):
            loss, _ = step_loss(step)
            (loss / GRAD_ACCUM).backward()
            running += loss.item()
            seen += 1
            if (index + 1) % GRAD_ACCUM:
                continue
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATE * lr_at(update)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            rate = (time.time() - step_started) / seen
            peak = [torch.cuda.max_memory_allocated(i) / 2**30 for i in memory]
            print(
                f"update {update}/{total_updates} loss={running / seen:.4f} grad={norm:.3f} "
                f"s/step={rate:.1f} peak_gib={max(peak):.1f} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )
            running, seen, step_started = 0.0, 0, time.time()
            if update % EVAL_EVERY == 0 or update == total_updates or time.time() > deadline:
                nll = evaluate(val)
                history.append({"update": update, "val_nll": nll})
                print(f"val_nll={nll:.4f} (best {best:.4f})", flush=True)
                if nll < best:
                    best = nll
                    save_adapter(WORKING / "main_lora")
            if time.time() > deadline:
                stop = True
                break
        if stop:
            break

    save_adapter(WORKING / "main_lora_final")
    metrics = {
        "smoke": SMOKE,
        "train_steps": len(train),
        "val_steps": len(val),
        "updates": update,
        "baseline_val_nll": baseline,
        "best_val_nll": best,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "config": {"r": R, "alpha": ALPHA, "lr": LEARNING_RATE, "tiers": TIERS},
    }
    (WORKING / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
