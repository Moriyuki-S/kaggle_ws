# gemma-4-developer-agent

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent
- Metric: resolution rate (share of tasks whose hidden tests pass after applying the patch)
- Deadline: entry and team merger 2026-11-25, final submission 2026-12-02 (23:59 UTC)

## Layout

```
gemma-4-developer-agent/
├── README.md        # overview and experiment log
├── notebooks/       # throwaway notebooks (EDA etc.)
└── exp/
    └── exp001/      # one experiment = one directory (copy to start the next)
        ├── build_submission.py  # validates submission/ and writes outputs/<exp>/submission.zip
        └── submission/          # agent config that is zipped
```

New experiment: `scripts/new_exp.sh gemma-4-developer-agent` (copies the latest exp and bumps the number)

## Experiment log

| exp | description | CV | Public LB | notes |
|-----|-------------|----|-----------|-------|
| exp001 | 31B single agent, no LoRA, no graph tools, 8 min / 28 calls | | | |
| exp002 | Kaggle L4x4 memory and speed test for LoRA training (bf16); 12k context ran out of memory | | | smoke only |
| exp003 | Colab LoRA training (nf4, r=16, alpha=16, native + external A) | | | |

## Setup

The harness packages are not on PyPI. They come from the `metric/gemma-4-developer-agent-wheelhouse` dataset.

```bash
uv venv --python 3.13 .venv
W=data/harness_wheels
uv pip install --python .venv/bin/python $W/adk_submission-*.whl $W/adk_eval_core-*.whl litellm pandas networkx cachetools python-dotenv
uv pip install --python .venv/bin/python --no-deps $W/swegemma-*.whl
.venv/bin/python exp/exp001/build_submission.py
```

## Drive folder

On Colab, everything goes to `MyDrive/Colab Notebooks/Kaggle/gemma-4-developer-agent/`: `<exp>/` holds a training run's checkpoint, metrics and adapters, and `validate/` holds validation results.

## Validation on Colab

`notebooks/colab_validate.ipynb` compares experiments on the public tasks without using a Kaggle submission. Select an A100 runtime, set `EXPS = "exp001,exp003"` (the first is the baseline), and run all. Keep `ARGS = "--num-tasks 2"` to check the setup, then clear it for 30 tasks.

`validate.py` serves the official QAT model and every adapter with vLLM, then runs `swegemma eval --sandbox subprocess` through `swegemma_eval.py`, which adds the event compaction and prompt caching that scoring uses (without them, long sessions fail with `ContextWindowExceededError`). It takes limits from each `eval_config.yaml` and each adapter from `<exp>/<adapter>/` (or `<exp>/<adapter>.zip`) in the Drive folder. Results go to `validate/v2_seed<seed>/` in the Drive folder, and running again skips finished tasks.

Trust the difference between experiments rather than the absolute rate: this differs from scoring in GPU layout, network access, and memory limits. `psf/requests` tasks are excluded because their tests need `pytest-httpbin`, which the published wheels do not include.

## LoRA training on Colab (exp003)

Use `notebooks/colab_runner.ipynb` on an A100 runtime with `KAGGLE_API_TOKEN` in Colab Secrets. Set `COMPETITION = "gemma-4-developer-agent"` and `EXP = "exp003"`, then run all. The default arguments train with `--max-context 6144 --time-limit-hours 10`; use `ARGS = "--smoke"` to measure speed and memory only. L4 (22 GiB) is too small: every step of a real run ran out of memory, so the script refuses GPUs under 30 GiB.

The `.skip_download` file stops the runner from downloading the full competition data (about 22 GB), so `train.py` fetches only what it needs.

Running again resumes from `checkpoint.pt` on Drive. Training data comes from `dmitriigluzdov/gemma-4-serving-and-lora-verification-kit` (CC BY 4.0).
