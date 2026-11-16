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
| exp001 | 31B single agent, no LoRA, no graph tools, 8 min / 28 calls | 8/30 | | 409 s/task, over the ~350 s budget |
| exp002 | Kaggle L4x4 memory and speed test for LoRA training (bf16); 12k context ran out of memory | | | smoke only |
| exp003 | LoRA (nf4, r=16, alpha=16, native + external A, 6144 context) on exp001's config | 7/30 | | val_nll 0.471 to 0.361; 7 tasks hit the 80-turn limit |
| exp005 | exp001 with a 5 min cap and a prompt that edits early | 7/30 | | 277 s/task; 18 tasks hit the time limit |
| exp006 | exp005 with prompt rules to write code raw in tool arguments, an edit_file example and recovery from a missing old_string | | | prompt-only fix for the broken edit_file calls |
| exp007 | LoRA on clean edit steps (native, external A and B; 8192 context, histories trimmed) on exp005's config | | | targets edit_file calls that lose old_string |

CV: `validate.py`, 30 public tasks (seed 42) on Kaggle 4x L4, tasks on Python 3.12. About a third of
edit_file calls in exp001 and exp005 lost `old_string`: tool results reach the model JSON-escaped, it
writes `\n` in new_string, and the broken string swallows the next argument.

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

`notebooks/colab_validate.ipynb` (Colab A100) or `notebooks/kaggle/validate` (Kaggle 4x L4, see below) compares experiments on the public tasks without using a Kaggle submission. On Colab, select an A100 runtime, set `EXPS = "exp001,exp003"` (the first is the baseline), and run all. Keep `ARGS = "--num-tasks 2"` to check the setup, then clear it for 30 tasks.

`validate.py` serves the official QAT model and every adapter with vLLM, then runs `swegemma eval --sandbox subprocess` through `swegemma_eval.py`, which adds the event compaction and prompt caching that scoring uses (without them, long sessions fail with `ContextWindowExceededError`). It takes limits from each `eval_config.yaml` and each adapter from `<exp>/<adapter>/` (or `<exp>/<adapter>.zip`) in the Drive folder. Results go to `validate/v3_seed<seed>/` in the Drive folder, and running again skips finished tasks.

Before any model runs, each sampled task is run with its reference patch (a stand-in model applies it), and tasks that fail even then are replaced by the next ones: the local sandbox lacks some test dependencies of the official image (for example `inline_snapshot`, or the `pygments` version that `rich` tests expect).

Trust the difference between experiments rather than the absolute rate: this differs from scoring in network access and memory limits (and, on Colab, in GPU layout). `psf/requests` tasks are excluded because their tests need `pytest-httpbin`, which the published wheels do not include.

## LoRA training on Colab (exp003)

Use `notebooks/colab_runner.ipynb` on an A100 runtime with `KAGGLE_API_TOKEN` in Colab Secrets. Set `COMPETITION = "gemma-4-developer-agent"` and `EXP = "exp003"`, then run all. The default arguments train with `--max-context 6144 --time-limit-hours 10`; use `ARGS = "--smoke"` to measure speed and memory only. L4 (22 GiB) is too small: every step of a real run ran out of memory, so the script refuses GPUs under 30 GiB.

The `.skip_download` file stops the runner from downloading the full competition data (about 22 GB), so `train.py` fetches only what it needs.

Running again resumes from `checkpoint.pt` on Drive. Training data comes from `dmitriigluzdov/gemma-4-serving-and-lora-verification-kit` (CC BY 4.0).

## Kaggle 4x L4 (when Colab compute units run out)

Kaggle offers 4x L4 (the scoring hardware) only to kernels without internet, so everything comes from attached inputs. The kernels run as private scripts on the competition's Docker image (`kernel-metadata.json`), which has Python 3.12 and torch 2.10, the version the wheelhouse's vLLM needs. They count against the weekly Kaggle GPU quota, a session ends after 12 hours, and the 4x L4 queue can take an hour or more to start.

- Code: `scripts/kaggle_code.sh` uploads the committed `src/` and `competitions/` to the private dataset `kaggle-ws-code`. Run it after every commit that a kernel should use.
- Evaluation environment: `kaggle_env.py` uploads the evaluation wheels, resolved for Linux and Python 3.12 with the same cutoff as Colab, to the private dataset `kaggle-ws-eval-wheels`. Run it again only when `EVAL_PACKAGES` changes. Tasks run on Python 3.12 here, while the scoring sandbox runs 3.13: Kaggle never finished creating a dataset that held a CPython 3.13 build.
- QAT model: `thomasplocher/gemma-4-31b-w4a16-ct-mirror`, whose `model.safetensors` and `tokenizer.json` match the Hugging Face sha256 (checked 2026-11-14).
- Training: edit `EXP` and `ARGS` in `notebooks/kaggle/train/run.py` (start with `--smoke`), then `uv run kaggle kernels push -p notebooks/kaggle/train`. transformers and bitsandbytes come from the wheelhouse; the nf4 model is split layer by layer over the four GPUs. Outputs go to `/kaggle/working/<exp>/`. To resume, attach the kernel's previous output as an input: `train.py` copies `<exp>/checkpoint.pt` and the files next to it from there.
- Validation: edit `ARGS` in `notebooks/kaggle/validate/run.py`, then `uv run kaggle kernels push -p notebooks/kaggle/validate`. vLLM runs on the image's Python with the wheelhouse installed and shards the model over the four GPUs (`--tensor-parallel-size 4`) as scoring does. Adapters come from any attached input holding `<exp>/<adapter>/` (the training kernel's output, or a dataset made from the Drive folder). Results go to `/kaggle/working/validate/`; attach the previous output to continue them.
