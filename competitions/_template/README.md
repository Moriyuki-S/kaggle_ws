# COMPETITION_NAME

- URL: https://www.kaggle.com/competitions/COMPETITION_NAME
- Metric:
- Deadline:

## Layout

```
COMPETITION_NAME/
├── README.md        # overview and experiment log
├── notebooks/       # throwaway notebooks (EDA etc.)
└── exp/
    ├── exp001/      # one experiment = one directory (copy to start the next)
    │   └── train.py
    └── exp002/
```

New experiment: `scripts/new_exp.sh COMPETITION_NAME` (copies the latest exp and bumps the number)

## Experiment log

| exp | description | CV | Public LB | notes |
|-----|-------------|----|-----------|-------|
| exp001 | baseline | | | |
