# kaggle_ws

ローカルでコードを書き、GitHub に push し、Colab で clone して実行する Kaggle 用ワークスペース。

## ディレクトリ構成

```
kaggle_ws/
├── src/kaggle_ws/          # コンペ共通のユーティリティ (pip install -e . で import 可能)
│   ├── paths.py            #   ローカル / Colab / Kaggle でデータ・出力パスを自動切替
│   └── utils.py            #   seed_everything, timer など
├── competitions/
│   ├── _template/          # 新コンペ用のひな形
│   └── <competition>/
│       ├── README.md       #   概要・実験ログ (CV / LB)
│       ├── notebooks/      #   EDA 用
│       └── exp/
│           ├── exp001/train.py
│           └── exp002/train.py
├── notebooks/
│   └── colab_runner.ipynb  # Colab 実行用テンプレート
├── scripts/
│   ├── new_comp.sh         # コンペディレクトリ作成
│   ├── new_exp.sh          # 実験ディレクトリを複製して番号 +1
│   └── download.sh         # kaggle CLI でデータ取得・解凍
├── data/                   # (git 管理外) ローカル用データ
└── outputs/                # (git 管理外) ローカル用出力
```

- 1 実験 = 1 ディレクトリ。過去の実験は書き換えずにコピーして次へ進む (再現性のため)。
- コンペ名・実験名はディレクトリ名から自動で決まり、出力は `outputs/<competition>/<exp>/` に保存される。
- Colab では出力先が `MyDrive/kaggle/outputs/<competition>/<exp>/` になる (Drive をマウントした場合)。

## ローカルのセットアップ

```bash
uv sync                      # .venv 作成 + 依存インストール (kaggle CLI 含む)
source .venv/bin/activate    # または各コマンドの頭に `uv run` を付ける
```

Kaggle API の認証 (どちらか):

- 新方式: Kaggle > Settings > API > **Generate New Token** → `export KAGGLE_API_TOKEN=...` (または `~/.kaggle/access_token` に保存)
- 旧方式: **Create Legacy API Key** で落ちる `kaggle.json` を `~/.kaggle/kaggle.json` に置き `chmod 600`

```bash
kaggle competitions list     # 動作確認
```

## 使い方

```bash
# 1. コンペを作成 (先に Kaggle のコンペページでルールに同意しておく)
scripts/new_comp.sh titanic

# 2. ローカルにデータを落として小さく動作確認 (任意)
scripts/download.sh titanic
uv run python competitions/titanic/exp/exp001/train.py --debug

# 3. push
git add -A && git commit -m "titanic exp001" && git push

# 4. Colab で notebooks/colab_runner.ipynb を開き、設定セルを書き換えて上から実行
#    (GitHub から直接開く: https://colab.research.google.com/github/Moriyuki-S/kaggle_ws/blob/main/notebooks/colab_runner.ipynb)

# 次の実験
scripts/new_exp.sh titanic   # exp001 -> exp002 にコピー
```

## Colab の初回設定

左サイドバーの 🔑 シークレットに登録し、ノートブックからのアクセスを ON にする:

- `KAGGLE_API_TOKEN` (旧方式なら `KAGGLE_USERNAME` と `KAGGLE_KEY`)
- `GITHUB_TOKEN`: private リポジトリの場合のみ (Fine-grained PAT、Contents: Read-only)
