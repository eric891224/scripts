# TP1 訓練環境

[回 training manuals](../../README.md)

此環境不依賴 `sm-dp`。資料在本機預處理後，上傳完整的 saved Dataset 目錄即可。

## 安裝

以下指令從 workspace 根目錄執行，使用 Python 3.12 以上：

```bash
uv sync --project scripts/training/envs/tp1 --locked
```

一般與 Slurm launcher 都預設使用這裡的 `.venv/bin/python`，不必先 activate。使用其他環境時才設定 `PYTHON_BIN=/absolute/path/to/python`。

目前 lockfile 包含 Transformers、TRL、W&B，以及 TRL 帶入的 Datasets、PyTorch、Accelerate。套件版本沿用現有 lockfile；GPU 驅動與 CUDA 相容性仍需在 server 確認。

使用 Slurm／DeepSpeed 時，另於相同環境安裝：

```bash
uv pip install --python scripts/training/envs/tp1/.venv/bin/python deepspeed
```

DeepSpeed 尚未納入 lockfile；請先 sync 再安裝，重新 sync 可能移除額外套件。部署時另記錄實際 DeepSpeed 版本。

## 選用：個人環境變數

```bash
cd scripts/training/envs/tp1
cp src/tp1/.env.example.sh .env.sh
# 編輯 .env.sh 後載入
source .env.sh
```

不要在此檔案保存 API key。回到 workspace 根目錄後，依 [Quickstart](../../docs/quickstart.md) 或 [Slurm manual](../../docs/slurm.md) 執行。
