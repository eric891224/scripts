# TP1 訓練環境

[回 training manuals](../../README.md)

此環境不依賴 `sm-dp`。資料在本機預處理後，上傳完整的 saved Dataset 目錄即可。

## 安裝

以下指令從 workspace 根目錄執行，使用 Python 3.12 以上：

```bash
uv sync --project scripts/training/envs/tp1 --locked
```

此處的設定範本將 `PYTHON_BIN` 預設為這裡的 `.venv/bin/python`，不必先 activate。使用其他環境時設定 `PYTHON_BIN=/absolute/path/to/python`；Slurm script 本身不依賴 TP1 或此目錄。

目前 lockfile 包含 Transformers、TRL、W&B，以及 TRL 帶入的 Datasets、PyTorch、Accelerate。套件版本沿用現有 lockfile；GPU 驅動與 CUDA 相容性仍需在 server 確認。

使用 Slurm／DeepSpeed 時，另於相同環境安裝：

```bash
uv pip install --python scripts/training/envs/tp1/.venv/bin/python deepspeed
```

DeepSpeed 尚未納入 lockfile；請先 sync 再安裝，重新 sync 可能移除額外套件。部署時另記錄實際 DeepSpeed 版本。

TP1 範本明確設定 `CC=/cm/local/apps/gcc/14.2.0/bin/gcc`、`CXX=/cm/local/apps/gcc/14.2.0/bin/g++`，覆寫 cluster 繼承的 NVHPC 編譯器：此次 server 測試中，Triton CUDA 輔助模組使用 `nvc` 編譯失敗，改用 GCC 後成功。這是站台工具鏈設定，不代表所有 TP1／Triton 環境都有此問題；換 cluster 時請修改路徑，並確認 compute node 可存取。這兩項是環境變數優先規則的例外，需直接修改設定檔；已有 `.env.sh` 請同步加入這兩行。

## 設定訓練參數（必要）

```bash
cp -n scripts/training/envs/tp1/.env.example.sh scripts/training/envs/tp1/.env.sh
# 編輯 .env.sh 的 MODEL、DATASET、batch、learning rate、W&B 等設定
```

`.env.sh` 是訓練預設值的唯一來源，已被 Git 忽略；兩個 launcher 自動載入，不必手動 `source`。已有舊版 `.env.sh` 時，請對照新範本補齊欄位；上面的指令不覆寫既有檔案。

- 優先順序：Python CLI > 已存在的環境變數 > `.env.sh` 預設值。保留範本的 `${VAR:-default}` 寫法，才能接受臨時覆寫；shell 中殘留的 export 也會優先。
- 範本採用 8 workers、micro-batch 1、gradient accumulation 2、ZeRO-2。單 process 設 `NPROC_PER_NODE=1`；停用 DeepSpeed 設 `DEEPSPEED_CONFIG=`。
- 一般 launcher 預設載入此目錄的 `.env.sh`；Slurm 必須明確指定 `TRAINING_ENV_FILE=/absolute/path/to/experiment.sh`，不會猜測路徑。兩者都可使用放在其他位置的配方；缺檔或缺少必要參數會報錯。
- Loader 提供絕對路徑 `TRAINING_DIR` 與 `WORKSPACE_DIR`，預設路徑不受執行目錄影響。自訂檔案請沿用範本、只放設定；Slurm preflight 與子 launcher 都會讀取它。
- GPU／CPU／RAM／time／partition 仍由 `.sbatch` 或 `sbatch` 參數管理；8-GPU template 會拒絕不相容的 worker 設定，不會強制覆寫。

只載入自己信任的 shell 設定，不要保存 API key。提交中的 job 在開始執行時才讀取設定檔；排隊期間不要修改該配方，實驗間可使用不同的 `TRAINING_ENV_FILE`。

接著依 [Quickstart](../../docs/quickstart.md) 或 [Slurm manual](../../docs/slurm.md) 執行。
