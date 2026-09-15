# W&B：啟用、設定與離線記錄

[回 manual 索引](../README.md)

本頁：[登入與啟用](#enable) · [設定表](#settings) · [離線](#offline) · [續訓注意事項](#resume)

下方 shell 指令都從 workspace 根目錄執行，不是從 `docs/` 執行。

<a id="enable"></a>

## 1. 登入並啟用

在訓練機器上先執行 `wandb login`，並確認 launcher 使用的 Python 環境已安裝 `wandb`。API key 不要放進 script、README 或版本控制。

以下直接執行範例會啟動訓練；Slurm 使用者應將 `REPORT_TO=wandb` 等環境變數加在 [sbatch 提交指令](slurm.md#training) 前，不要直接在 login node 執行。

使用預設專案與名稱，啟動訓練並連接 W&B：

```bash
REPORT_TO=wandb bash scripts/training/run_training.sh
```

也可以覆蓋名稱與其他訓練參數：

```bash
REPORT_TO=wandb \
WANDB_PROJECT=sm-dp-training \
WANDB_NAME=qwen35-domain80-retention20-seed123 \
SEED=123 \
LOGGING_STEPS=10 \
OUTPUT_DIR=outputs/qwen-domain-retention-seed123 \
bash scripts/training/run_training.sh
```

這些指令會真正啟動訓練並將相關記錄同步到 W&B；terminal 會顯示 run 連結。`LOGGING_STEPS` 控制訓練指標記錄間隔。`--dry-run` 不會建立 W&B run，目前也不會產生 domain／retention evaluation 分數。

若直接執行 `training.py` 而不經過 launcher，bash 裡的 W&B 預設值不會套用，需自行提供環境變數及 `--report-to wandb`。換 server 時也需要在該機器登入，並非本機登入一次就會自動同步憑證。

<a id="settings"></a>

## 2. Project、run 與上傳設定

`run_training.sh` 有獨立的 W&B 設定區塊，可以直接修改預設值：

| 設定 | 預設值 | 用途 |
| --- | --- | --- |
| `REPORT_TO` | `none` | 改為 `wandb` 才啟用 W&B 記錄。 |
| `WANDB_ENTITY` | `s96006730-siliconmind` | 目前使用者的 W&B entity；換帳號／team 時請覆蓋。 |
| `WANDB_PROJECT` | `sm-dp-training` | 將相關實驗放在同一專案。 |
| `WANDB_NAME` | `qwen35-domain80-retention20-v1` | 本次 run 的顯示名稱，不是續訓用的 run ID；換配方／模型時應一併修改。 |
| `WANDB_LOG_MODEL` | `false` | 不自動上傳模型 checkpoints；仍可記錄 metrics、設定與 logs。 |

這些 `WANDB_*` 變數會被 `export` 給 Python process，由 Trainer/W&B 整合直接讀取，不需要另外呼叫 `wandb.init()`。設定方式可參考 [W&B 環境變數文件](https://docs.wandb.ai/models/track/environment-variables)。

Slurm launcher 預設用 `qwen-zero2-<job-id>` 作為 run name，可用 `WANDB_NAME` 覆蓋，詳見 [Slurm 預設值](slurm.md#defaults)。其餘訓練參數統一在 [Training 參數表](training.md#parameters)。

<a id="offline"></a>

## 3. 離線記錄

Compute node 不能連外時，設定 `WANDB_MODE=offline REPORT_TO=wandb`，之後在可連網環境同步。模型的 `--local-files-only` 與 W&B 離線模式是獨立設定，詳見 [Slurm 離線執行](slurm.md#offline)。

<a id="resume"></a>

## 4. 續訓注意事項

W&B 是否延續同一個 run 是另一件事：Trainer checkpoint 的續訓不代表 W&B run 也自動續接；不要只靠相同的顯示名稱判斷它們是同一個 run。
