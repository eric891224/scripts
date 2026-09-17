# Quickstart：環境、資料與 dry-run

[回 manual 索引](../README.md)

本頁：[準備環境與資料](#setup) · [Dry-run](#dry-run) · [離線／本地模型](#offline)

下方 shell 指令都從 workspace 根目錄執行，不是從 `docs/` 執行。

<a id="setup"></a>

## 1. 準備環境與資料

下方指令都假設目前位於 workspace 根目錄，例如 `/home/siliconmind/cl`：

```text
workspace/
  sm-dp/                     # pyproject.toml、uv.lock、.venv/
  scripts/training/
  dataset/mixed/siliconmind-retention-v1/
```

使用 Python 3.12 以上，以及 `sm-dp` 的依賴環境。若尚未建立環境，可在根目錄執行：

```bash
uv sync --project sm-dp --locked
```

Launcher 預設使用 `sm-dp/.venv/bin/python`。Server 若使用另一個環境，可以設定 `PYTHON_BIN=/path/to/python`。GPU 驅動、PyTorch/CUDA 相容性及 BF16 支援仍需在 server 確認；建立 Python 環境不代表 GPU 已配置完成。

Chat template 預設直接使用模型 tokenizer 附帶的版本，不需要額外放置 `qwen.jinja`。只有需要自訂時才設定 `CHAT_TEMPLATE=/path/to/custom.jinja` 或 `--chat-template /path/to/custom.jinja`。缺少 template 或 TRL 不支援時會明確報錯，不會自動改用其他模型的格式。

訓練輸入必須是由 Hugging Face `Dataset.save_to_disk()` 儲存的**單一 Dataset**，包含 canonical `messages`。不是 JSONL 路徑、Hub dataset ID 或 `DatasetDict`。程式只檢查基本結構，不能取代前面的 converter/schema validation。

如果已經有混合資料，就不必重跑 data preparation。若要重新產生，先檢查 `data_preparation.py` 的資料路徑及輸出位置，再執行：

```bash
PYTHONPATH=sm-dp/src sm-dp/.venv/bin/python scripts/training/data_preparation.py
```

注意：data preparation 目前的 `DATASET_ROOT` 寫死為 `/home/siliconmind/cl/dataset`，尚未提供 CLI；換 server 時要調整。重新產生前請選好輸出位置，避免覆寫先前 recipe 或混用資料版本。

目前 preparation script 的配方是 45,000 筆：36,000 筆 spec2rtl、4,500 筆 everyday-conversations、4,500 筆 metamathqa。這是 **80% domain + 20% retention 的樣本比例**，不是 token 比例。training script 不會重新混合或重新保證這個比例。

<a id="dry-run"></a>

## 2. 先做 dry-run

```bash
bash scripts/training/run_training.sh --dry-run --preview-samples 3
```

這會讀取資料、載入 tokenizer、解析 template，並檢查前 3 筆樣本，不載入模型權重、不執行訓練，也不建立訓練輸出目錄。未使用離線模式時，載入 tokenizer 可能下載相關檔案。

每筆預覽包含：

| 欄位 | 意義 |
| --- | --- |
| `tokens` | 整段 conversation 套用 template 後的原始 token 數。 |
| `retained_tokens` | 按 `MAX_LENGTH` 截斷後保留的 token 數。 |
| `loss_tokens` | 截斷後參與 loss 的 assistant token 數；計數考慮 causal LM 的 label shift。 |
| `truncated` | 是否超過長度上限。 |
| `loss_preview` | 參與 loss 的文字預覽，最多顯示 1,500 個字元，並非完整訓練內容。 |

優先確認 reasoning 與答案有出現、user/system 訊息沒有被算入 loss。`loss_tokens=0` 表示該筆截斷後沒有可訓練的 assistant token。

Dry-run 只檢查指定的前幾筆，不是全資料集驗證，也不能確認 GPU 記憶體是否足夠。

<a id="offline"></a>

### 離線或使用本地模型

如果 config 與 tokenizer 已快取：

```bash
bash scripts/training/run_training.sh --dry-run --local-files-only
```

也可以直接指定模型目錄：

```bash
MODEL=/path/to/Qwen3.5-9B \
DATASET=/path/to/mixed-dataset \
bash scripts/training/run_training.sh --dry-run --local-files-only
```

離線 dry-run 只需要 config/tokenizer；真正訓練還需要完整權重。Dry-run 成功不代表權重已備齊。

## 下一步

Dry-run 成功後，前往 [Slurm 兩步 smoke test](slurm.md#smoke-test)。若不使用 Slurm，請看 [一般訓練操作](training.md#smoke-test)。Reasoning／loss 的細節在 [實作說明](internals.md#reasoning)。
