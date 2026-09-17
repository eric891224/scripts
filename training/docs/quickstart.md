# Quickstart：環境、資料與 dry-run

[回 manual 索引](../README.md)

本頁：[準備環境與資料](#setup) · [Dry-run](#dry-run) · [離線／本地模型](#offline)

下方 shell 指令都從 workspace 根目錄執行，不是從 `docs/` 執行。

<a id="setup"></a>

## 1. 準備環境與資料

先在本機完成資料轉換與混合，server 只負責訓練，不需安裝或上傳 `sm-dp`。目錄配置：

```text
workspace/
  scripts/training/                       # 包含 envs/tp1/ 訓練環境
  dataset/mixed/siliconmind-retention-v1/   # 上傳的完整 saved Dataset
```

在 server 建立獨立訓練環境：

```bash
uv sync --project scripts/training/envs/tp1 --locked
```

Launcher 預設使用 `scripts/training/envs/tp1/.venv/bin/python`；其他環境可用 `PYTHON_BIN` 覆蓋。DeepSpeed 安裝與個人設定見 [TP1 環境說明](../envs/tp1/README.md)。

Chat template 預設直接使用模型 tokenizer 附帶的版本，不需要額外放置 `qwen.jinja`。只有需要自訂時才設定 `CHAT_TEMPLATE=/path/to/custom.jinja` 或 `--chat-template /path/to/custom.jinja`。缺少 template 或 TRL 不支援時會明確報錯，不會自動改用其他模型的格式。

上傳 `Dataset.save_to_disk()` 產生的完整目錄（包括 Arrow shards 與 metadata），不是單一 JSONL、Hub dataset ID 或 `DatasetDict`。Dataset 需包含 canonical `messages`；資料驗證、配額與混合在本機完成。Server 仍會套用 chat template、tokenize 與建立 loss mask。

<a id="dry-run"></a>

## 2. 先做 dry-run

```bash
DATASET=/path/to/uploaded-dataset \
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
