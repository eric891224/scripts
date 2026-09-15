# 實作說明：流程、reasoning、loss 與測試

[回 manual 索引](../README.md)

本頁：[檔案分工](#files) · [程式流程](#flow) · [Reasoning／loss](#reasoning) · [限制](#limits) · [測試](#tests)

下方 shell 指令都從 workspace 根目錄執行，不是從 `docs/` 執行。

只想操作訓練時，請直接看 [Training](training.md) 或 [Slurm manual](slurm.md)。本頁集中說明程式行為，不是提交 job 的步驟。

<a id="files"></a>

## 檔案與分工

| 檔案 | 負責的事情 |
| --- | --- |
| [data_preparation.py](../data_preparation.py) | 載入原始資料、轉成 canonical schema、按固定配額混合、儲存 Dataset 與 mixture recipe。 |
| [training.py](../training.py) | 載入已儲存的 Dataset、轉接 reasoning 欄位、預覽 loss tokens、執行 SFT 並儲存結果。 |
| [run_training.sh](../run_training.sh) | 把環境變數轉成 Python CLI 參數；不負責轉換或混合資料。 |
| [submit_training.sbatch](../submit_training.sbatch) | 申請單節點 8 GPU，設定 job 輸出位置，啟動一個 Slurm task。 |
| [deepspeed_zero2.json](../deepspeed_zero2.json) | ZeRO-2 配置，batch size 與 precision 由 Trainer 同步。 |
| [test_training.py](../tests/test_training.py) | 測試欄位轉接、template、loss mask、CLI、輸出保護及小型 CPU 訓練。 |
| [test_slurm.py](../tests/test_slurm.py) | 不提交 job 的 launcher 測試、DeepSpeed 參數與跨 rank 輸出檢查測試。 |

完整流程：

```text
原始 domain / retention datasets
  → data_preparation.py：convert → fixed-quota mix → save_to_disk
  → training.py：load_from_disk → reasoning adapter → chat template
  → TRL：tokenize → assistant loss mask → truncate → train
  → checkpoints、final model/tokenizer、metrics
```

<a id="flow"></a>

## `training.py` 依序做了什麼？

1. `parse_args()` 解析參數，檢查樣本數、步數等基本條件，並設定 seed。
2. `load_training_dataset()` 用 `load_from_disk()` 讀取資料，必要時選取前 N 筆。
3. `resolve_model_path()` 在離線模式下將 Hub ID 解析到已有的 config 所在 snapshot；載入 tokenizer 並選擇 training template。
4. `preview_sample()` 顯示前幾筆的長度、截斷情形與 loss 文字。若為 dry-run，到此結束。
5. `build_training_config()` 建立 TRL 設定與分散式狀態；`check_distributed_output_directory()` 只讓 rank 0 檢查輸出／續訓路徑，再將檢查結果廣播給所有 ranks，避免快的 rank 建檔後被慢的 rank 誤判成既有 run。
6. 用 `Dataset.map(adapt_sample)` 轉接全部選取資料的 reasoning 欄位，保留其他 provenance 欄位。主 process 優先建立 cache，其他 ranks 等待後再載入；各 rank 應使用相同、可寫入 cache 的資料路徑。
7. 建立 `SFTTrainer`。Trainer 載入模型、協調 template/tokenize preprocessing、建立 labels、截斷並移除完全沒有 loss token 的樣本。
8. 列印 preprocessing 前後的筆數；若沒有樣本剩下就報錯。寫出執行參數與 training template。
9. `trainer.train()` 執行訓練或續訓，依間隔記錄與儲存 checkpoints。
10. 正常完成後，儲存 final model/tokenizer、Trainer state 及 training metrics。

除了可調參數，程式還固定設定：

| 設定 | 原因／行為 |
| --- | --- |
| `assistant_only_loss=True` | 只監督 assistant 輸出。 |
| `loss_type="nll"` | 使用標準負對數似然 loss，不走 `chunked_nll` 路徑。 |
| `packing=False` | 不將多筆 conversation 打包成同一訓練序列。 |
| `truncation_mode="keep_start"` | 超長時保留前 `MAX_LENGTH` 個 tokens，截掉尾端。 |
| `eos_token="<\|im_end\|>"` | 使用目前 Qwen template 的 turn 結束 token。 |
| `use_cache=False` | 模型載入時關閉 generation KV cache。 |
| `device_map=None` | 由 Trainer 放置模型，不使用 inference-style 自動切分。 |

特別注意：`keep_start` 可能截掉 reasoning 的後半段或最終答案；即使仍有部分 assistant tokens，該筆也會繼續訓練。只有完全沒有可用 loss token 的樣本才會被丟掉。因此「處理前後筆數相同」不代表沒有截斷。

<a id="reasoning"></a>

## Reasoning 與 loss 是怎麼處理的？

### Canonical schema 與 Qwen 欄位轉接

資料中的 assistant 訊息例如：

```json
{"role": "assistant", "content": "答案", "reasoning": "推理過程"}
```

`adapt_sample()` 建立模型專用的訊息副本：

```json
{"role": "assistant", "content": "答案", "reasoning_content": "推理過程"}
```

原因是目前 `qwen.jinja` 讀取 `reasoning_content`，不會讀 `reasoning`。這個 adapter 不修改 canonical schema，也不重新儲存／覆寫原始 Dataset；Hugging Face 在正式 preprocessing 時仍可能建立衍生 cache 檔案。

當 reasoning 是 `None` 或缺少時，adapter 給 Qwen 空字串，所以 retention 訊息會渲染成：

```text
<think>

</think>

答案
```

空的 thinking 區塊也會參與 loss。這不是把 retention 排除於 reasoning 訓練之外，而是教模型在這些樣本上直接回答、不產生額外 reasoning。程式目前沒有「只訓練答案、遮住 reasoning」的開關。

### 為什麼不用原始 template 直接訓練？

提供的 Qwen inference template 會省略較早 assistant 輪次的 reasoning，而且沒有標示 assistant loss 範圍的 Jinja `generation` markers。

`resolve_training_template()` 使用 TRL 的 `get_training_chat_template()` 取得相容版本，供預覽使用。建立 `SFTTrainer` 時，TRL 也會因 `assistant_only_loss=True` 對相同的 inference template 套用對應 training template。它會：

- 保留較早 assistant 輪次的 reasoning，不只處理最後一輪。
- 將每輪 assistant 的 thinking 區塊、答案及 `<|im_end|>` 納入 loss。
- 將 user/system 訊息及 assistant role header 排除於 loss；這些 token 仍可作為模型輸入 context。

原始 `qwen.jinja` 不會被改寫，tokenizer 儲存的仍是 inference template。實際 training template 另存為 `training_chat_template.jinja`，方便檢查。

這依賴目前安裝的 TRL 對該 template 的支援；不支援時會明確報錯。更換模型時，除了 `MODEL`，也要確認 `CHAT_TEMPLATE`、special tokens 與模型匹配；本腳本不是任意模型都能直接套用的通用入口。

<a id="limits"></a>

## 限制與實驗注意事項

- 目前提供單節點 DeepSpeed ZeRO-2，沒有 LoRA／QLoRA、量化、FSDP 或多節點 launcher。9B 全參數訓練的記憶體與吞吐量需在目標 server 確認，不能用 inference 的記憶體需求推估。
- 目前使用 tokenizer 走純文字訓練路徑，不是圖片／影片訓練 pipeline。
- 不會建立 validation split，也沒有 benchmark evaluation。混合資料可能包含 oversampling 的重複 ID，直接切分混合結果可能造成 train/eval 洩漏；應先規劃獨立且去重的評估資料。
- Training loss 下降不代表 domain 能力提升或 retention 成功；比較 domain-only 與 domain+retention，仍需要相同基底模型、明確的訓練預算與獨立評估。
- 80/20 樣本比例不等於 token／loss-token 比例。截斷或移除樣本也可能改變實際組成。腳本目前只顯示前幾筆 preview 與總筆數，沒有全資料集的分類別 token 統計。

<a id="tests"></a>

## 測試

在 workspace 根目錄執行：

```bash
sm-dp/.venv/bin/python -m pytest scripts/training/tests -q
bash -n scripts/training/run_training.sh scripts/training/submit_training.sbatch
```

測試使用本地建立的 tokenizer、小型隨機 GPT-2 模型與暫存 Dataset，不需下載模型或使用 GPU。包含 reasoning 欄位轉接、多輪 loss mask、空 reasoning、截斷、離線路徑、CLI 傳遞、輸出保護、dry-run 不建立 Trainer，以及真正的一步 CPU 訓練和儲存。

Slurm 相關測試使用假的 `srun`／Python process 捕捉參數，不會提交 job、初始化 NCCL 或執行 DeepSpeed。另測試多 rank 輸出檢查的廣播邏輯。

這些測試能驗證程式串接，不代表 Qwen 9B 已在 server 上完成 8 GPU／DeepSpeed 訓練；仍需 [Slurm 實機 smoke test](slurm.md#smoke-test)。
