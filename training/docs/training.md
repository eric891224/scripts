# MVP 訓練配方與輸出

[回索引](../README.md)

## 必填設定

手動 source .env.sh 後，launcher 要求 PYTHON_BIN、MODEL、DATASET、OUTPUT_DIR、REPORT_TO；缺少就報錯，不補預設值。

REPORT_TO 只接受 none／wandb；選 wandb 才要求 WANDB_ENTITY、WANDB_PROJECT、WANDB_NAME。API key 不放在設定檔。

直接執行 training.py 時，--model、--dataset、--output-dir、--report-to 也都是必填。

## 預設配方

只在 training.py 的 RECIPE 維護，不再提供對應的環境變數／CLI 旋鈕：

| 項目                 | 值                                                       |
| -------------------- | -------------------------------------------------------- |
| 訓練方式             | Full fine-tuning、BF16、ZeRO-2                           |
| Epoch／learning rate | 1／2e-5                                                  |
| 每卡 batch／累積步數 | 1／2；8 GPU global batch 為 16                           |
| 長度／截斷           | 4096 tokens／keep-start                                  |
| Loss                 | assistant reasoning + answer；assistant-only             |
| 其他                 | gradient checkpointing 開啟、packing 關閉、SDPA、seed 42 |
| 記錄／checkpoint     | 每 10／250 steps，保留最近 2 個 checkpoints              |

操作只保留 --dry-run、--smoke、--resume-from-checkpoint PATH。訓練步數均指 optimizer steps。

Smoke 使用前 32 筆，不能保證 domain／retention 比例；正式訓練使用全部資料。長樣本截斷或無 loss 樣本被移除，都可能改變有效混合比例。

## 輸出與安全

- run_arguments.json：此次輸入與固定配方；續訓另寫 resume_arguments.json。
- training_chat_template.jinja、deepspeed_config.json：訓練設定快照。
- checkpoint-N/：續訓用，包含 optimizer 等狀態。
- final/：模型與 tokenizer，供推論／載入，不是完整續訓狀態。
- trainer_state.json、train_results.json：進度與 metrics。

非空輸出目錄會拒絕新訓練；只有明確指定 resume 才允許沿用。重現實驗仍需另外保存 dataset／recipe、model revision、code commit、lockfile 與 DeepSpeed 版本。

不使用 Slurm 時，先取得 GPU 使用權，再明確提供 NPROC_PER_NODE 執行 run_training.sh。MVP 不提供 CPU 正式訓練、LoRA 或 evaluation pipeline。
