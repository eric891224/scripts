# Domain + retention SFT

用已預處理並上傳的 Dataset 對 Qwen 做 full fine-tuning，server 不需要 `sm-dp`。支援單 process，以及單節點 8 × H100 的 Slurm + DeepSpeed ZeRO-2；目前沒有 LoRA 或 evaluation pipeline。

## 我想做什麼？

| 需求 | Manual |
| --- | --- |
| **在 Slurm／8 × H100 上訓練** | [Slurm 操作手冊](docs/slurm.md) |
| 第一次使用：環境、資料、dry-run | [Quickstart](docs/quickstart.md) |
| 啟動一般訓練、續訓、找輸出 | [Training 操作手冊](docs/training.md) |
| 查環境變數、CLI、batch size 等設定 | [完整參數表](docs/training.md#parameters) |
| 接 W&B、設定 project／run、離線記錄 | [W&B 操作手冊](docs/wandb.md) |
| 理解 reasoning、loss mask、截斷與程式流程 | [實作說明](docs/internals.md) |
| 執行測試 | [測試方式](docs/internals.md#tests) |

## 建議使用順序

第一次：**[準備環境與 dry-run](docs/quickstart.md) → [Slurm 兩步 smoke test](docs/slurm.md#smoke-test) → [正式提交](docs/slurm.md#training)**。

文件中的 shell 指令都從 workspace 根目錄（例如 `/home/siliconmind/cl`）執行，不是從 `scripts/training/docs/` 執行。Slurm 的 partition、account 與環境載入方式需要依你的 cluster 設定。

完整模型與 8 GPU 訓練仍需在 server 驗證；dry-run 與本地測試不代表 GPU 記憶體足夠。
