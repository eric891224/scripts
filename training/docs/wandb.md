# W&B

[回索引](../README.md)

先在 server 登入：

```bash
scripts/training/envs/tp1/.venv/bin/wandb login
```

在 .env.sh 修改 REPORT_TO="wandb"，並填好 WANDB_ENTITY、WANDB_PROJECT、WANDB_NAME；重新 source 後照 [Slurm manual](slurm.md) 提交。

- REPORT_TO="none"：不啟用 W&B。
- MVP 只記錄 metrics／設定，不自動上傳模型 checkpoints。
- API key 不寫入 script 或版本控制。
- Dry-run 不建立 W&B run。
- 無網路時可額外 export WANDB_MODE=offline，之後用 wandb sync 同步。
- Trainer resume 不代表 W&B 自動接回同一個 run；相同 WANDB_NAME 只是顯示名稱。
