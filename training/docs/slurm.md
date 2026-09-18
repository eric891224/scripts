# Slurm：單節點 8 GPU

[回索引](../README.md)

固定從 workspace 根目錄提交，先手動載入環境：

```bash
source scripts/training/envs/tp1/.env.sh
```

目前 .sbatch 是 p01、1 node、8 GPUs、1 task、32 CPUs、256 GB host RAM、2 小時。換站台時只修改 .sbatch 的資源設定；不要把 task 數改成 8。GPU request 與 NPROC_PER_NODE=8 必須一起調整，launcher 會檢查 GPU 可見數。

## Smoke test

```bash
sbatch --time=00:30:00 scripts/training/submit_training.sbatch --smoke
```

只用前 32 筆、跑 2 個 optimizer steps，每步記錄並存 checkpoint。仍載入完整模型、儲存完整訓練狀態，並不省下模型／optimizer 所需記憶體。

查看 Slurm 回傳的 job ID：

```bash
squeue -j JOB_ID
tail -f slurm-qwen-sft-test-JOB_ID.out
tail -n 80 slurm-qwen-sft-test-JOB_ID.err
```

確認 loss 有輸出、job 正常結束，且輸出目錄有 checkpoint-1、checkpoint-2 和 final。不能只以模型成功載入判定跑通。

## 正式訓練與續訓

正式訓練前修改 .env.sh 的 OUTPUT_DIR／WANDB_NAME，選一個新的 run，重新 source：

```bash
source scripts/training/envs/tp1/.env.sh
sbatch --time=12:00:00 scripts/training/submit_training.sbatch
```

12 小時只是提交範例，不是訓練時間保證。

續訓先在 .env.sh 指回原 OUTPUT_DIR，再 source；其他配方與 world size 保持一致：

```bash
sbatch scripts/training/submit_training.sbatch --resume-from-checkpoint /path/to/checkpoint-250
```

也可加 --smoke 從 checkpoint-1 試著續到總步數 2；不是再增加 2 步。Checkpoint 必須保留所有 ranks 的狀態，final 不是續訓 checkpoint。

## 分工與注意事項

.sbatch 只申請資源，再透過一個 srun task 啟動 launcher；launcher 檢查設定並以 torchrun 啟動 workers。設定由提交環境繼承，不使用 --export=NONE，也不改寫 CUDA_VISIBLE_DEVICES。

建議從 login node 提交。若在既有 allocation 裡提交，舊的 SLURM_MEM_PER_* 可能與新的 --mem 衝突，請先確認環境。已取得 8 GPU allocation 時，可直接用：

```bash
NPROC_PER_NODE=8 srun --ntasks=1 --gpu-bind=none bash scripts/training/run_training.sh --smoke
```

這會使用原 job 的資源與剩餘時間，不另提交 batch job。
