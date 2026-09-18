# 實作與測試

[回索引](../README.md)

## 流程

手動 source → sbatch 申請資源 → srun 一個 task → launcher 檢查輸入／GPU → torchrun → training.py。

training.py 載入 saved Dataset，將 assistant 的 reasoning 轉接成 reasoning_content；沒有 reasoning 的樣本保留空 think 區塊，不修改原始資料。

TRL patch tokenizer template 以保留多輪 reasoning 並建立 assistant loss mask，再 tokenize、截斷、訓練與儲存。User／system 不計入 loss。

Qwen3.5 的 cache 設定在 text_config；模型建立後透過 get_text_config() 關閉 KV cache，不把 use_cache 當 constructor 參數。輸出目錄由 rank 0 檢查並廣播錯誤，避免各 rank 建檔的競態。

## 本機測試

從 workspace 根目錄執行：

```bash
uv run --project scripts/training/envs/tp1 --locked --with pytest python -m pytest scripts/training/tests -q
bash -n scripts/training/run_training.sh
bash -n scripts/training/submit_training.sbatch
bash -n scripts/training/envs/tp1/.env.example.sh
```

包含必填參數、手動 source 覆寫、Slurm/torchrun 拓撲、GPU mismatch、reasoning/loss mask、Qwen3.5 小型模型載入、CPU backward/save 與輸出保護。

測試不提交 Slurm job、不下載 9B 權重。CPU 測試使用測試專用的小型配方；不代表正式 8-GPU 環境或長跑已驗證。
