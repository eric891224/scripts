# TP1 環境

[回 training 索引](../../README.md)

## 安裝

從 workspace 根目錄執行：

```bash
uv sync --project scripts/training/envs/tp1 --locked
```

## 設定

```bash
cp -n scripts/training/envs/tp1/.env.example.sh scripts/training/envs/tp1/.env.sh
# 手動修改 .env.sh
source scripts/training/envs/tp1/.env.sh
```

設定只有四類：Python／model／dataset、output、W&B、TP1 的 CC/CXX patch。全部直接賦值，每次 source 都覆寫舊值；不會推導路徑或自動產生 run 名稱。範本使用目前 TP1 server 路徑，換機器請改字串。

CC/CXX 指定 GCC 14.2，處理這個站台使用 NVHPC nvc 時的 Triton 編譯失敗。這是 TP1 設定，不放進共用 launcher；其他站台需核對 compiler 路徑。

.env.sh 已被 Git 忽略，不要放 API key。已有舊版設定時請手動更新，cp -n 不會覆寫。

接著：[dry-run](../../docs/quickstart.md) → [Slurm smoke test](../../docs/slurm.md)。
