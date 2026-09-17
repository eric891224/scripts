# TP1 Environment Setup

## Activate venv

Install dependencies (first time only)

```shell
uv venv # specify python version if needed
source .venv/bin/activate
uv sync --locked
```

activate the venv

```shell
source .venv/bin/activate
```

## Setup Environment Variables

```shell
cp .env.example.sh .env.sh
# after setting .env.sh, run
source .env.sh
```
