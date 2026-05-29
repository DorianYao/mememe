# 远程 Pod 操作流程

适用于 RunPod / 云 GPU：已 `git clone` 且 SSH 连上，从零跑到打包拉回结果。

**产出物**：`outputs/metrics/` 下的 JSON / CSV / log（无训练图片）。  
**不打包**：`data/processed/`（.npz）、`outputs/models/`（.pt）、`outputs/figures/`。

---

## 0. 环境（首次）

```bash
cd /workspace/mememe   # 或你的 clone 路径

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# GPU 主机（A40 / L4 等，CUDA 12.x）
pip install torch --index-url https://download.pytorch.org/whl/cu124
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

建议用 `tmux` 防止断连：

```bash
tmux new -s mememe
```

---

## 1. 下载数据

```bash
source .venv/bin/activate
cd /workspace/mememe

# 五类 × 16 币 + legacy meme8，去重约 41 个 symbol
python3 scripts/download_all_category_data.py

# 确认 41/41 就绪
python3 scripts/check_category_data_ready.py
python3 scripts/validate_binance_symbols.py   # API 451 时自动改用本地 CSV 校验
```

若 Pod 无法访问 `api.binance.com`（HTTP **451**），请先 **scp 传入 `data/raw/`**，再跑训练；无需能连 Binance API。也可显式：

```bash
export BINANCE_VALIDATE_OFFLINE=1
python3 scripts/validate_binance_symbols.py --offline
```
```

原始 CSV 在 `data/raw/`（约几百 MB），需保留到训练结束。

---

## 2. 磁盘检查（训练前必做）

每个 k×w 组合特征阶段会临时占用约 **15–25 GB**（16 个 `.npz`）。  
Pod 磁盘建议 **≥ 30 GB 可用**；若之前跑过 scan，先清理：

```bash
python3 scripts/cleanup_disk.py --status
python3 scripts/cleanup_disk.py --npz-completed --dry-run
python3 scripts/cleanup_disk.py --npz-completed
```

训练脚本默认 **每个 combo 跑完后自动删该 combo 的 .npz/.pt**（`AUTO_CLEANUP_NPZ=1`）。

若 `.npz` 损坏（`BadZipFile`），删掉对应 combo 重建：

```bash
rm -f data/processed/bluechip16_15m_w208_loso_*_ratio_label_k12.npz
```

---

## 3. 开始训练

### 3a. Bluechip extended k×w 峰值搜索（当前主线）

```bash
source .venv/bin/activate
cd /workspace/mememe

# 自动：GPU batch=16384、LOSO_JOBS=4、特征 8 线程、无图片输出
bash scripts/run_category_kw_extended.sh bluechip
```

断点续跑：已有完整 `*_summary.json`（16 轮）的 combo 会自动 `[skip]`。

### 3b. 单类 Phase A 基线扫描（5 组合）

矩阵定义见 `scripts/category_kw_matrix.py`（Phase A 5 combo + Phase B 22 combo）。

```bash
bash scripts/run_category_kw_scan.sh bluechip
python3 scripts/category_kw_compare.py --category bluechip
```

### 3c. 五类全跑（磁盘与时间充足时）

```bash
bash scripts/run_all_category_kw_scans.sh
```

### 可选环境变量

| 变量 | 高端 GPU 默认（≥70GB） | 说明 |
|------|------------------------|------|
| `LOSO_JOBS` | **8** | 并行 held-out 训练（A40 48GB 为 6） |
| `FEATURE_WORKERS` | min(cpu−1, **16**) | 建窗口进程数 |
| `NPZ_JOBS` | **12** | 并行写 LOSO `.npz` 线程数 |
| `BATCH_SIZE` | **65536**（A40 为 32768） | 自动按显存档 |
| `AUTO_CLEANUP_NPZ` | 1 | combo 完成后删 npz/pt |

档位由 `python3 scripts/detect_runtime.py --json` 检测（`high` / `mid` / `entry`）。

手动覆盖示例（H100 / 大内存 Pod）：

```bash
LOSO_JOBS=8 NPZ_JOBS=12 BATCH_SIZE=65536 bash scripts/run_category_kw_extended.sh bluechip
```

---

## 4. 查看进度 / 汇总

```bash
# 日志
tail -f outputs/metrics/bluechip_kw_extended_ratio.log

# extended 汇总（只写 JSON，不加 --plot 则不生成热力图）
python3 scripts/category_kw_extended_compare.py --category bluechip

# 关键结果文件
ls outputs/metrics/*summary*.json
ls outputs/metrics/category_kw*.json
```

---

## 5. 打包（仅 metrics）

在 **repo 根目录** 打包（避免解压路径嵌套）：

```bash
cd /workspace/mememe

tar czf mememe_metrics.tgz \
  outputs/metrics/

ls -lh mememe_metrics.tgz
```

需要一并带走扫描日志时，上面已含 `*.log`。  
若还要 optimal 汇总 JSON，也在 `outputs/metrics/` 内，无需额外目录。

**不要打进包**：`data/processed/`、`outputs/models/`、`data/raw/`（raw 本地可重新下载）。

---

## 6. 拉回本地（WSL）

RunPod 用 **SSH over exposed TCP**（支持 SCP），不要用 `ssh.runpod.io` 那条。

```bash
# 本地
cd ~/mememe

scp -P <PORT> -i ~/.ssh/id_ed25519 \
  root@<IP>:/workspace/mememe/mememe_metrics.tgz \
  ./

tar xzf mememe_metrics.tgz
# 得到 outputs/metrics/...
```

或 rsync 增量同步：

```bash
export REMOTE_HOST="root@<IP>"
export REMOTE_SSH="ssh -p <PORT> -i ~/.ssh/id_ed25519"
export REMOTE_DIR=/workspace/mememe
bash scripts/sync_results_from_remote.sh
```

---

## 7. 最小命令清单（复制用）

```bash
cd /workspace/mememe && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124

python3 scripts/download_all_category_data.py
python3 scripts/check_category_data_ready.py
python3 scripts/cleanup_disk.py --npz-completed

tmux new -s kw
bash scripts/run_category_kw_extended.sh bluechip

# 完成后
python3 scripts/category_kw_extended_compare.py --category bluechip
tar czf mememe_metrics.tgz outputs/metrics/
```
