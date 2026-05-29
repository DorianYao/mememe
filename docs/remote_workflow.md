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

---

## 0.5 tmux 会话管理（强烈建议）

SSH 断连后，前台进程会被 SIGHUP 终止。**长跑训练务必在 tmux 内执行**。

### 基本操作

```bash
# 新建会话（命名便于识别）
tmux new -s mememe          # 通用
tmux new -s base_eco        # base_eco 扩展扫描
tmux new -s kw              # k×w 扫描

# 从会话内暂时退出（训练继续跑）
# 按键：Ctrl+b 然后 d

# 列出所有会话
tmux ls

# 重新 attach
tmux attach -t mememe
tmux attach -t base_eco

# 在已 attach 的会话里开新窗口（可选，并行看日志 + 跑命令）
# Ctrl+b 然后 c          # 新建窗口
# Ctrl+b 然后 n          # 下一窗口
# Ctrl+b 然后 p          # 上一窗口
# Ctrl+b 然后 0/1/2      # 跳到指定窗口
```

### 常用场景

**场景 A：启动长跑训练**

```bash
tmux new -s base_eco
source .venv/bin/activate
cd /workspace/mememe
bash scripts/run_category_kw_extended.sh base_eco
# Ctrl+b d  断开，训练继续
```

**场景 B：断连后回来查看**

```bash
tmux attach -t base_eco
# 若已在跑，直接看终端输出；另开窗口 tail 日志：
# Ctrl+b c
tail -f outputs/metrics/base_eco_kw_extended_ratio.log
```

**场景 C：会话已存在，避免重复 new**

```bash
tmux ls
# 若 base_eco 已存在：
tmux attach -t base_eco
# 不要 tmux new -s base_eco（会报错 duplicate session）
```

**场景 D：训练结束，清理会话**

```bash
# 在会话内
exit                        # 或 Ctrl+d，关闭当前 pane
# 或从外部强制结束
tmux kill-session -t base_eco
```

**场景 E：会话「卡住」、无输出**

```bash
# 另开 SSH，attach 后 Ctrl+c 中断当前命令
tmux attach -t base_eco
# 检查磁盘 / GPU 后重新跑（脚本支持 --skip-existing 断点续跑）
python3 scripts/cleanup_disk.py --status
bash scripts/run_category_kw_extended.sh base_eco
```

### 快捷键速查

| 操作 | 按键 |
| --- | --- |
| 断开（detach，训练继续） | `Ctrl+b` → `d` |
| 新建窗口 | `Ctrl+b` → `c` |
| 切换窗口 | `Ctrl+b` → `n` / `p` |
| 纵向分屏 | `Ctrl+b` → `%` |
| 横向分屏 | `Ctrl+b` → `"` |
| 在 pane 间切换 | `Ctrl+b` → 方向键 |
| 滚动查看历史 | `Ctrl+b` → `[`，方向键/`PgUp`，`q` 退出 |
| 关闭当前 pane | `Ctrl+b` → `x` 或 `exit` |

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
rm -f data/processed/base_eco16_15m_w224_loso_*_ratio_label_k15.npz
```

---

## 3. 开始训练

### 3a. base_eco 严格 calendar 验证（当前主线）

Phase A + Phase B 已完成（开发协议）。可信峰值：**k=1.5, w=224, AUC≈0.738**（test_n≈126k）。  
下一步：严格 calendar walk-forward，验证能否向前外推。

```bash
source .venv/bin/activate
cd /workspace/mememe

tmux new -s strict
python3 main.py --mode loso --category base_eco --stage all --model mlp \
  --label-k 1.5 --window-size 224 --split-mode calendar --ablation-tag label_k15
# Ctrl+b d
```

备选对照（样本更充裕）：k=1.2, w=208。

```bash
python3 main.py --mode loso --category base_eco --stage all --model mlp \
  --label-k 1.2 --window-size 208 --split-mode calendar --ablation-tag label_k12
```

### 3a-alt. base_eco extended k×w 峰值搜索（Phase B，已完成）

Phase A：k∈{1.0,1.2}, w∈{96,128,192}。  
Phase B：k∈[1.2,2.5], w 延伸至 288（26 combo，**已全部完成**）。

```bash
tmux new -s base_eco
bash scripts/run_category_kw_extended.sh base_eco
# Ctrl+b d
```

断点续跑：已有完整 `*_summary.json`（16 轮）的 combo 会自动 `[skip]`。

汇总与可信峰值：

```bash
python3 scripts/category_kw_extended_compare.py --category base_eco
python3 scripts/category_kw_extended_compare.py --category base_eco --plot
```

### 3b. 单类 Phase A 基线扫描（5 组合）

矩阵定义见 `scripts/category_kw_matrix.py`（Phase A 5 combo + Phase B 26 combo）。

```bash
tmux new -s scan
bash scripts/run_category_kw_scan.sh bluechip
python3 scripts/category_kw_compare.py --category bluechip
```

### 3c. 五类全跑（磁盘与时间充足时）

```bash
tmux new -s all_kw
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
LOSO_JOBS=8 NPZ_JOBS=12 BATCH_SIZE=65536 bash scripts/run_category_kw_extended.sh base_eco
```

---

## 4. 查看进度 / 汇总

```bash
# 日志（可在 tmux 第二窗口 tail）
tail -f outputs/metrics/base_eco_kw_extended_ratio.log

# extended 汇总（只写 JSON，不加 --plot 则不生成热力图）
python3 scripts/category_kw_extended_compare.py --category base_eco

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

本地解压后建议放到 `outputs/metricsB/` 与历史结果区分（见 [REPORT_CATEGORY.md](../REPORT_CATEGORY.md)）。

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
# 得到 outputs/metrics/...；若与本地 metricsB 区分，可：
# mkdir -p outputs/metricsB && cp -a outputs/metrics outputs/metricsB/
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

# 长跑务必 tmux
tmux new -s strict
python3 main.py --mode loso --category base_eco --stage all --model mlp \
  --label-k 1.5 --window-size 224 --split-mode calendar --ablation-tag label_k15
# Ctrl+b d

# 断连后恢复
tmux attach -t strict

# 完成后
python3 scripts/category_kw_extended_compare.py --category base_eco
tar czf mememe_metrics.tgz outputs/metrics/
```
