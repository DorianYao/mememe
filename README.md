# finPaper — Meme 币 15m 方向预测与跨域评估

> 8 个 Meme 币、动态波动率标签、Leave-One-Symbol-Out（LOSO）；**区分开发协议与严格 calendar walk-forward 协议**。

本项目从「单币 BTC 二分类」演进为 **跨币种零样本评估**。核心结论（与 [`paper/main.tex`](paper/main.tex) 一致）：

| 评估协议 | 冻结配置 `k=1.2, w=192`（MLP） | 含义 |
| --- | ---: | --- |
| **开发协议**（ratio-LOSO，held-out **全时段**） | AUC **0.743** ± 0.072 | 超参探索、消融、机制分析；**非**跨域泛化主结论 |
| **严格协议**（calendar walk-forward，仅测 **>85% 未来**） | AUC **0.510** ± 0.014 | **主结论**：向前外推 ≈ 随机 |
| 外部确认（TURBO，未参与调参） | AUC **≈ 0.497** | 支持严格协议结论 |

泄漏审计 **PASS**（无 future-bar 泄露）；开发协议高分主要来自 **训练池与 held-out 共享日历区间**（共享市场状态），而非实现错误。详见 **[REPORT.md](REPORT.md)**。

---

## 核心设计

- **数据**：Binance 现货 15m OHLCV，8 币 × 约 720 天（`data/raw/`）
- **特征**：15 维去币种化（形态、波动、RSI/MACD、taker 等）
- **标签**：`|r_{t+1}| > k·σ_t` 的方向事件（小波动丢弃）
- **模型**：MLP 为主；XGBoost / 1D CNN 作对照
- **评估**：ROC-AUC（主指标）、MCC、Macro-F1；开发协议下可选 PnL 回测

---

## 两种 LOSO 协议（必读）

| 协议 | CLI | 训练 | 测试（held-out 币） |
| --- | --- | --- | --- |
| **开发**（默认 `split_mode=ratio`） | `--split-mode ratio` | 其余 7 币池化，时间前 85% / 验证末 15% | held-out **全样本时段**（与训练池日历可重叠） |
| **严格**（主结论） | `--split-mode calendar` | 其余 7 币，$t \le 70\%$ / 验证 70–85% | held-out **仅** $t > 85\%$，且 $\max(\text{train})<\min(\text{test})$ |

```bash
# 严格协议主实验（论文主指标）
python3 main.py --mode loso --stage all --model mlp \
  --label-k 1.2 --window-size 192 --split-mode calendar \
  --ablation-tag label_k12

# 开发协议（超参扫描 / 对照）
python3 main.py --mode loso --stage all --model mlp \
  --label-k 1.2 --window-size 192 --ablation-tag label_k12
```

---

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

依赖：`numpy`, `pandas`, `scikit-learn`, `torch>=2.1`, `xgboost`, `matplotlib`, `scipy`, `pytest` 等（见 `requirements.txt`）。无需 Binance API key（数据已落盘或自行下载）。

---

## 项目结构

```
finPaper/
├── main.py                          # single / pooled / loso
├── paper/main.tex                   # 课程/研究论文 PDF 源文件
├── REPORT.md                        # 实验全记录（单一入口）
├── config/default.yaml
├── scripts/
│   ├── generate_paper_figures.py
│   ├── paper_mechanism_figures.py   # 相关矩阵 + 协议时间轴图
│   ├── statistical_significance.py
│   ├── regime_analysis.py
│   ├── backtest_pnl_realistic.py
│   ├── leakage_audit.py
│   └── holdout_coins_eval.py
├── src/                             # 数据、特征、multi、训练、评估
├── tests/test_loso_splits.py
├── data/raw/                        # {SYMBOL}_15m_720d.csv
└── outputs/metrics/                 # JSON 汇总与日志
```

---

## 常用命令

```bash
# LOSO 基线（开发协议）
python3 main.py --mode loso --stage all --model mlp --label-k 0.3 --window-size 20

# 超参扫描（开发协议）
bash scripts/run_label_k_scan.sh
bash scripts/run_window_scan_k1.sh
bash scripts/run_pruned_kw_scan.sh

# 机制图与论文图
python3 scripts/paper_mechanism_figures.py
python3 scripts/generate_paper_figures.py

# 统计检验 / regime / 审计 / 外部币
python3 scripts/statistical_significance.py --window 192 --tag label_k12
python3 scripts/regime_analysis.py --window 192 --tag label_k12
python3 scripts/leakage_audit.py --window 192 --label-k 1.2
python3 scripts/holdout_coins_eval.py --holdout-symbols TURBOUSDT --epochs 30 --skip-download

# 论文 PDF
cd paper && make pdf
```

---

## 实证支撑（共享市场状态）

时间对齐的 15m 收益相关（8 币，$n \approx 69{,}119$）：

- 全样本平均配对相关：**0.725**
- BTC 高波动子样本：**0.782**
- BTC 低波动子样本：**0.620**

脚本：`scripts/paper_mechanism_figures.py` → `paper/figures/meme_return_correlation.png`

---

## Meme 币种池

`DOGEUSDT`, `SHIBUSDT`, `PEPEUSDT`, `WIFUSDT`, `BONKUSDT`, `FLOKIUSDT`, `BOMEUSDT`, `1000SATSUSDT`  
外部确认：`TURBOUSDT`（及脚本支持 MOG / POPCAT / NEIRO）

---

## 文档与复现

| 文档 | 内容 |
| --- | --- |
| [REPORT.md](REPORT.md) | 研究历程、全部实验表、审计与 walk-forward 结果 |
| [paper/main.tex](paper/main.tex) | 正式论文（双协议叙事） |
| `outputs/metrics/*_summary.json` | 数值以 JSON 为准 |

**指标 JSON（冻结 k=1.2, w=192）**

- 严格：`outputs/metrics/meme8_*_wf7085_label_k12_summary.json`
- 开发：`outputs/metrics/meme8_*_loso_label_k12_summary.json`（或 `*_w192_loso_*`）
- 相关矩阵：`outputs/metrics/meme_return_correlation.json`

**切勿**将开发协议 AUC（0.74）与严格协议 AUC（0.51）混报为同一「泛化能力」。
