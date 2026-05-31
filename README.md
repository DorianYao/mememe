# mememe — Meme 币 15m 方向预测与跨域评估

> 8 个 Meme 币、动态波动率标签、Leave-One-Symbol-Out（LOSO）；**区分开发协议与严格 calendar walk-forward 协议**。

本项目从「单币 BTC 二分类」演进为 **跨币种零样本评估**。核心结论（与 [REPORT.md](REPORT.md) 一致）：

| 评估协议 | 冻结配置 `k=1.2, w=192`（MLP，meme8） | 含义 |
| --- | ---: | --- |
| **开发协议**（ratio-LOSO，held-out **全时段**） | AUC **0.743** ± 0.072 | 超参探索、消融、机制分析；**非**跨域泛化主结论 |
| **严格协议**（calendar walk-forward，仅测 **>85% 未来**） | AUC **0.510** ± 0.014 | **主结论**：向前外推 ≈ 随机 |
| 外部确认（NEIRO / PNUT，未参与调参） | AUC **0.496 / 0.509** | 严格协议；**不用 TRUMP**（政治叙事） |
| 外部确认 TURBO（历史） | AUC **≈ 0.497** | 2026-05-28 |

**分域进展**（[REPORT_CATEGORY.md](REPORT_CATEGORY.md)）：五大类 × 16 币 Phase A 已完成；**base_eco Phase B + 严格验证已完成**（严格 AUC **0.511**）。其余四类 strict **未重跑**（由 meme8/base_eco 先例外推 ≈0.51）。论文产物见 `paper/main.tex`、`paper/figures/`。

| 五类 strict（2026-05-31） | 说明 |
| --- | --- |
| base_eco | **实测** 0.511（k=1.5,w=224） |
| bluechip / midcap / solana_fast / micro_cap | **外推** ≈0.51（不另跑 16 轮 LOSO） |

泄漏审计 **PASS**（无 future-bar 泄露）；开发协议高分主要来自 **训练池与 held-out 共享日历区间**（共享市场状态），而非实现错误。详见 **[REPORT.md](REPORT.md)** 与 **[REPORT_CATEGORY.md](REPORT_CATEGORY.md)**。

---

## 核心设计

- **数据**：Binance 现货 15m OHLCV，8 币 × 约 720 天（`data/raw/`，首次运行可 `--stage download` 拉取）
- **特征**：15 维去币种化（形态、波动、RSI/MACD、taker 等，见 `src/config.py` → `feature_columns`）
- **标签**：`|r_{t+1}| > k·σ_t` 的方向事件（`label_mode=volatility`；小波动丢弃）
- **模型**：MLP 为主；XGBoost / 1D CNN 作对照（`--model mlp|cnn|xgboost|both`）
- **评估**：ROC-AUC（主指标）、MCC、Macro-F1；开发协议下可用 `scripts/backtest_pnl*.py` 做 PnL 回测

---

## 运行模式

`main.py` 支持三种模式（默认 **`--mode loso`**）：

| 模式 | 说明 |
| --- | --- |
| `single` | 单币（默认配置 `BTCUSDT`，可用 `--symbol` 覆盖） |
| `pooled` | 8 币池化，按时间切 train/val/test |
| `loso` | 留一币种：训练其余 N−1 币，测试 held-out 币（主实验） |

阶段：`--stage download|features|train|evaluate|all`（默认 `all`）。

### 五大类 × 16 币（独立 LOSO）

按 [advice/ClassificationSuggestion.md](advice/ClassificationSuggestion.md) 划分研究类别，**每类 16 币、类内独立训练**（不跨类池化）。币单见 [`config/categories.yaml`](config/categories.yaml)；上线前运行：

```bash
python3 scripts/validate_binance_symbols.py
```

| `category_id` | 含义 |
| --- | --- |
| `bluechip` | 蓝筹 Meme |
| `midcap` | 中盘趋势 Meme |
| `solana_fast` | Solana 高频 Meme |
| `base_eco` | 新生态 Meme（Binance 可交易代理） |
| `micro_cap` | 超小盘 Meme（Binance 策展） |
| `meme8` | Legacy 8 币对照（`--category meme8`） |

```bash
# 单类严格协议（16 轮 LOSO）
python3 main.py --mode loso --category bluechip --stage all --model mlp \
  --label-k 1.2 --window-size 192 --split-mode calendar --ablation-tag label_k12

# 五大类顺序跑满 + 跨类对比
bash scripts/run_all_categories.sh

# 单类单轮 smoke
python3 main.py --mode loso --category bluechip --held-out DOGEUSDT \
  --stage all --model mlp --label-k 1.2 --window-size 192 --split-mode calendar \
  --ablation-tag label_k12
```

输出命名：`{category_id}16_{interval}_w{window}_loso_{HELDOUT}_wf7085_{ablation}_*`（legacy 仍为 `meme8_*`）。  
最新实验数据：`outputs/metricsB/metrics/`；跨类汇总见 [REPORT_CATEGORY.md](REPORT_CATEGORY.md)。

### base_eco k×w 扫描（Phase B 已完成）

| 配置 | AUC | test_n | 判定 |
| --- | ---: | ---: | --- |
| **k=1.5, w=224**（可信峰值） | **0.738** | 125,839 | 建议冻结 → 严格验证 |
| k=1.2, w=208 | 0.734 | 201,533 | 备选（样本更充裕） |
| k=1.2, w=192（Phase A） | 0.715 | 202,152 | 已被超越 |
| k=2.5, w=240 | 0.812 | 28,947 | 低样本伪高分，勿用 |

```bash
# 类内 k×w 扫描（开发协议）
bash scripts/run_category_kw_scan.sh base_eco
python3 scripts/category_kw_compare.py --category base_eco

# Phase B 扩展峰值搜索（26 combo）
bash scripts/run_category_kw_extended.sh base_eco
python3 scripts/category_kw_extended_compare.py --category base_eco

# 严格 calendar 验证（待执行，可信峰值）
python3 main.py --mode loso --category base_eco --stage all --model mlp \
  --label-k 1.5 --window-size 224 --split-mode calendar --ablation-tag label_k15
```

---

## 两种 LOSO 协议（必读）

| 协议 | CLI / 配置 | 训练 | 测试（held-out 币） |
| --- | --- | --- | --- |
| **开发**（默认 `split_mode=ratio`） | `--split-mode ratio` | 其余 7 币池化，时间前 85% / 验证末 15% | held-out **全样本时段**（与训练池日历可重叠） |
| **严格**（主结论） | `--split-mode calendar` | 其余 7 币，$t \le 70\%$ / 验证 70–85% | held-out **仅** $t > 85\%$，且 $\max(\text{train})<\min(\text{test})$ |

严格协议在输出文件名中带 `wf7085`（由 `time_train_fraction=0.70`、`time_val_fraction=0.85` 生成）。

```bash
# 严格协议主实验（论文主指标）
python3 main.py --mode loso --stage all --model mlp \
  --label-k 1.2 --window-size 192 --split-mode calendar \
  --ablation-tag label_k12

# 开发协议（超参扫描 / 对照；省略 --split-mode 即 ratio）
python3 main.py --mode loso --stage all --model mlp \
  --label-k 1.2 --window-size 192 --ablation-tag label_k12

# 只跑某一 held-out 轮
python3 main.py --mode loso --stage all --model mlp \
  --held-out DOGEUSDT --label-k 1.2 --window-size 192
```

消融：用 `--ablation-tag TAG` 区分输出；`--exclude-features` / `--include-features` 改特征子集。标签 $\sigma$ 窗口可与特征波动窗口解耦：`--label-volatility-window`。

---

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

依赖见 `requirements.txt`：`numpy`, `pandas`, `scikit-learn`, `torch>=2.1`, `xgboost`, `matplotlib`, `scipy`, `pytest`, `PyYAML`, `requests` 等。下载 K 线走 Binance 公开接口，**无需** API key。

默认超参见 `config/default.yaml`（如 `window_size: 20`, `label_k: 0.3`, `split_mode: ratio`）；CLI 参数会覆盖 YAML。

---

## 项目结构

```
mememe/
├── main.py                 # single / pooled / loso 入口
├── REPORT.md               # 实验全记录（单一入口）
├── config/default.yaml
├── config/categories.yaml   # 五大类 × 16 币
├── scripts/                # 扫描、对比、审计、作图（见下节）
├── src/
│   ├── config.py           # ExperimentConfig、路径与 universe_tag
│   ├── data.py             # Binance 下载与 OHLCV 清洗
│   ├── features.py         # 去币种化特征、窗口、标签
│   ├── multi.py            # pooled / LOSO 切分（ratio & calendar）
│   ├── models.py           # MLP、1D CNN
│   ├── tabular.py          # XGBoost
│   ├── train.py            # 训练与 majority baseline
│   ├── evaluate.py         # 指标、LOSO 聚合、预测 CSV
│   └── plots.py            # 训练曲线、ROC、LOSO 汇总图
├── tests/test_loso_splits.py
├── data/raw/               # {SYMBOL}_15m_720d.csv（gitignore）
├── data/processed/         # .npz 窗口数据（gitignore）
└── outputs/
    ├── models/             # .pt / .joblib（gitignore）
    ├── metrics/            # JSON、CSV、predictions（gitignore）
    └── figures/            # main.py 评估图（gitignore）
```

运行 `scripts/generate_paper_figures.py` 或 `scripts/paper_mechanism_figures.py` 会在本地创建 **`paper/figures/`**（已在 `.gitignore`，需先有 `outputs/metrics/` 实验结果）。

---

## 脚本一览

| 类别 | 脚本 |
| --- | --- |
| **主流程** | `main.py` |
| **五大类 k×w** | `run_category_kw_scan.sh`, `run_category_kw_extended.sh`, `category_kw_compare.py`, `category_kw_extended_compare.py`, `run_all_category_kw_scans.sh` |
| **超参 / 对比** | `run_label_k_scan.sh`, `run_window_scan_k1.sh`, `run_window_fine_k12.sh`, `run_window_extended_k12.sh`, `run_pruned_kw_scan.sh`, `validate_w192.sh`, `label_k_compare.py`, `window_scan_compare.py`, `window_extended_compare.py`, `pruned_kw_compare.py`, `ablation_compare.py`, `walk_forward_loso_compare.py`, `run_label_decouple.py` |
| **机制与论文图** | `paper_mechanism_figures.py`, `generate_paper_figures.py` |
| **统计 / regime / PnL** | `statistical_significance.py`, `regime_analysis.py`, `confidence_threshold_scan.py`, `backtest_pnl.py`, `backtest_pnl_realistic.py` |
| **审计与外部币** | `leakage_audit.py`, `holdout_coins_eval.py` |

---

## 常用命令

```bash
# LOSO 基线（开发协议，默认 k/w）
python3 main.py --mode loso --stage all --model mlp --label-k 0.3 --window-size 20

# 超参扫描（开发协议）
bash scripts/run_label_k_scan.sh
bash scripts/run_window_scan_k1.sh
bash scripts/run_pruned_kw_scan.sh

# 机制图与论文图（需已有 metrics）
python3 scripts/paper_mechanism_figures.py
python3 scripts/generate_paper_figures.py

# 统计检验 / regime / 审计
python3 scripts/statistical_significance.py --window 192 --tag label_k12
python3 scripts/regime_analysis.py --window 192 --tag label_k12
python3 scripts/leakage_audit.py --window 192 --label-k 1.2
python3 scripts/walk_forward_loso_compare.py

# 外部 holdout（默认严格 calendar；可改 --holdout-symbols）
python3 scripts/holdout_coins_eval.py --holdout-symbols TURBOUSDT --epochs 30 --skip-download

# LOSO 切分单元测试
pytest tests/test_loso_splits.py -q
```

---

## 实证支撑（共享市场状态）

时间对齐的 15m 收益相关（8 币，$n \approx 69{,}119$）：

- 全样本平均配对相关：**0.725**
- BTC 高波动子样本：**0.782**
- BTC 低波动子样本：**0.620**

脚本：`scripts/paper_mechanism_figures.py` → `paper/figures/meme_return_correlation.png`（及 `outputs/metrics/meme_return_correlation.json`）

---

## Meme 币种池

**开发宇宙**（`config/default.yaml`）：  
`DOGEUSDT`, `SHIBUSDT`, `PEPEUSDT`, `WIFUSDT`, `BONKUSDT`, `FLOKIUSDT`, `BOMEUSDT`, `1000SATSUSDT`

**外部 holdout**（与 legacy 8 币同类、**非**政治叙事）：推荐 `NEIROUSDT`, `PNUTUSDT`；历史曾用 `TURBOUSDT`。**勿用 `TRUMPUSDT`** 作 8 币外部确认。脚本默认仍为 `TURBOUSDT`, `MOGUSDT`, `POPCATUSDT`, `NEIROUSDT`（`--holdout-symbols` 可覆盖）。

---

## 文档与复现

| 文档 / 路径 | 内容 |
| --- | --- |
| [REPORT.md](REPORT.md) | meme8 研究历程、双协议主结论、审计与 walk-forward |
| [REPORT_CATEGORY.md](REPORT_CATEGORY.md) | 五大类 × 16 币、base_eco Phase B 扩展扫描 |
| [docs/remote_workflow.md](docs/remote_workflow.md) | 远程 Pod 训练、tmux、打包拉回 |
| `config/default.yaml` | 默认超参与 8 币列表 |
| `outputs/metricsB/metrics/` | 最新分域实验 JSON / log（需先跑实验或从远程拉回） |

**指标 JSON（meme8，冻结 k=1.2, w=192）**

- 严格：`outputs/metrics/meme8_*_wf7085_label_k12_summary.json`
- 开发：`outputs/metrics/meme8_*_loso_label_k12_summary.json`（无 `wf7085` 后缀）
- 相关矩阵：`outputs/metrics/meme_return_correlation.json`

**指标 JSON（base_eco，开发协议 Phase B）**

- Phase A 汇总：`outputs/metricsB/metrics/category_kw_optimal.json`
- Phase B 汇总：`outputs/metricsB/metrics/category_kw_extended_optimal.json`
- 可信峰值明细：`outputs/metricsB/metrics/base_eco16_15m_w224_loso_ratio_label_k15_summary.json`

**切勿**将开发协议 AUC（meme8 ≈0.74、base_eco ≈0.74）与严格协议 AUC（≈0.51）混报为同一「泛化能力」。
