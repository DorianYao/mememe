# finPaper 实验报告

**项目**：15 分钟 K 线下一根涨跌二分类（MLP + 1D CNN）  
**核心实验**：8 个 Meme 币 Leave-One-Symbol-Out（LOSO）跨币种泛化  
**数据区间**：Binance 现货，各币约 720 天 × 15m  
**报告更新**：2026-05-28（含 **严格 calendar walk-forward** 全 8 币、TURBO 外部确认、收益相关矩阵、论文 `main.tex` 双协议叙事）

> 完整实验日志与 JSON 汇总见 `outputs/metrics/`；本报告为 **研究历程 + 全部实验结果** 的单一入口。  
> **跨域泛化主结论以严格协议为准**；开发协议 AUC 仅作探索与对照。

---

## 1. 摘要

本项目从「单币种 BTC 下一根 K 线涨跌预测」演进为 **8 个 Meme 币 Leave-One-Symbol-Out（LOSO）** 评估框架：动态波动率标签、去币种化特征、MLP 主模型。

### 1.1 双协议结论（冻结 `k=1.2`, `w=192`）

| 协议 | MLP 8 币平均 AUC | 说明 |
| --- | ---: | --- |
| **开发**（ratio-LOSO，held-out 全时段） | **0.7434** ± 0.0720 | 超参选定、消融、regime 分层、统计检验（$p<0.001$） |
| **严格**（calendar walk-forward，仅未来 15%） | **0.5100** ± 0.0141 | **主结论**：向前外推 ≈ 随机 |
| 外部确认 TURBO（未参与 $k$/$w$ 扫描） | **≈ 0.497** | 严格协议 |

**协议落差约 −23.3 pp**；泄漏审计 **PASS**（无 future-bar 泄露）。落差归因于 **共享日历区间下的板块共动（共享市场状态）**，而非代码泄漏。8 币 15m 收益时间对齐后：平均配对相关 **0.725**（高波动子样本 **0.782**）。

汇总 JSON：`meme8_*_wf7085_label_k12_summary.json`（严格）、`meme8_*_label_k12_summary.json`（开发，无 `wf7085` 后缀）。

### 1.2 开发协议下的探索结论（不可直接外推）

- **标签工程 >> 特征工程**：fixed_label 消融 −0.63 pp；扩展技术指标 ROI < 0.3 pp。
- **$k$ × $w$ 协同**：$w=192$ 为开发峰值；$w>192$ 回落。
- **MLP ≈ XGBoost**（严格协议差异 < 2 pp）；CNN 长窗口不适用。
- **开发协议 PnL**（$\tau=0.70$，30 bps）：约 +42.9% / 2y，Sharpe(日) ≈ 11.5；$t{+}2$ + spread 后大幅恶化，**不宜**作为严格外推可交易性依据。

### 1.3 推荐配置速查

| 用途 | k | w | 协议 |
| --- | ---: | ---: | --- |
| **论文主指标** | 1.2 | 192 | **严格** calendar |
| 超参探索 / 对照 | 1.2 | 192 | 开发 ratio-LOSO |
| 低开销基线 | 0.5 | 20 | 开发 |

---

## 2. 研究历程与方向演进

本节按时间线记录 **最初做法 → 为何改方向 → 逐步优化 → 当前工作**，避免遗漏。

### 2.1 阶段 A：单币 BTC baseline（起点）

**做法**：

- 模式：`--mode single`，BTCUSDT，时间切分 70/15/15。
- 特征：初版 11 维 → 增强 13 维（加入 taker 买卖强度）。
- 标签：初版固定 `ε=0.0005` → 改为动态波动率阈值 `k=0.3`。
- 模型：MLP + 1D CNN 双模型对比。

**结果**：

| 版本 | MLP AUC | 说明 |
| --- | ---: | --- |
| 固定 ε 标签 | 0.5361 | 接近随机 |
| 动态波动率标签 | 0.5384 | 略有提升 |

**问题**：单币 train/test 同域，模型可能记忆 BTC 特有模式；AUC ~0.54 无法证明「跨币种可迁移」。

---

### 2.2 阶段 B：转向 meme 池 + LOSO（方向性转变）

**为何改方向**：

1. Paper 卖点需 **零样本新币泛化**，而非同币内插值预测。
2. Meme 币共享高波动、高散户参与等微观结构，适合验证「普适规律」。
3. 需 **去币种化特征**（volume z-score、比率型指标），防止模型用 raw volume 量级当币种 ID。

**做法**：

- 重构 pipeline：`src/multi.py` 实现池化 + LOSO 切分。
- 8 币 universe：DOGE, SHIB, PEPE, WIF, BONK, FLOKI, BOME, 1000SATS。
- 15 维去币种化特征 + `StandardScaler` 仅在训练池 fit。
- 默认：`k=0.3`, `window_size=20`, MLP/CNN 双模型。

**结果（LOSO baseline，2026-05-26 主跑）**：

| 模型 | ROC-AUC | MCC | 耗时 |
| --- | ---: | ---: | --- |
| **MLP** | **0.5664** ± 0.0136 | **0.0944** | ~23 min |
| CNN | 0.5611 ± 0.0130 | 0.0858 | （合计） |

8/8 held-out AUC > 0.54，MCC 均为正——**在更严格的 LOSO 协议下仍优于 BTC 单币 ~0.538**，说明多币池化确实学到了可迁移信号。

**日志**：`outputs/metrics/loso_full_run.log`

---

### 2.3 阶段 C：特征与标签消融（找杠杆）

**动机**：baseline 已正，需回答「该优化什么」——特征？标签？模型？

**做法**：4 组消融，各 8 轮 LOSO × 2 模型，`--ablation-tag` 隔离输出。

| ID | 改动 | MLP AUC | Δ |
| --- | --- | ---: | ---: |
| Baseline | 15 维 + 动态标签 | 0.5664 | — |
| no_volume_z | 去掉 volume_z_50 | 0.5643 | −0.21 pp |
| raw_volume | volume_z → volume_log | 0.5661 | −0.03 pp |
| no_indicators | 去掉 5 个技术指标 | 0.5644 | −0.20 pp |
| **fixed_label** | 固定 ε=0.0005 | **0.5601** | **−0.63 pp** |

**结论 → 方向调整**：

- **停止卷 feature**：边际 ΔAUC 均 < 0.3 pp。
- **聚焦 label**：fixed_label 是唯一稳健负向消融，8/8 held-out 全变差。
- 汇总脚本：`scripts/ablation_compare.py` → `outputs/figures/ablation_summary.png`
- **日志**：`outputs/metrics/ablation_chain.log`（~89 min）

---

### 2.4 阶段 D：`label_k` 系统扫描（第一杠杆）

**动机**：消融证明标签是主杠杆；`k=0.3` 为拍脑袋默认值，需系统扫描。

**做法**：k ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}，固定 w=20，LOSO × MLP/CNN。脚本 `scripts/run_label_k_scan.sh`，~1h52m。

| k | 含义 | test_n | MLP AUC | Δ vs k=0.3 |
| ---: | --- | ---: | ---: | ---: |
| 0.1 | 保留更多样本 | 486,511 | 0.5587 | −0.77 pp |
| 0.2 | 轻去噪 | 448,115 | 0.5620 | −0.44 pp |
| 0.3 | baseline | 403,804 | 0.5664 | — |
| **0.5** | **强去噪** | **316,984** | **0.5708** | **+0.44 pp** |
| 0.7 | 极强去噪 | 239,838 | 0.5652 | −0.12 pp |
| 1.0 | 只保留大波动 | 152,548 | 0.5684 | +0.20 pp |

**发现**：

- k 太小 → 噪声样本稀释信号（0.1 最差）。
- k=0.5 为 **通用场景 sweet spot**（样本仍够，MCC 首次 > 0.10）。
- k 继续增大时，单扫 k 收益有限（1.0 仅 +0.2 pp）。

**补测 k=2.0**（~2.5 min）：AUC 0.6287，但样本仅 3.3 万（−92%），不可与 k≤1.0 直接比。

**日志**：`outputs/metrics/label_k_scan.log`, `label_k20.log`

---

### 2.5 阶段 E：`window_size` 扫描（第二杠杆，与 k 协同）

**动机**：label_k 扫描发现 k=1.0 时 AUC 略升但样本大减；假设 **大波动事件需要更长历史上下文**，故在 k=1.0 下扫 window。

**做法**：w ∈ {20, 40, 60, 96}，固定 k=1.0。脚本 `scripts/run_window_scan_k1.sh`，~1h01m。

| window | 回看 | MLP AUC | Δ vs w=20 | MLP MCC | CNN AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 5h | 0.5684 | — | 0.0971 | 0.5604 |
| 40 | 10h | 0.5936 | +2.52 pp | 0.1301 | 0.5554 |
| 60 | 15h | 0.5891 | +2.07 pp | 0.1219 | 0.5475 |
| **96** | **24h** | **0.6372** | **+6.88 pp** | **0.1912** | 0.5371 |

**结论 → 方向再次调整**：

- **k × window 协同**：长窗口只在 k≥1.0 下爆发式提升；window 扫描阶段最优为 k=1.0,w=96（0.637）。
- **CNN 放弃**：w=96 时 CNN AUC 0.537；后续实验 **仅跑 MLP**。

**日志**：`outputs/metrics/window_scan_k1.log`

---

### 2.6 阶段 F：基于规律剪枝 + k×window 定向精扫（已完成）

**动机**（见 `advice2.md`）：

- 已有足够实验支撑 **剪枝**，不应再全参数暴力搜索。
- 高价值区域：k ∈ [0.5, 1.2]，window ∈ [40, 96]。
- 永久放弃：CNN、k < 0.3、window < 20、k=2.0 大规模扫描、继续卷 feature。

**做法**：8 组矩阵，**仅 MLP**，跳过已有 summary。脚本 `scripts/run_pruned_kw_scan.sh`，~57 min（20:42–21:39）。

| k | window | 回看 | MLP AUC | std | MLP MCC | test_n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 40 | 10h | 0.5740 | 0.010 | 0.1041 | 316,528 |
| 0.5 | 80 | 20h | 0.5851 | 0.013 | 0.1183 | 315,634 |
| 0.7 | 40 | 10h | 0.5777 | 0.015 | 0.1080 | 239,461 |
| 0.7 | 80 | 20h | 0.5936 | 0.022 | 0.1308 | 238,722 |
| 1.0 | 40 | 10h | 0.5936 | 0.023 | 0.1301 | 152,282 |
| 1.0 | 80 | 20h | 0.6019 | 0.035 | 0.1445 | 151,838 |
| 1.0 | 96 | 24h | 0.6372 | 0.038 | 0.1912 | 151,658 |
| **1.2** | **96** | **24h** | **0.7156** | 0.057 | **0.3063** | **110,599** |

**主要发现**：

1. **低 k + 长窗口无效**：k=0.5 从 w=40→80 仅 +1.1 pp（0.574→0.585），远低于 k=1.0 同期（0.594→0.602）。
2. **k 与 window 同向递增**：在同一 w=96 下，k 从 1.0→1.2 带来 +7.8 pp AUC（0.637→0.716）。
3. **全局最优 k=1.2, w=96**：MCC 0.306、Macro-F1 0.653；7/8 held-out AUC > 0.67，SHIB 偏弱（0.605）。
4. **样本 trade-off**：k=1.2 使 test 样本从 152k 降至 111k（−27%），std(AUC)=0.057 高于 k=1.0,w=96 的 0.038。

**k=1.2, w=96 各 held-out（MLP）**：

| Held-out | ROC-AUC | MCC | Test 样本 |
| --- | ---: | ---: | ---: |
| PEPEUSDT | **0.7908** | **0.4264** | 13842 |
| FLOKIUSDT | 0.7687 | 0.3888 | 14134 |
| BOMEUSDT | 0.7646 | 0.3865 | 14026 |
| BONKUSDT | 0.7316 | 0.3248 | 14086 |
| DOGEUSDT | 0.7094 | 0.2888 | 13392 |
| 1000SATSUSDT | 0.6799 | 0.2450 | 13618 |
| WIFUSDT | 0.6741 | 0.2415 | 13553 |
| SHIBUSDT | 0.6054 | 0.1483 | 13948 |

图：`outputs/figures/pruned_kw_scan_heatmap.png`  
日志：`outputs/metrics/pruned_kw_scan.log`

---

### 2.7 性能演进总表（MLP LOSO mean AUC）

| 里程碑 | 配置 | AUC | 累计 Δ | 关键洞察 |
| --- | --- | ---: | ---: | --- |
| BTC 单币 | 固定 ε, w=20 | 0.5361 | — | 起点 |
| BTC 单币 | 动态标签, w=20 | 0.5384 | +0.2 pp | 动态标签略好 |
| **LOSO baseline** | k=0.3, w=20 | **0.5664** | +2.8 pp | 跨币泛化成立 |
| 标签消融 | fixed ε | 0.5601 | −0.6 pp | 标签是主杠杆 |
| label_k 扫描 | k=0.5, w=20 | 0.5708 | +0.4 pp | 通用最优 k |
| label_k 扫描 | k=1.0, w=20 | 0.5684 | +0.2 pp | 大波动标签 |
| window 扫描 | k=1.0, w=40 | 0.5936 | +2.7 pp | 长窗口开始生效 |
| **window 扫描** | k=1.0, w=96 | 0.6372 | +7.1 pp | k×window 协同 |
| 定向精扫 | k=0.5, w=40 | 0.5740 | +0.8 pp | 长窗口对低 k 无效 |
| 定向精扫 | k=1.0, w=80 | 0.6019 | +3.6 pp | w=80 介于 40–96 |
| **定向精扫** | **k=1.2, w=96** | **0.7156** | **+14.9 pp** | **当前全局最优** |
| k=2.0 探测 | k=2.0, w=20 | 0.6287 † | +6.3 pp | 样本量仅 8%，不可比 |
| **Confidence 过滤** | k=1.2, w=96, τ=0.70 | MCC **0.530** | — | 无需重训，coverage 43% |
| **实盘 PnL** | k=1.2, w=96, τ=0.70 | **+36%** / 2y | — | open→close + 30 bps + max 3 仓 |

---

### 2.8 阶段 G：Confidence threshold 与 PnL 回测（已完成）

**动机**：AUC 0.716 已足够强，需验证概率能否转化为可交易收益；低置信度预测应弃权。

**做法**：

1. **Confidence threshold 扫描**（无需重训）：对 8 轮 LOSO 预测 CSV 做后处理，`prob≥τ` 做多、`prob≤1−τ` 做空，中间弃权。脚本 `scripts/confidence_threshold_scan.py`。
2. **理想化 PnL 回测**：close→close、20 bps 手续费。脚本 `scripts/backtest_pnl.py`。
3. **实盘约束 PnL**：t 收盘信号 → t+1 open 进 / t+1 close 出；手续费 10 bps + 滑点 5 bps；最多 3 仓（按 `|prob−0.5|` 取 Top）；固定名义本金、不复利。脚本 `scripts/backtest_pnl_realistic.py`。

**Confidence 扫描（k=1.2,w=96，pooled）**：

| τ | coverage | accuracy | MCC |
| ---: | ---: | ---: | ---: |
| 0.50 | 100% | 0.654 | 0.308 |
| 0.60 | 64.0% | 0.717 | 0.434 |
| **0.70** | **43.4%** | **0.765** | **0.530** |
| 0.75 | 35.3% | 0.788 | 0.576 |
| 0.85 | 20.7% | 0.837 | 0.675 |

**实盘约束 PnL（k=1.2,w=96，max 3 仓，30 bps 往返）**：

| τ | coverage | 扣费 bps/trade | 胜率 | Sharpe(日) | maxDD | 2y 总收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 50% | 15.2 | 68.1% | 6.9 | −21.5% | +28.1% |
| **0.70** | **37%** | **26.8** | **72.9%** | **9.6** | **−11.7%** | **+36.2%** |
| 0.75 | 31% | 33.5 | 75.6% | 10.4 | −9.0% | **+37.8%** |
| 0.85 | 19% | 50.1 | 81.4% | 10.8 | −5.1% | +34.7% |

**结论**：

- Confidence 过滤显著抬升 MCC（0.308 → 0.530 @ τ=0.70），证明 `prob_up` 有排序与弃权价值。
- 实盘约束下 τ=0.70–0.75 为 sweet spot；τ=0.50 虽 AUC 高但 PnL 风险大（maxDD −52%）。
- k=1.2 全面优于 k=1.0（同 τ=0.70：+36% vs +16% 总收益）。
- 滑点压力测试（40 bps 往返，τ=0.70）：仍 +22.7% 总收益，edge 未消失。

产物：`outputs/metrics/confidence_threshold_w96_label_k12.json`，`backtest_pnl_w96_label_k12.json`，`backtest_realistic_w96_label_k12.json`；图 `backtest_realistic_w96_label_k12.png`。

---

### 2.9 累计实验清单（无遗漏）

| # | 实验 | 模式 | 模型 | 组数×轮数 | 状态 | 日志 / 脚本 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | BTC 单币 baseline | single | MLP+CNN | 2 版 | ✅ | 对话记录 |
| 2 | LOSO 主实验 | loso | MLP+CNN | 1×8 | ✅ | `loso_full_run.log` |
| 3 | 特征/标签消融 ×4 | loso | MLP+CNN | 4×8 | ✅ | `ablation_chain.log` |
| 4 | label_k 扫描 ×6 | loso | MLP+CNN | 5×8 | ✅ | `label_k_scan.log` |
| 5 | k=2.0 补测 | loso | MLP+CNN | 1×8 | ✅ | `label_k20.log` |
| 6 | window 扫描 ×4 | loso | MLP+CNN | 3×8 | ✅ | `window_scan_k1.log` |
| 7 | k×window 定向精扫 ×8 | loso | **MLP only** | 8 组 | ✅ | `pruned_kw_scan.log` |
| 8 | Confidence threshold 扫描 | 后处理 | MLP | 8 τ × 2 tag | ✅ | `confidence_threshold_scan.py` |
| 9 | PnL 回测（理想 + 实盘约束） | 后处理 | MLP | k12/k10 | ✅ | `backtest_pnl*.py` |

**汇总脚本**：`ablation_compare.py`, `label_k_compare.py`, `window_scan_compare.py`, `pruned_kw_compare.py`, `confidence_threshold_scan.py`, `backtest_pnl.py`, `backtest_pnl_realistic.py`

**累计 CPU 时间（估算）**：~8 小时训练 + 后处理 < 1 min。

---

### 2.10 阶段 H：严格 calendar walk-forward LOSO（主结论，2026-05-28）

**动机**：开发协议下 held-out 使用 **全时段**，训练池与测试币 **日历重叠**，AUC 0.74 可能含「共享市场冲击」而非向前可迁移规律。论文提出 **共享时间市场状态假说**；需用严格时间外推验证。

**做法**（`--split-mode calendar`，`split_tag=wf7085`）：

- 全局时间轴：训练 ≤ **70%**，验证 70–85%，测试 held-out 仅 **> 85%**。
- 强制 $\max(\text{train}) < \min(\text{test})$。
- 超参 **冻结**：$k=1.2$, $w=192$（仅在开发协议上选定）。

**命令**：

```bash
python3 main.py --mode loso --stage all --model mlp \
  --label-k 1.2 --window-size 192 --split-mode calendar --ablation-tag label_k12
```

**结果（MLP，8 轮）**：

| 指标 | 值 |
| --- | ---: |
| 平均 ROC-AUC | **0.5100** ± 0.0141 |
| 平均 MCC | 0.0085 |
| 单币范围 | 0.495 – 0.533 |

| Held-out | AUC | n_test |
| --- | ---: | ---: |
| SHIBUSDT | 0.5330 | 2261 |
| WIFUSDT | 0.5247 | 1559 |
| PEPEUSDT | 0.5216 | 2092 |
| BOMEUSDT | 0.5120 | 2174 |
| DOGEUSDT | 0.4977 | 2008 |
| FLOKIUSDT | 0.4993 | 2175 |
| BONKUSDT | 0.4960 | 2205 |
| 1000SATSUSDT | 0.4946 | 2010 |

**对照**：

| 模型 | DOGE 严格 AUC |
| --- | ---: |
| MLP | 0.4977 |
| XGBoost | 0.5170 |

**外部确认**：TURBOUSDT（9 币宇宙 LOSO，未参与调参）AUC **≈ 0.497**（`holdout_confirm_k12`）。

**机制证据**：`scripts/paper_mechanism_figures.py` → 收益相关矩阵（全样本 0.725；高波动 0.782）；`paper/figures/loso_protocol_timeline.png`。

**解读**：严格协议切断日历重叠后，跨币 **向前预测** 接近随机；开发协议 0.743 主要度量 **同期可排序性**。

---

### 2.11 阶段 I：统计检验、regime、审计与论文整合

| 项目 | 结果 | 协议范围 |
| --- | --- | --- |
| Bootstrap / 置换 | pooled AUC ≈ 0.75，$p<0.001$ | **仅开发** |
| DeLong w192 vs w96 | Δ ≈ +3.1 pp | **仅开发** |
| Regime 分层 | 高波动 AUC ≈ 0.79；BTC 偏弱 ≈ 0.79 | **仅开发** pooled |
| Leakage 审计 | **PASS** | 实现层 |
| 标签–特征解耦（DOGE smoke） | 严格 AUC 未提升 | 严格 |

论文源文件：`paper/main.tex`；复现见附录复现命令。

---

## 3. 任务定义

| 项目 | 设定 |
| --- | --- |
| 输入 | 过去 `N` 根 15m K 线 × `F=15` 维去币种化特征 |
| 输出 | 二分类：下一根 **显著方向**（涨/跌） |
| 标签 | `y=1` 若 `r_{t+1} > k·σ_t`；`y=0` 若 `r_{t+1} < -k·σ_t`；否则丢弃 |
| 论文冻结超参 | **k=1.2**, **w=192** |
| **开发** LOSO 切分 | 训练：7 币池化前 85%；验证：池化末 15%；测试：held-out **全时段** |
| **严格** LOSO 切分 | 训练：7 币 $t \le T_{train}$；验证：$T_{train} < t \le T_{val}$；测试：held-out 仅 $t > T_{val}$（约末 15%） |
| 标准化 | `StandardScaler` 仅在训练集 fit |

---

## 4. 模型与训练配置

| 项目 | MLP | 1D CNN |
| --- | --- | --- |
| 输入 | `(batch, N, F)` → 展平 `N×F` | `(batch, F, N)` |
| 结构 | Linear×3 + BN + ReLU + Dropout | Conv1d×2 + Pool + FC |
| 参数量（N=20,F=15） | ~109k | ~12k |
| 损失 | BCEWithLogits + 动态 pos_weight | 同左 |
| 优化 | AdamW, lr=1e-3, wd=1e-4 | 同左 |
| 早停 | val macro-F1, patience=8 | 同左 |

**当前建议**：新实验 **仅跑 MLP**（`--model mlp`），CNN 已证明不适合本任务尤其长窗口。

默认：`batch_size=256/512`, `epochs=50`, `dropout=0.3`, `random_state=42`。

---

## 5. 特征（15 维，全部去币种化）

| 类别 | 特征 |
| --- | --- |
| 价格 | `log_return` |
| K 线形态 | `upper_wick_ratio`, `lower_wick_ratio`, `body_ratio`, `body_abs_ratio` |
| 成交量 | `volume_z_50`（50 根滚动 z-score，**跨币可比**） |
| 波动 | `volatility_50`, `atr_14_ratio` |
| 趋势 | `ma_10_ratio`, `ema_10_ratio` |
| 动量 | `rsi_14`, `macd_hist` |
| 通道 | `bb_position` |
| 买卖强度 | `taker_buy_ratio`, `taker_buy_ratio_change` |

消融已证：去掉 indicators / 换 raw volume，ΔAUC < 0.3 pp。**不建议继续扩展特征维度**。

---

## 6. Meme 币种池

`DOGEUSDT`, `SHIBUSDT`, `PEPEUSDT`, `WIFUSDT`, `BONKUSDT`, `FLOKIUSDT`, `BOMEUSDT`, `1000SATSUSDT`

数据：`data/raw/{SYMBOL}_15m_720d.csv`（Binance 公开 API，无需 key）

---

## 7. 单币 BTCUSDT 结果（对照）

| 版本 | MLP AUC | CNN AUC | 样本 test |
| --- | ---: | ---: | ---: |
| 固定 ε | 0.5361 | 0.5307 | 10362 |
| 动态标签 + taker | 0.5384 | 0.5396 | 7796 |

单币 AUC ~0.54；LOSO 更难但 MLP 达 0.5664，说明跨币训练有效。

---

## 8. LOSO 主实验（Baseline：k=0.3, w=20）

**命令**：`python3 main.py --mode loso --stage all --model both --batch-size 512`

| 模型 | ROC-AUC | MCC | Macro-F1 |
| --- | ---: | ---: | ---: |
| **MLP** | **0.5664** ± 0.0136 | **0.0944** ± 0.0210 | 0.5465 |
| CNN | 0.5611 ± 0.0130 | 0.0858 ± 0.0192 | 0.5417 |

**MLP 各 held-out AUC**：PEPE 0.596 > WIF 0.578 > SHIB 0.565 > … > DOGE 0.549

---

## 9. 特征与标签消融

| 实验 | tag | MLP AUC | Δ |
| --- | --- | ---: | ---: |
| Baseline | — | 0.5664 | — |
| no_volume_z | `no_volume_z` | 0.5643 | −0.21 pp |
| raw_volume | `raw_volume` | 0.5661 | −0.03 pp |
| no_indicators | `no_indicators` | 0.5644 | −0.20 pp |
| **fixed_label** | `fixed_label` | **0.5601** | **−0.63 pp** |

图：`outputs/figures/ablation_summary.png`

---

## 10. `label_k` 超参扫描（w=20 固定）

| k | test_n | MLP AUC | MLP MCC | CNN AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 486,511 | 0.5587 | 0.0835 | 0.5553 |
| 0.2 | 448,115 | 0.5620 | 0.0886 | 0.5622 |
| 0.3 | 403,804 | 0.5664 | 0.0944 | 0.5611 |
| **0.5** | **316,984** | **0.5708** | **0.1007** | 0.5618 |
| 0.7 | 239,838 | 0.5652 | 0.0916 | 0.5620 |
| 1.0 | 152,548 | 0.5684 | 0.0971 | 0.5604 |
| 2.0 † | 32,651 | 0.6287 | 0.1842 | 0.5996 |

图：`outputs/figures/label_k_scan_summary.png`  
复现：`bash scripts/run_label_k_scan.sh`

---

## 11. `window_size` 超参扫描（k=1.0 固定）

| window | 回看 | MLP AUC | MLP MCC | CNN AUC |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 5h | 0.5684 | 0.0971 | 0.5604 |
| 40 | 10h | 0.5936 | 0.1301 | 0.5554 |
| 60 | 15h | 0.5891 | 0.1219 | 0.5475 |
| **96** | **24h** | **0.6372** | **0.1912** | 0.5371 |

**MLP held-out × window（AUC）**：

| 币 | w=20 | w=40 | w=60 | w=96 |
| --- | ---: | ---: | ---: | ---: |
| BOME | 0.558 | 0.564 | 0.582 | **0.695** |
| WIF | 0.603 | 0.602 | 0.591 | **0.667** |
| PEPE | 0.584 | 0.629 | 0.608 | **0.656** |
| BONK | 0.580 | 0.617 | 0.589 | 0.576 |

图：`outputs/figures/window_scan_k1_summary.png`  
复现：`bash scripts/run_window_scan_k1.sh`

---

## 12. k×window 定向精扫（advice2，已完成）

**设计原则**：剪枝后只扫 8 组，MLP only，跳过已有 summary。耗时 ~57 min（20:42–21:39）。

### 12.1 完整结果表

| k | w | 回看 | MLP AUC | std | MLP MCC | Macro-F1 | test_n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 40 | 10h | 0.5740 | 0.010 | 0.1041 | 0.5514 | 316,528 |
| 0.5 | 80 | 20h | 0.5851 | 0.013 | 0.1183 | 0.5581 | 315,634 |
| 0.7 | 40 | 10h | 0.5777 | 0.015 | 0.1080 | 0.5536 | 239,461 |
| 0.7 | 80 | 20h | 0.5936 | 0.022 | 0.1308 | 0.5645 | 238,722 |
| 1.0 | 40 | 10h | 0.5936 | 0.023 | 0.1301 | 0.5642 | 152,282 |
| 1.0 | 80 | 20h | 0.6019 | 0.035 | 0.1445 | 0.5711 | 151,838 |
| 1.0 | 96 | 24h | 0.6372 | 0.038 | 0.1912 | 0.5944 | 151,658 |
| **1.2** | **96** | **24h** | **0.7156** | 0.057 | **0.3063** | **0.6528** | **110,599** |

### 12.2 规律总结

1. **k↑ + w↑ 同向提升**：8 组内 AUC 随 k、window 增大整体单调上升（k=0.5,w=40 最低，k=1.2,w=96 最高）。
2. **长窗口需大 k**：k=0.5,w=80（0.585）仍低于 k=1.0,w=40（0.594）。
3. **k=1.2 是精扫范围内最优**：较 k=1.0,w=96 +7.8 pp AUC，MCC 从 0.191 升至 0.306。
4. **跨币方差增大**：k=1.2,w=96 的 std(AUC)=0.057；SHIB 仅 0.605，其余 7 币 > 0.67。

### 12.3 k=1.2, w=96 各 held-out（MLP）

| Held-out | ROC-AUC | MCC | Test 样本 |
| --- | ---: | ---: | ---: |
| PEPEUSDT | **0.7908** | **0.4264** | 13842 |
| FLOKIUSDT | 0.7687 | 0.3888 | 14134 |
| BOMEUSDT | 0.7646 | 0.3865 | 14026 |
| BONKUSDT | 0.7316 | 0.3248 | 14086 |
| DOGEUSDT | 0.7094 | 0.2888 | 13392 |
| 1000SATSUSDT | 0.6799 | 0.2450 | 13618 |
| WIFUSDT | 0.6741 | 0.2415 | 13553 |
| SHIBUSDT | 0.6054 | 0.1483 | 13948 |

图：`outputs/figures/pruned_kw_scan_heatmap.png`  
复现：`bash scripts/run_pruned_kw_scan.sh`；汇总：`python3 scripts/pruned_kw_compare.py`

### 12.4 后续验证（已完成，详见 §13）

Confidence threshold 扫描与 PnL 回测（含实盘约束版）已完成；结果支持 k=1.2,w=96 + τ=0.70 作为论文主推配置。

---

## 13. Confidence threshold 与 PnL 回测

### 13.1 Confidence threshold 扫描

**规则**：`prob≥τ` → 预测涨；`prob≤1−τ` → 预测跌；中间弃权。无需重训。

**k=1.2,w=96（pooled，110,599 样本）**：

| τ | coverage | n_kept | accuracy | MCC |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 100.0% | 110,599 | 0.654 | 0.308 |
| 0.55 | 79.8% | 88,264 | 0.687 | 0.375 |
| 0.60 | 64.0% | 70,821 | 0.717 | 0.434 |
| 0.65 | 52.7% | 58,227 | 0.741 | 0.483 |
| **0.70** | **43.4%** | **48,019** | **0.765** | **0.530** |
| 0.75 | 35.3% | 39,087 | 0.788 | 0.576 |
| 0.80 | 27.8% | 30,781 | 0.812 | 0.625 |
| 0.85 | 20.7% | 22,903 | 0.837 | 0.675 |

**k=1.0,w=96 对照（τ=0.70）**：coverage 15.1%，MCC 0.471（k12 为 0.530）。

复现：`python3 scripts/confidence_threshold_scan.py --tag label_k12 --window 96`

### 13.2 理想化 PnL 回测

close→close 成交，往返手续费 20 bps，等权 concurrent。上界参考，**不作实盘结论**。

**k=1.2,w=96（τ=0.70）**：扣费 **45.7 bps/trade**，胜率 **76.5%**，Sharpe **54.2**，maxDD **−10%**（加性 portfolio）。

复现：`python3 scripts/backtest_pnl.py --tag label_k12 --window 96`

### 13.3 实盘约束 PnL 回测（主推）

| 假设 | 设定 |
| --- | --- |
| 成交 | t 收盘出信号 → **t+1 open 进 → t+1 close 出** |
| 成本 | 手续费 10 bps + 滑点 5 bps = **30 bps 往返** |
| 仓位 | 最多 **3** 仓；超出按 `\|prob−0.5\|` 取 Top |
| 资金 | 固定名义本金（每仓 1/3），**不复利**，加性 equity |
| Sharpe | 日度 PnL / 初始资金，年化 √365 |

**k=1.2,w=96**：

| τ | trades | 扣费 bps | 胜率 | Sharpe(日) | maxDD | 2y 总收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 55,340 | 15.2 | 68.1% | 6.9 | −21.5% | +28.1% |
| **0.70** | **40,543** | **26.8** | **72.9%** | **9.6** | **−11.7%** | **+36.2%** |
| 0.75 | 33,792 | 33.5 | 75.6% | 10.4 | −9.0% | +37.8% |

**压力测试**：滑点加倍至 10 bps/边（40 bps 往返），τ=0.70 仍 **+22.7%** 总收益。

**k=1.0,w=96 对照（τ=0.70）**：+16.2% 总收益，coverage 仅 15%，Sharpe 7.3。

复现：`python3 scripts/backtest_pnl_realistic.py --tag label_k12 --window 96`

产物：`outputs/metrics/backtest_realistic_w96_label_k12.json`，`outputs/figures/backtest_realistic_w96_label_k12.png`

---

## 14. 推荐配置速查

| 场景 | k | window | 协议 | MLP AUC | 备注 |
| --- | ---: | ---: | --- | ---: | --- |
| **跨域泛化主结论** | **1.2** | **192** | **严格 calendar** | **0.510** | 8 币均值；勿与 0.74 混报 |
| 开发探索峰值 | 1.2 | 192 | ratio-LOSO | **0.743** | 超参扫描终点；对照用 |
| 开发备选（低 std） | 1.2 | 184 | ratio-LOSO | 0.737 | std ≈ 0.055 |
| 历史精扫峰值 | 1.2 | 96 | ratio-LOSO | 0.7156 | 2026-05-26 阶段 |
| 通用低开销 | 0.5 | 20 | ratio-LOSO | 0.5708 | 样本保留 ~78% |
| LOSO baseline | 0.3 | 20 | ratio-LOSO | 0.5664 | 初版 |
| 开发 PnL 参考 | 1.2 | 192 | ratio-LOSO + τ=0.70 | — | +42.9%/2y；非严格可交易上界 |

---

## 15. 图表与产物索引

### 15.1 汇总图

| 文件 | 内容 |
| --- | --- |
| `outputs/figures/ablation_summary.png` | 消融 × MLP/CNN |
| `outputs/figures/label_k_scan_summary.png` | label_k 扫描 |
| `outputs/figures/window_scan_k1_summary.png` | window 扫描（k=1.0） |
| `outputs/figures/pruned_kw_scan_heatmap.png` | k×window 精扫热力图 |
| `outputs/figures/backtest_realistic_w96_label_k12.png` | 实盘约束 PnL（τ 扫描） |
| `outputs/figures/meme8_*_loso_test_*.png` | LOSO baseline 各 held-out |

### 15.2 指标 JSON

| 类型 | 路径模式 |
| --- | --- |
| LOSO baseline | `meme8_*_loso_summary.json` |
| 消融 | `meme8_*_loso_{tag}_summary.json` |
| label_k | `meme8_*_loso_label_k{01,02,05,07,10,20}_summary.json` |
| window（k=1.0） | `meme8_*_w{20,40,60,96}_loso_label_k10_summary.json` |
| 精扫 | `meme8_*_w{40,80,96}_loso_label_k{05,07,10,12}_summary.json` |
| Confidence | `confidence_threshold_w96_{tag}.json` |
| PnL | `backtest_pnl_w96_{tag}.json`, `backtest_realistic_w96_{tag}.json` |
| Leakage | `leakage_audit_w192_k12.json`（PASS） |

### 15.3 脚本一览

| 脚本 | 用途 |
| --- | --- |
| `scripts/ablation_compare.py` | 消融汇总 |
| `scripts/run_label_k_scan.sh` | label_k 链式扫描 |
| `scripts/label_k_compare.py` | label_k 汇总图 |
| `scripts/run_window_scan_k1.sh` | window 扫描（k=1.0） |
| `scripts/window_scan_compare.py` | window 汇总图 |
| `scripts/run_pruned_kw_scan.sh` | k×window 定向精扫 |
| `scripts/pruned_kw_compare.py` | 精扫热力图 |
| `scripts/confidence_threshold_scan.py` | Confidence τ 扫描 |
| `scripts/backtest_pnl.py` | 理想化 PnL 回测 |
| `scripts/backtest_pnl_realistic.py` | 实盘约束 PnL 回测 |
| `scripts/leakage_audit.py` | Leakage / 因果性正式审计 |
| `scripts/generate_paper_figures.py` | 论文插图生成 |

### 15.4 复现命令

```bash
# LOSO baseline
python3 main.py --mode loso --stage all --model both

# 当前最优 AUC（k=1.2, w=96）
python3 main.py --mode loso --stage all --model mlp \
  --label-k 1.2 --window-size 96 --ablation-tag label_k12

# 稳健备选（k=1.0, w=96）
python3 main.py --mode loso --stage all --model mlp \
  --label-k 1.0 --window-size 96 --ablation-tag label_k10

# 批量扫描
bash scripts/run_label_k_scan.sh
bash scripts/run_window_scan_k1.sh
bash scripts/run_pruned_kw_scan.sh

# 汇总
python3 scripts/ablation_compare.py
python3 scripts/label_k_compare.py
python3 scripts/window_scan_compare.py
python3 scripts/pruned_kw_compare.py

# Confidence + PnL（无需重训）
python3 scripts/confidence_threshold_scan.py --tag label_k12 --window 96
python3 scripts/backtest_pnl.py --tag label_k12 --window 96
python3 scripts/backtest_pnl_realistic.py --tag label_k12 --window 192
python3 scripts/leakage_audit.py --window 192 --label-k 1.2
```

---

## 16. 讨论与局限

### 16.1 双协议下的结果解读

| 视角 | AUC 量级 | 解释 |
| --- | ---: | --- |
| **严格 calendar LOSO** | **~0.51** | 向前外推接近随机；**跨域泛化主结论** |
| **开发 ratio-LOSO** | **~0.57–0.74** | 度量同期可排序性 / 超参探索；**不可**等同于严格泛化 |
| BTC 单币 baseline | ~0.54 | 历史起点对照 |

- 15m 方向事件近 50:50；LOSO 比单币切分更严。
- 开发协议下 8/8 held-out 正 MCC、$k{=}1.2,w{=}192$ 均值 0.743；统计检验显著（§2.11）——均在 **日历可重叠** 设定下成立。
- **协议落差（−23 pp）** 与 8 币收益高相关（0.725）一致：开发高分主要来自 **共享市场状态**，而非实现泄漏（§16.3 **PASS**）。
- 开发协议 PnL（+42.9%/2y 等）仅说明 **同期** 排序—收益关系；$t{+}2$ 与 spread 下大幅恶化。

### 16.2 局限

1. **Leakage 正式审计 PASS**（w=192,k=1.2）：无 future-bar 泄露（§16.3）；**不能**解释开发 vs 严格 AUC 落差。
2. **严格 walk-forward 已完成**（§2.10）：主结论 AUC ≈ 0.51；开发协议回测仍非在线重训场景。
3. 概率未做 calibration（见 §17.4）。
4. Regime 分层仅在 **开发协议** pooled 上完成（§2.11）；严格协议下未分 regime。
5. 仅 8 meme 币 + TURBO 外部点；MOG/POPCAT/NEIRO 待扩展（§17.3）。
6. k=1.2 样本 ~110k，开发协议 std(AUC)≈0.072 偏高。
7. w=192 MLP 输入 2880 维，未做参数量对齐消融。
8. CNN / Transformer 未再投入（CNN 已证伪于本任务）。
9. **标签条件化**：`volatility_50[t]` 同时用于特征与标签阈值（非泄露，但样本非随机子集）。

### 16.3 Leakage / 因果性正式审计（2026-05-27，w=192,k=1.2）

**动机**：AUC 0.743 / PnL +43% 足够强，必须排除 48h 长窗口下 rolling 特征泄漏未来 bar 的可能。

**脚本**：`python3 scripts/leakage_audit.py --window 192 --label-k 1.2`  
**产物**：`outputs/metrics/leakage_audit_w192_k12.json`

**Verdict：PASS**（未发现 future-bar 泄露）

| 检查项 | 方法 | 结果 |
| --- | --- | --- |
| **1. 静态代码扫描** | `src/` 搜索 `center=True`、`shift(-k)`、`bfill` | **0 命中** |
| **2. 特征因果性（扰动）** | 截断 OHLCV 至 bar t，重算指标；200 个随机 t 对比 | **max diff = 0** |
| **3. 滑窗边界** | 张量行 ∈ `[t−191, t]`；标签 = `log(close[t+1]/close[t])` | **0 违规** |
| **4. Rolling 最大回看** | 最长指标 warmup = **50 bars**（volume_z / volatility）；均 backward | 48h 窗口 **不引入 t+1 数据** |
| **5. StandardScaler** | LOSO 仅在 `raw_train` fit；与 train+test 混合 fit 可区分 | **PASS**（diff=0 vs train） |
| **6. 毒化对照** | 注入 `future_return` 为第 16 维特征 → AUC 应 → 1.0 | clean **0.684** → poison **1.000**（检测有效） |

**48h window 专项结论**：

- 模型输入 192 根 bar 的特征矩阵，但**每行特征在 bar i 仅依赖 OHLCV ≤ i**。
- `rolling(50)` / `ewm(adjust=False)` 均为 pandas 默认**后向**窗口，无 `center=True`。
- 决策时刻 t = 窗口末 bar；标签严格来自 **t+1** 收益；特征窗口**不含 t+1**。

**保留注意事项（非 leakage，但影响解释）**：

1. **标签条件化**：动态标签用 `σ_t = volatility_50[t]` 过滤模糊样本，该列亦在特征中——属于 regime / 样本选择偏差，**不是未来信息**。
2. **Val 阈值调参**：`evaluate.py` 在 val 上搜 threshold 再用于 test，带来轻度 nested optimism，与 bar 级泄露无关。
3. **同 bar 收盘价**：特征含 `close[t]`、K 线形态等——对应「bar 收盘后出信号、预测下一根」，与回测成交假设一致。

**复现**：

```bash
python3 scripts/leakage_audit.py --window 192 --label-k 1.2
python3 scripts/leakage_audit.py --window 96 --label-k 1.2   # 可选：对照短窗口
```

---

## 17. 下一步建议

> **严格 calendar walk-forward（§2.10）与泄漏审计（§16.3）均已完成。** 下文为后续研究，非课程提交阻塞项。

### 17.1 ~~Leakage 审计~~ ✅ 已完成

`scripts/leakage_audit.py`；w=192,k=1.2 **PASS**（§16.3）。

### 17.2 ~~严格 Walk-forward LOSO~~ ✅ 已完成

8 币 MLP 均值 AUC **0.510**；TURBO 外部确认 **≈0.50**。实现：`--split-mode calendar`；测试 `tests/test_loso_splits.py`。

### 17.3 第一优先级：外部确认币扩展

MOG / POPCAT / NEIRO 等严格协议复评（`scripts/holdout_coins_eval.py`）。

### 17.4 第二优先级：概率校准（Probability Calibration）

**动机**：`prob_up` 已呈现明显 confidence–accuracy 单调关系（§13.1）；校准后可进一步优化 threshold 与仓位。

**建议做法**：

- 在 LOSO val 集上 fit **Platt scaling** 或 **Isotonic regression**。
- 对比校准前后：可靠性图（reliability diagram）、Brier score、τ=0.70 下 MCC / PnL 变化。
- 若校准后 threshold 策略更稳定，可写入论文主推配置。

### 17.5 第三优先级：Regime Classifier（两阶段模型）

**动机**：实验已表明 **regime（大 k、长 window）是核心杠杆**——并非所有时段同等可预测。

**建议架构**：

1. **Stage-1**：二分类「是否处于高可预测 regime」（可用波动率分位、|r|>kσ 比例、或自监督标签）。
2. **Stage-2**：仅在 Stage-1 判为 predictable 时，启用现有方向 MLP。

**评估**：对比单阶段 vs 两阶段在 coverage、MCC、实盘 PnL、maxDD 上的 trade-off。

### 17.6 第四优先级：市值层级实验

**动机**：验证 alpha 是否随 **流动性 / 市值 / 波动率** 变化；支持跨域泛化叙事。

**做法（不要无限扩币）**：

按类别分层，而非盲目加币：

| 层级 | 示例 | 假设 |
| --- | --- | --- |
| Large meme | DOGE, SHIB | 流动性好，滑点低，edge 可能被压缩 |
| Mid meme | PEPE, WIF, BONK, FLOKI | 当前主战场 |
| Micro meme | BOME, 1000SATS | 波动大，滑点高，AUC 可能高但 PnL 难 |

**分析维度**：各层 AUC、MCC、实盘 PnL（同 τ、同成本假设）；回答「alpha 是否随市值单调变化」。

**低优先级（暂缓）**：k=1.1/1.3 局部细扫、`no_taker` 消融、Transformer。

---

## 18. 结论

本项目完整记录了从 **BTC 单币 ~0.54 AUC** 到 **开发协议 LOSO 0.743** 的探索路径，并通过 **严格 calendar walk-forward** 将跨域泛化主结论修正为 **AUC ≈ 0.51（接近随机）**。

1. **问题定义主导性能**：动态标签 + 长窗口在 **开发协议** 下将 AUC 从 ~0.57 推至 0.74；**严格协议** 下同等冻结超参回落至 0.51。
2. **开发协议高分 ≠ 向前可迁移**：日历重叠时存在强收益相关（均值 **0.725**）；切断重叠后排序能力消失。
3. **实现无 future-bar 泄露**：审计 PASS；落差来自 **评估设定与共享市场状态**。
4. **架构非关键**：严格协议下 MLP 与 XGBoost 接近；CNN 长窗口不适用。
5. **开发协议 PnL** 仅说明同期样本上的排序—收益关系；延迟与 spread 敏感性高，不能外推为严格可交易性。

**论文叙事**（`paper/main.tex`）：协议批判 + 机制假说 + 负结果同样构成完整研究贡献。报告数值以 `outputs/metrics/*_summary.json` 为准。

---

## 附录 A：指标说明

| 指标 | 含义 | 重要性 |
| --- | --- | --- |
| ROC-AUC | 概率排序能力，阈值无关 | **最高** |
| MCC | 马修斯相关系数 | 高 |
| Macro-F1 | 两类 F1 平均 | 高 |
| Accuracy | 正确率 | 参考 |

---

## 附录 B：实验运行日志

| 日志 | 说明 | 耗时 |
| --- | --- | --- |
| `loso_full_run.log` | LOSO baseline | ~23 min |
| `ablation_chain.log` | 4 组消融 | ~89 min |
| `ablation_*.log` | 各消融 stdout | — |
| `label_k_scan.log` | k=0.1–1.0 扫描 | ~1h52m |
| `label_k20.log` | k=2.0 补测 | ~2.5 min |
| `window_scan_k1.log` | w=40/60/96 扫描 | ~1h01m |
| `pruned_kw_scan.log` | k×window 精扫 8 组 | ~57 min（20:42–21:39） |
| `confidence_threshold_w96_*.json` | Confidence τ 扫描 | < 1 min |
| `backtest_realistic_w96_*.json` | 实盘约束 PnL | ~15 s |

---

*数值以 `outputs/metrics/*_summary.json` 为准。*

- **严格主结论**：`meme8_*_wf7085_label_k12_summary.json`（AUC ≈ 0.510）
- **开发对照**：`meme8_*_loso_label_k12_summary.json` 或 `*_w192_loso_*_label_k12_*`（AUC ≈ 0.743）
- **相关矩阵**：`outputs/metrics/meme_return_correlation.json`
