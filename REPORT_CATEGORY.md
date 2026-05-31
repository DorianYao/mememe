# 五大类分域实验报告（承接 REPORT.md）

**项目**：15 分钟 K 线下一根涨跌二分类（MLP）  
**前置报告**：[REPORT.md](REPORT.md)（legacy **meme8** × 8 币、双协议主结论、严格 calendar AUC ≈ 0.51）  
**本报告范围**：2026-05-28 起的 **五大研究类 × 16 币** 独立 LOSO、类内 $k \times w$ 调参  
**报告更新**：2026-05-31（**规划验证与四组 PnL 已全部完成**，见 §8）  

> 主产物：`outputs/metrics/`（严格验证、holdout、ratio 复跑、PnL JSON）；Phase A/B 扫描亦在 `outputs/metricsB/metrics/`。  
> **与 REPORT.md 的关系**：meme8 严格协议 ≈ 随机 **不变**；base_eco 严格复测 **同样 ≈ 0.51**，开发协议 AUC 0.68–0.75 **不可外推**。

---

## 1. 摘要

### 1.1 本阶段做了什么

| 里程碑 | 状态 | 说明 |
| --- | --- | --- |
| 41 币 OHLCV 下载（五大类并集） | ✅ | `scripts/download_all_category_data.py` |
| **五类 Phase A** 基线 $k \times w$ 精扫（各 5 组） | ✅ | ratio-LOSO，16 held-out / 组 |
| **base_eco Phase B** 扩展峰值搜索（$k \to 2.5$, $w \to 288$） | ✅ **已完成** | 26 combo，16/16 LOSO 全完成 |
| bluechip / midcap / solana_fast / micro_cap Phase B | ⏸ 暂停 | Phase A 已定各类 interim 最优，扩展扫留待 base_eco 严格验证后 |
| base_eco 规划验证（严格 + holdout + 四组 PnL） | ✅ | 严格/holdout + 四组 ratio 预测与 PnL/τ 扫描（16/16） |
| 五大类其余类 strict calendar | ⏸ | 按退出条件暂停 |

**Phase B 结论（2026-05-29）**：

- **可信峰值**（test_n ≥ 80k）：**k=1.5, w=224** → AUC **0.7383**，较 Phase A interim（k=1.2, w=192, 0.7151）**+2.3 pp**。
- **原始最高分** k=2.5, w=240（AUC 0.8119）因 test_n 仅 **29k**（−86% vs Phase A），判定为 **低样本伪高分**，与 meme8 上 k=2.0 教训一致。
- k≥1.8 区域普遍 AUC 偏高但样本崩塌（test_n < 80k），**不可作为冻结配置**。

### 1.2 五类 Phase A 横向对比（开发协议 ratio-LOSO）

| 类别 | 中文 | interim 最优 $(k,w)$ | mean AUC | mean MCC | mean F1 | test_n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **base_eco** | 新生态 Meme | **1.2 / 192** | **0.7151** | **0.3081** | **0.6535** | 202,152 |
| bluechip | 蓝筹 Meme | 1.2 / 128 | 0.6744 | 0.2473 | 0.6227 | 192,576 |
| solana_fast | Solana 高频 Meme | 1.2 / 192 | 0.6434 | 0.2023 | 0.5988 | 180,104 |
| midcap | 中盘趋势 Meme | 1.2 / 96 | 0.6378 | 0.1921 | 0.5956 | 180,878 |
| micro_cap | 超小盘 Meme | 1.2 / 96 | 0.6332 | 0.1860 | 0.5921 | 178,801 |

汇总 JSON：`outputs/metricsB/metrics/category_kw_optimal.json`（Phase A）、`category_kw_extended_optimal.json`（Phase B）

**解读**：

- **base_eco 显著领先**（比第二名 bluechip 高 **+4.1 pp AUC**），且 MCC / F1 同步最高。
- 五类 Phase A interim 最优均为 **k=1.2**，但最优窗口因类而异：base_eco / solana_fast 偏长（192），midcap / micro_cap 偏短（96），bluechip 居中（128）。
- base_eco Phase B 证实：**适度提高 k（1.2→1.5）并延长窗口（192→224）** 可在保持充足样本（test_n ≈ 126k）下继续提升；但 k≥1.8 进入样本崩塌区，AUC 虚高不可信。

### 1.3 推荐配置速查（base_eco，开发协议）

| 用途 | k | w | 协议 | mean AUC | test_n |
| --- | ---: | ---: | --- | ---: | ---: |
| **可信峰值（建议冻结）** | **1.5** | **224** | ratio-LOSO | **0.7383** | 125,839 |
| k=1.2 延伸窗口局部峰 | 1.2 | 208 | ratio-LOSO | 0.7343 | 201,533 |
| Phase A interim | 1.2 | 192 | ratio-LOSO | 0.7151 | 202,152 |
| 原始最高分（低样本，勿用） | 2.5 | 240 | ratio-LOSO | 0.8119 † | 28,947 |
| 短窗口对照 | 1.2 | 128 | ratio-LOSO | 0.6395 | 205,109 |
| 低 $k$ 对照 | 1.0 | 128 | ratio-LOSO | 0.6056 | 280,490 |

† test_n < 80k 可信阈值，仅作对照。

---

## 2. 实验设计（相对 REPORT.md 的增量）

### 2.1 五大类划分

依据 [advice/ClassificationSuggestion.md](advice/ClassificationSuggestion.md)，每类 **16 个 Binance USDT 现货对**，**类内 LOSO**（15 训 + 1 测），**不跨类池化**。

| category_id | 中文 | 研究焦点 | 16 币示例 |
| --- | --- | --- | --- |
| `bluechip` | 蓝筹 Meme | 市场 Beta / 深流动性 | DOGE, SHIB, PEPE, BONK, WIF … |
| `midcap` | 中盘趋势 Meme | 动量 / breakout | BOME, DOGS, NOT, HMSTR, 1000SATS … |
| `solana_fast` | Solana 高频 Meme | 高频投机 | BONK, WIF, PNUT, 1000SATS, DOGS … |
| `base_eco` | 新生态 Meme | 叙事扩散（Binance 代理） | IO, PORTAL, PIXEL, ETHFI, EIGEN, XAI … |
| `micro_cap` | 超小盘 Meme | 异常 / 高波动 | BOME, 1000SATS, DOGS, NOT, HMSTR … |

配置详情：[config/categories.yaml](config/categories.yaml)

输出命名：`{category}16_15m_w{window}_loso_{ratio|wf7085}_{ablation}_*`

### 2.2 两阶段 $k \times w$ 协议（每类独立）

| 阶段 | split | 矩阵 | 脚本 |
| --- | --- | --- | --- |
| **A 基线精扫** | ratio（开发） | **5 组**：$k \in \{1.0,1.2\}$，$w \in \{96,128,192\}$ 剪枝 | `scripts/run_category_kw_scan.sh` |
| **B 扩展峰值** | ratio | **26 组**：$k \in \{1.0,1.2,1.5,1.8,2.0,2.5\}$，$w \in \{192,208,\ldots,288\}$ 剪枝 | `scripts/run_category_kw_extended.sh` |
| **C 严格验证** | calendar wf7085 | 冻结 $(k,w)$ 后单点复跑 | base_eco **已完成**（§8） |

矩阵定义：`scripts/category_kw_matrix.py`

**可信峰值规则**（`scripts/category_kw_extended_compare.py`）：

- 16 轮 LOSO 完整；
- `test_n ≥ 80,000`（避免高 $k$ 样本崩塌伪高分）；
- 在已测网格上为 **局部极大**。

### 2.3 工程优化

扫描阶段启用 `--fast --skip-existing --dataloader-workers 0`：

- `batch_size` 自动按 GPU 档位（entry 1024 → high 65536）；
- 跳过绘图与 predictions CSV；
- 已有 `.npz` / metrics 断点续跑；
- combo 完成后自动删 npz/pt（`AUTO_CLEANUP_NPZ=1`）。

---

## 3. base_eco 基线精扫（Phase A，已完成）

**协议**：ratio-LOSO，MLP only，16 held-out / combo  
**日志**：`outputs/metricsB/metrics/base_eco_kw_scan_ratio.log`  
**汇总**：`outputs/metricsB/metrics/category_kw_optimal.json` → `base_eco` 段

### 3.1 全矩阵结果

| k | w | 回看 | mean AUC | std | mean MCC | test_n | Δ vs k=1.0,w=96 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 96 | 24h | 0.6009 | 0.039 | 0.1405 | 282,999 | — |
| 1.0 | 128 | 32h | 0.6056 | 0.044 | 0.1478 | 280,490 | +0.5 pp |
| 1.2 | 96 | 24h | 0.5889 | 0.030 | 0.1227 | 206,848 | −1.2 pp |
| 1.2 | 128 | 32h | 0.6395 | 0.081 | 0.1962 | 205,109 | +3.9 pp |
| **1.2** | **192** | **48h** | **0.7151** | **0.067** | **0.3081** | **202,152** | **+11.4 pp ★** |

热力图：`outputs/figures/category_kw_heatmap_ratio.png`（五类合图）

### 3.2 最优配置各 held-out（k=1.2, w=192, ratio-LOSO）

**汇总 JSON**：`outputs/metricsB/metrics/base_eco16_15m_w192_loso_ratio_label_k12_summary.json`

| Held-out | ROC-AUC | MCC | Test 样本 |
| --- | ---: | ---: | ---: |
| PORTALUSDT | **0.8199** | **0.4689** | 12,563 |
| PIXELUSDT | 0.8138 | 0.4586 | 13,824 |
| ETHFIUSDT | 0.8089 | 0.4566 | 13,798 |
| REZUSDT | 0.7995 | 0.4340 | 11,976 |
| XAIUSDT | 0.7875 | 0.4119 | 13,409 |
| ZKUSDT | 0.7137 | 0.2971 | 14,121 |
| EIGENUSDT | 0.7186 | 0.3145 | 11,012 |
| AEVOUSDT | 0.7031 | 0.2882 | 11,968 |
| ACEUSDT | 0.6961 | 0.2808 | 10,941 |
| MANTAUSDT | 0.6836 | 0.2539 | 13,260 |
| NFPUSDT | 0.6723 | 0.2382 | 11,946 |
| IOUSDT | 0.6637 | 0.2325 | 12,147 |
| LISTAUSDT | 0.6613 | 0.2323 | 11,481 |
| BBUSDT | 0.6497 | 0.2136 | 12,463 |
| ZROUSDT | 0.6386 | 0.1958 | 13,712 |
| ALTUSDT | 0.6107 | 0.1531 | 13,531 |

**观察**：

- 16/16 held-out AUC > 0.61，MCC 均为正；**PORTAL / PIXEL / ETHFI / REZ / XAI** 领先。
- 弱 held-out：**ALT / ZRO / BB**（AUC 0.61–0.65），仍高于随机。
- base_eco 宇宙为 L2/Launch 叙事 alt 代理（非真实 Base 链 meme），但信号一致性最好。

### 3.3 $k \times w$ 协同（base_eco 实证）

1. **短窗口 + 高 $k$ 反效果**：$k=1.2, w=96$（0.589）**低于** $k=1.0, w=96$（0.601），说明 base_eco 高 $k$ 标签必须配长上下文。
2. **长窗口跃升**：$k=1.2$ 在 $w=128 \to 192$ 提升 **+7.6 pp**（0.640 → 0.715），为五类中最大窗口增益。
3. **样本 trade-off 可接受**：$k=1.2, w=192$ 时 test_n ≈ 202k，远高于可信阈值 80k。

---

## 4. 扩展峰值搜索（Phase B，base_eco，已完成）

**动机**：Phase A 在 $k=1.2, w=192$ 处为矩阵内最高；向 **$k=2.5$、$w=288$** 延伸以确认峰值可信，并补充 **$k \in [1.2, 2.5]$** 全段测试。

**矩阵**（26 组，跳过 Phase A 已有 `k=1.2,w=192`）：

| k | w（bars） | tag | 说明 |
| ---: | ---: | --- | --- |
| 1.0 | 192 | label_k10 | 补 Phase A 缺口 |
| 1.5 | 192, 208, 224, 240, 256 | label_k15 | 新 $k$ @ 基准/延伸窗口 |
| 1.8 | 192, 224, 240, 256, 272 | label_k18 | 同上 |
| 2.0 | 192, 224, 240, 256, 272 | label_k20 | 同上 |
| 2.5 | 192, 240, 256, 272, 288 | label_k25 | 同上 |
| 1.2 | 208, 224, 240, 256, 272 | label_k12 | 固定 $k$ 延伸 $w$ |

**进度**（2026-05-29）：

| 项目 | 状态 |
| --- | --- |
| Phase A（5 组） | ✅ 16/16 完成 |
| Phase B（26 组） | ✅ **26/26 完成** |
| 可信峰值 | **k=1.5, w=224**（AUC 0.7383，test_n=125,839） |
| 原始最高分 | k=2.5, w=240（AUC 0.8119，test_n=28,947，**低样本**） |

**日志**：`outputs/metricsB/metrics/base_eco_kw_extended_ratio.log`（完成于 2026-05-29T15:29 UTC）  
**汇总**：`outputs/metricsB/metrics/category_kw_extended_optimal.json`

### 4.1 扩展区域全矩阵（k≥1.0，ratio-LOSO）

| k | w | 回看 | mean AUC | std | mean MCC | test_n | 备注 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | 96 | 24h | 0.6009 | 0.039 | 0.1405 | 282,999 | Phase A |
| 1.0 | 128 | 32h | 0.6056 | 0.044 | 0.1478 | 280,490 | Phase A |
| 1.0 | 192 | 48h | 0.6712 | 0.046 | 0.2428 | 276,154 | Phase B 补测 |
| 1.2 | 96 | 24h | 0.5889 | 0.030 | 0.1227 | 206,848 | Phase A |
| 1.2 | 128 | 32h | 0.6395 | 0.081 | 0.1962 | 205,109 | Phase A |
| 1.2 | 192 | 48h | 0.7151 | 0.067 | 0.3081 | 202,152 | Phase A ★ |
| 1.2 | 208 | 52h | 0.7343 | 0.063 | 0.3368 | 201,533 | k=1.2 局部峰 |
| 1.2 | 224 | 56h | 0.7273 | 0.076 | 0.3304 | 200,952 | |
| 1.2 | 240 | 60h | 0.6729 | 0.077 | 0.2492 | 200,335 | w>224 回落 |
| 1.2 | 256 | 64h | 0.6721 | 0.090 | 0.2491 | 199,619 | |
| 1.2 | 272 | 68h | 0.6621 | 0.096 | 0.2343 | 199,076 | |
| 1.5 | 192 | 48h | 0.6908 | 0.068 | 0.2732 | 126,609 | |
| 1.5 | 208 | 52h | 0.7230 | 0.066 | 0.3216 | 126,216 | |
| **1.5** | **224** | **56h** | **0.7383** | **0.060** | **0.3451** | **125,839** | **可信峰值 ★** |
| 1.5 | 240 | 60h | 0.6946 | 0.074 | 0.2767 | 125,428 | |
| 1.5 | 256 | 64h | 0.6845 | 0.088 | 0.2689 | 124,953 | |
| 1.8 | 192 | 48h | 0.6968 | 0.062 | 0.2830 | 79,420 | 低样本 |
| 1.8 | 224 | 56h | 0.7411 | 0.074 | 0.3496 | 78,929 | 低样本 |
| 1.8 | 240 | 60h | 0.7784 | 0.070 | 0.4067 | 78,681 | 低样本 |
| 1.8 | 256 | 64h | 0.7709 | 0.052 | 0.3956 | 78,386 | 低样本 |
| 1.8 | 272 | 68h | 0.6437 | 0.068 | 0.2035 | 78,202 | 低样本 |
| 2.0 | 192 | 48h | 0.6759 | 0.035 | 0.2502 | 58,848 | 低样本 |
| 2.0 | 224 | 56h | 0.7015 | 0.056 | 0.2900 | 58,498 | 低样本 |
| 2.0 | 240 | 60h | 0.8037 | 0.037 | 0.4470 | 58,305 | 低样本 |
| 2.0 | 256 | 64h | 0.7404 | 0.068 | 0.3492 | 58,078 | 低样本 |
| 2.0 | 272 | 68h | 0.6599 | 0.027 | 0.2267 | 57,943 | 低样本 |
| 2.5 | 192 | 48h | 0.7123 | 0.028 | 0.3074 | 29,226 | 低样本 |
| 2.5 | 240 | 60h | 0.8119 | 0.041 | 0.4754 | 28,947 | 原始最高 † |
| 2.5 | 256 | 64h | 0.7442 | 0.047 | 0.3669 | 28,830 | 低样本 |
| 2.5 | 272 | 68h | 0.6965 | 0.026 | 0.2871 | 28,758 | 低样本 |
| 2.5 | 288 | 72h | 0.6999 | 0.028 | 0.2897 | 28,669 | 低样本 |

† AUC 最高但 test_n 远低于 80k 可信阈值。

### 4.2 可信峰值各 held-out（k=1.5, w=224, ratio-LOSO）

**汇总 JSON**：`outputs/metricsB/metrics/base_eco16_15m_w224_loso_ratio_label_k15_summary.json`

| Held-out | ROC-AUC | MCC | Test 样本 |
| --- | ---: | ---: | ---: |
| AEVOUSDT | **0.8708** | **0.5712** | 7,213 |
| BBUSDT | 0.7966 | 0.4339 | 7,790 |
| PORTALUSDT | 0.7833 | 0.4232 | 7,851 |
| NFPUSDT | 0.7761 | 0.4071 | 7,438 |
| ACEUSDT | 0.7722 | 0.3982 | 6,773 |
| EIGENUSDT | 0.7651 | 0.3968 | 6,982 |
| MANTAUSDT | 0.7611 | 0.3744 | 8,276 |
| IOUSDT | 0.7464 | 0.3541 | 7,418 |
| ALTUSDT | 0.7375 | 0.3322 | 8,455 |
| XAIUSDT | 0.7356 | 0.3349 | 8,293 |
| ZKUSDT | 0.7243 | 0.3132 | 8,691 |
| LISTAUSDT | 0.7150 | 0.3101 | 7,258 |
| REZUSDT | 0.6902 | 0.2613 | 7,489 |
| PIXELUSDT | 0.6789 | 0.2447 | 8,606 |
| ETHFIUSDT | 0.6500 | 0.2085 | 8,671 |
| ZROUSDT | 0.6100 | 0.1576 | 8,635 |

**观察**：

- 16/16 held-out AUC > 0.61，MCC 均为正；**AEVO / BB / PORTAL / NFP** 领先。
- 弱 held-out：**ZRO / ETHFI / PIXEL**（AUC 0.61–0.68）。
- 提高 k 至 1.5 后单币 test 样本约 **6.8k–8.7k**（Phase A k=1.2 约 11k–14k），总 test_n 仍充足。

### 4.3 Phase B 规律总结

1. **k=1.2 延伸窗口**：$w=192 \to 208$ 提升 **+1.9 pp**（0.715 → 0.734），$w>224$ 开始回落，与 meme8 上 $w>192$ 回落模式类似但峰值更晚（208 vs 192）。
2. **k 与 w 协同延续**：$k=1.5, w=224$ 在可信样本区内为全局最优，较 Phase A **+2.3 pp**。
3. **高 k 样本陷阱再现**：$k \ge 1.8$ 时 test_n 跌破 80k，AUC 可飙至 0.78–0.81，但 **不可信**（meme8 k=2.0 @ 33k 样本教训）。
4. **建议冻结**：**k=1.5, w=224** 进入严格 calendar 验证；k=1.2, w=208 可作为样本更充裕的备选（AUC 0.7343，test_n=201k）。

**复现**：

```bash
bash scripts/run_category_kw_extended.sh base_eco
python3 scripts/category_kw_extended_compare.py --category base_eco \
  --output outputs/metricsB/metrics/category_kw_extended_optimal.json
python3 scripts/category_kw_extended_compare.py --category base_eco --plot
```

**产物**：

- `outputs/metricsB/metrics/category_kw_extended_optimal.json`
- `outputs/metricsB/metrics/base_eco_kw_extended_ratio.log`
- `outputs/figures/base_eco_kw_extended_heatmap_ratio.png`（若已生成）

---

## 5. 其他四类（Phase A 完成，Phase B 暂停）

| 类别 | Phase A 最优 | AUC | Phase B |
| --- | ---: | ---: | --- |
| bluechip | k=1.2, w=128 | 0.6744 | ⏸ 本地已终止，待 base_eco 定峰 |
| solana_fast | k=1.2, w=192 | 0.6434 | ⏸ |
| midcap | k=1.2, w=96 | 0.6378 | ⏸ |
| micro_cap | k=1.2, w=96 | 0.6332 | ⏸ |

各类 Phase A 日志：`outputs/metricsB/metrics/{category}_kw_scan_ratio.log`

---

## 6. 与 REPORT.md 主结论的对齐说明

| 主题 | REPORT.md（meme8） | 本报告（base_eco，2026-05-31） |
| --- | --- | --- |
| **严格 calendar** | AUC ≈ **0.510** | **0.5114 ± 0.0178**（k=1.5,w=224）；对照 **0.5052 ± 0.0152**（k=1.2,w=208） |
| **外部 holdout** | TURBO 等 ≈ 0.51 | NEIRO **0.511** / PNUT **0.515** / TURBO **0.518**（17 训 + 1 测） |
| **开发 ratio-LOSO** | AUC ≈ **0.743** | 四组 **0.703–0.806**（见 §8）；相对严格 **−20～−25 pp** |
| 泄漏审计 | PASS | **PASS**（`leakage_audit_w224_k15.json`，`--skip-poison`） |
| PnL（realistic, τ=0.70） | 开发协议有正收益 | 四组均为正（见 §8.3）；**仅开发协议，非严格主指标** |

**结论**：严格 + 外部 holdout 与 meme8 一致，**无法声称跨域可交易 α**；开发协议 PnL 为正 **不满足** [规划(动态调整的).txt](advice/规划(动态调整的).txt) 三重通过标准 → **转向** [备用研究方向建议.md](advice/备用研究方向建议.md)。

---

## 7. 下一步（规划退出后）

1. ~~base_eco 严格 calendar + holdout~~ → **已完成，未通过**。
2. ~~四组配置 PnL / τ~~ → **已完成**（`scripts/run_pnl_finish.sh`；产物见 §8.3）。
3. **暂停** GeminiAdvice 实战系统与其余四类 Phase B（除非改研究问题）。
4. 磁盘维护：combo 完成后执行 `python3 scripts/cleanup_disk.py --npz-completed`（本次释放 **~120GB** `.npz`）。

---

## 8. 规划验证结果（2026-05-30/31，已完成）

依据 [advice/规划(动态调整的).txt](advice/规划(动态调整的).txt)；日志 `outputs/metrics/base_eco_validation_resume.log`、`base_eco_pnl.log`。

### 8.1 双协议 AUC（base_eco，MLP，16/16 LOSO）

| 配置 | split | mean AUC | std | test_n（ratio） | 严格 Δ（vs 同 k 严格或 0.51） |
| --- | --- | ---: | ---: | ---: | ---: |
| **k=1.5, w=224** | calendar wf7085 | **0.5114** | 0.0178 | — | — |
| k=1.2, w=208 | calendar wf7085 | 0.5052 | 0.0152 | — | — |
| **k=1.5, w=224** | ratio（开发） | 0.7326 | 0.0443 | 125,839 | **−22.1 pp** |
| k=1.2, w=208 | ratio | 0.7522 | 0.0659 | 201,533 | **−24.7 pp** |
| k=1.2, w=192 | ratio | 0.7030 | 0.0784 | 202,152 | −19.8 pp |
| k=2.5, w=240 | ratio | 0.8064 | 0.0475 | **28,947** | −29.5 pp † |

† w=240 的 test_n **仅 29k**（< 80k 可信阈值），AUC 0.81 仍为 **低样本伪高分**；与 Phase B 扫描结论一致。

**协议落差**：开发 − 严格 ≈ **20–25 pp**（可信配置），与 meme8 先例一致；`leakage_audit_w224_k15.json` → **PASS**（`--skip-poison`）。

### 8.2 外部 holdout（k=1.5, w=224, calendar）

| 测试币 | test AUC |
| --- | ---: |
| NEIROUSDT | 0.5110 |
| PNUTUSDT | 0.5153 |
| TURBOUSDT | 0.5177 |

### 8.3 开发协议 PnL（realistic，`backtest_realistic_w*.json`，τ=0.70）

| k | w | tag | total_return_net † | Sharpe_daily | n_trades | 预测轮次 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1.5 | 224 | label_k15 | 35.45 | 8.41 | 32,972 | 16/16 |
| 1.2 | 208 | label_k12 | 48.45 | 9.59 | 59,648 | 16/16 |
| 1.2 | 192 | label_k12 | 43.00 | 10.43 | 58,187 | 16/16 |
| 2.5 | 240 | label_k25 | 28.19 | 9.45 | 8,542 | 16/16 |

† 与 REPORT §13 同字段：`final_equity ≈ 1 + total_return_net`（固定名义本金、无复利）。  
配套：`confidence_threshold_w{window}_{tag}.json`、`backtest_pnl_w*.json`（idealized）。

**解读**：四组开发协议 PnL 均为正，但 **严格 + holdout AUC ≈ 0.51** → **不满足**规划「通过标准」；高 k/w=240 收益样本更少（test_n≈29k），不可作为冻结配置。

### 8.4 规划四组对照（开发 vs 严格）

| 配置 | ratio AUC | 严格 AUC | realistic PnL (τ=0.7) | 判定 |
| --- | ---: | ---: | ---: | --- |
| **k=1.5, w=224**（冻结峰） | 0.7326 | 0.5114 | +35.45 | 严格未过；PnL 不可外推 |
| k=1.2, w=208 | 0.7522 | 0.5052 | +48.45 | 同上 |
| k=1.2, w=192 | 0.7030 | — | +43.00 | Phase A 基准，已被 w224/w208 超越 |
| k=2.5, w=240 | 0.8064 | — | +28.19 | ❌ test_n<80k，仅对照 |

### 8.5 复现命令

```bash
bash scripts/run_base_eco_validation_resume.sh
bash scripts/run_base_eco_pnl.sh          # 或收尾：bash scripts/run_pnl_finish.sh
python3 scripts/leakage_audit.py --category base_eco --window 224 --label-k 1.5 --skip-poison
python3 scripts/cleanup_disk.py --npz-completed
```

---

## 附录 A：产物索引

| 类型 | 路径 |
| --- | --- |
| 五类 Phase A 汇总 | `outputs/metricsB/metrics/category_kw_optimal.json` |
| base_eco Phase B 汇总 | `outputs/metricsB/metrics/category_kw_extended_optimal.json` |
| base_eco 基线日志 | `outputs/metricsB/metrics/base_eco_kw_scan_ratio.log` |
| base_eco 扩展日志 | `outputs/metricsB/metrics/base_eco_kw_extended_ratio.log` |
| 基线热力图 | `outputs/figures/category_kw_heatmap_ratio.png` |
| 扩展热力图 | `outputs/figures/base_eco_kw_extended_heatmap_ratio.png` |
| 类别币单 | `config/categories.yaml` |
| 严格验证汇总 | `outputs/metrics/base_eco16_15m_w224_loso_wf7085_label_k15_summary.json` 等 |
| 四组 PnL / τ | `outputs/metrics/backtest_realistic_w*.json`、`confidence_threshold_w*.json` |
| 泄漏审计 | `outputs/metrics/leakage_audit_w224_k15.json` |
| 验证日志 | `outputs/metrics/base_eco_validation_resume.log`、`base_eco_pnl.log` |
| 远程操作手册 | [docs/remote_workflow.md](docs/remote_workflow.md) |

## 附录 B：严格验证（已执行，2026-05-30）

```bash
bash scripts/run_base_eco_validation_resume.sh   # P0/P1 严格 + holdout + ratio 四组
python3 scripts/leakage_audit.py --category base_eco --window 224 --label-k 1.5 --skip-poison
bash scripts/run_pnl_finish.sh                   # w192/w240 预测 + PnL 扫描（tmux）
```

## 附录 C：远程复现命令（base_eco Phase B，已完成）

```bash
cd /workspace/mememe && source .venv/bin/activate

# 1. 磁盘清理（训练前）
python3 scripts/cleanup_disk.py --status
python3 scripts/cleanup_disk.py --npz-completed

# 2. 启动扩展扫描（26 combo，断点续跑）
tmux new -s base_eco
bash scripts/run_category_kw_extended.sh base_eco

# 3. 查看进度
tail -f outputs/metrics/base_eco_kw_extended_ratio.log

# 4. 汇总 + 热力图
python3 scripts/category_kw_extended_compare.py --category base_eco
python3 scripts/category_kw_extended_compare.py --category base_eco --plot

# 5. 打包拉回
tar czf mememe_metrics.tgz outputs/metrics/
```

高端 GPU 可加速：

```bash
LOSO_JOBS=8 NPZ_JOBS=12 BATCH_SIZE=65536 bash scripts/run_category_kw_extended.sh base_eco
```

---

*分类扫描数值以 `outputs/metricsB/metrics/*_summary.json` 为准；规划验证以 `outputs/metrics/*_summary.json` 为准。*
