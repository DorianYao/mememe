# 五大类分域实验报告（承接 REPORT.md）

**项目**：15 分钟 K 线下一根涨跌二分类（MLP）  
**前置报告**：[REPORT.md](REPORT.md)（legacy **meme8** × 8 币、双协议主结论、严格 calendar AUC ≈ 0.51）  
**本报告范围**：2026-05-28 起的 **五大研究类 × 16 币** 独立 LOSO、类内 $k \times w$ 调参  
**报告更新**：2026-05-28  

> 完整 JSON / 日志见 `outputs/metrics/`。  
> **与 REPORT.md 的关系**：§1–§18 的 meme8 主结论（尤其严格协议 ≈ 随机）**不变**；本报告记录 **分大类后的开发协议超参探索**，尚未对 bluechip 最优配置跑严格 calendar 验证。

---

## 1. 摘要

### 1.1 本阶段做了什么

| 里程碑 | 状态 | 说明 |
| --- | --- | --- |
| 41 币 OHLCV 下载（五大类并集） | ✅ | `scripts/download_all_category_data.py` |
| **bluechip** 基线 $k \times w$ 精扫（9 组） | ✅ | ratio-LOSO，16 held-out / 组 |
| **bluechip** 扩展峰值搜索（$k \to 2.5$, $w \to 250$） | 🔄 进行中 | 16 组新 combo；当前跑 `k=1.2, w=208` |
| midcap / solana_fast / base_eco / micro_cap | ⏳ 待 bluechip 定峰后 | 按类独立调参，不跨类池化 |
| 五大类 strict calendar（wf7085） | ⏳ 未开始 | 待各类 $(k,w)$ 冻结后再验证 |

### 1.2 bluechip 当前结论（开发协议 ratio-LOSO）

在 **16 币蓝筹 Meme** 宇宙上，9 组剪枝矩阵的 **开发协议** 最优为：

| 指标 | k=1.2, w=192 | 对比 meme8 同配置（REPORT §1.1） |
| --- | ---: | ---: |
| mean ROC-AUC | **0.7352** ± 0.065 | 0.7434 ± 0.072 |
| mean MCC | **0.3391** ± 0.104 | — |
| mean Macro-F1 | **0.6684** | — |
| test 样本合计 | 191,437 | — |
| held-out 轮数 | 16/16 | 8/8 |

**解读**：

- bluechip 在 **更大、更流动的 16 币宇宙** 上，AUC 与 meme8 冻结配置（0.743）**同量级**，略低约 **−0.8 pp**，说明信号在扩展蓝筹池上仍成立。
- $k \times w$ **协同规律与 meme8 一致**：低 $k$ 加长窗口收益有限；$k \ge 1.0$ 后需 $w \ge 80$；$k=1.2$ 在 $w=192$ 处明显优于 $w=96$（+5.5 pp AUC）。
- **尚未证明** $k=1.2, w=192$ 为全局峰值：扩展扫描（§5）正在向 $k=2.5$、$w=250$ 探测；meme8 上 $k=2.0$ 曾出现高 AUC 但样本骤减，本阶段引入 **可信峰值** 判定（test_n ≥ 80k）。

### 1.3 推荐配置速查（仅 bluechip，开发协议）

| 用途 | k | w | 协议 | mean AUC |
| --- | ---: | ---: | --- | ---: |
| **当前 interim 最优** | 1.2 | 192 | ratio-LOSO | **0.7352** |
| 低开销对照 | 0.5 | 40 | ratio-LOSO | 0.5696 |
| $k/w$ 协同拐点参考 | 1.0 | 96 | ratio-LOSO | 0.6420 |

---

## 2. 实验设计（相对 REPORT.md 的增量）

### 2.1 五大类划分

依据 [advice/ClassificationSuggestion.md](advice/ClassificationSuggestion.md)，每类 **16 个 Binance USDT 现货对**，**类内 LOSO**（15 训 + 1 测），**不跨类池化**。

| category_id | 中文 | 研究焦点 | 配置 |
| --- | --- | --- | --- |
| `bluechip` | 蓝筹 Meme | 市场 Beta / 深流动性 | [config/categories.yaml](config/categories.yaml) |
| `midcap` | 中盘趋势 Meme | 动量 / breakout | 同上 |
| `solana_fast` | Solana 高频 Meme | 高频投机 | 同上 |
| `base_eco` | 新生态 Meme | 叙事扩散（Binance 代理） | 同上 |
| `micro_cap` | 超小盘 Meme | 异常 / 高波动 | 同上 |

输出命名：`{category}16_15m_w{window}_loso_{ratio|wf7085}_{ablation}_*`（legacy 仍为 `meme8_*`）。

### 2.2 两阶段 $k \times w$ 协议（每类独立）

| 阶段 | split | 矩阵 | 脚本 |
| --- | --- | --- | --- |
| **A 基线精扫** | ratio（开发） | 9 组：$k \in \{0.5,0.7,1.0,1.2\}$，$w \in \{40,80,96,192\}$ 剪枝 | `scripts/run_category_kw_scan.sh` |
| **B 扩展峰值** | ratio | 16 组：$k \in \{1.2,1.5,1.8,2.0,2.5\}$，$w \in \{208,224,240,250\}$ 剪枝 | `scripts/run_category_kw_extended.sh` |
| **C 严格验证** | calendar wf7085 | 各类冻结 $(k,w)$ 后单点复跑 | 待 A/B 完成后 |

**可信峰值规则**（`scripts/category_kw_extended_compare.py`）：

- 16 轮 LOSO 完整；
- `test_n ≥ 80,000`（避免 meme8 式 $k=2.0$ 样本崩塌伪高分）；
- 在已测网格上为 **局部极大**（可选参考）。

### 2.3 工程优化（2026-05-28）

扫描阶段启用 `--fast --skip-existing --dataloader-workers 0`：

- `batch_size=1024`，跳过绘图与 predictions CSV；
- 已有 `.npz` / metrics 断点续跑；
- 单组 16 轮 LOSO 耗时由 ~25 min 降至约 **8–12 min**（CPU，无 CUDA）。

---

## 3. bluechip 基线精扫（Phase A，已完成）

**协议**：ratio-LOSO，MLP only，16 held-out / combo  
**日志**：`outputs/metrics/bluechip_kw_scan_ratio.log`  
**汇总**：`outputs/metrics/category_kw_optimal.json`

### 3.1 全矩阵结果

| k | w | 回看 | mean AUC | std | mean MCC | test_n | Δ vs k=0.5,w=40 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 40 | 10h | 0.5696 | 0.016 | 0.0961 | 552,265 | — |
| 0.5 | 80 | 20h | 0.5822 | 0.018 | 0.1148 | 549,934 | +1.3 pp |
| 0.7 | 40 | 10h | 0.5749 | 0.018 | 0.1040 | 417,916 | +0.5 pp |
| 0.7 | 80 | 20h | 0.5855 | 0.021 | 0.1185 | 416,041 | +1.6 pp |
| 1.0 | 40 | 10h | 0.5764 | 0.020 | 0.1045 | 266,036 | +0.7 pp |
| 1.0 | 80 | 20h | 0.6247 | 0.045 | 0.1727 | 264,802 | +5.5 pp |
| 1.0 | 96 | 24h | 0.6420 | 0.068 | 0.1995 | 264,319 | +7.2 pp |
| 1.2 | 96 | 24h | 0.6803 | 0.043 | 0.2509 | 193,160 | +11.1 pp |
| **1.2** | **192** | **48h** | **0.7352** | **0.065** | **0.3391** | **191,437** | **+16.6 pp ★** |

热力图：`outputs/figures/category_kw_heatmap_ratio.png`

### 3.2 与 meme8 历程对照

| 里程碑 | 宇宙 | 配置 | mean AUC | 备注 |
| --- | --- | --- | ---: | --- |
| LOSO baseline | meme8 | k=0.3, w=20 | 0.5664 | REPORT §2.2 |
| 精扫峰值 | meme8 | k=1.2, w=96 | 0.7156 | REPORT §2.6 |
| 冻结配置 | meme8 | k=1.2, w=192 | **0.7434** | REPORT §1.1 开发协议 |
| **本报告** | **bluechip16** | k=1.2, w=192 | **0.7352** | 16 币独立 LOSO |

bluechip 在 $k=1.2, w=192$ 上 **MCC 0.34、Macro-F1 0.67**，分类质量显著高于 $k \le 0.7$ 区域，与 meme8 上「大 $k$ + 长窗口」叙事一致。

### 3.3 最优配置各 held-out（k=1.2, w=192, ratio-LOSO）

**汇总 JSON**：`outputs/metrics/bluechip16_15m_w192_loso_ratio_label_k12_summary.json`

| Held-out | ROC-AUC | MCC | Test 样本 |
| --- | ---: | ---: | ---: |
| SHIBUSDT | **0.8425** | **0.5164** | 13,859 |
| PEOPLEUSDT | 0.8040 | 0.4551 | 13,075 |
| BONKUSDT | 0.8177 | 0.4779 | 14,066 |
| DOGEUSDT | 0.7780 | 0.4055 | 13,338 |
| FLOKIUSDT | 0.7870 | 0.4183 | 14,118 |
| 1MBABYDOGEUSDT | 0.7856 | 0.4147 | 11,937 |
| NEIROUSDT | 0.7682 | 0.3837 | 11,718 |
| MEMEUSDT | 0.7596 | 0.3725 | 14,014 |
| PEPEUSDT | 0.7530 | 0.3684 | 13,771 |
| MUBARAKUSDT | 0.7050 | 0.2818 | 8,013 |
| TRUMPUSDT | 0.6733 | 0.2442 | 8,994 |
| ACTUSDT | 0.6731 | 0.2412 | 8,937 |
| WIFUSDT | 0.6683 | 0.2406 | 13,263 |
| PENGUUSDT | 0.6679 | 0.2285 | 10,143 |
| TURBOUSDT | 0.6503 | 0.1977 | 11,886 |
| PNUTUSDT | 0.6293 | 0.1792 | 10,305 |

**观察**：

- 16/16 held-out AUC > 0.62，MCC 均为正；**SHIB / BONK / PEOPLE** 领先，**PNUT / TURBO / 新币**（TRUMP、PENGU、ACT）偏弱。
- 弱 held-out 多为 **上市时间较短** 或 **与训练池日历重叠较少** 的标的，与 REPORT 中 SHIB 偏弱的模式类似，但 bluechip 池内弱币更多集中在 2024–2025 新上线 meme。

---

## 4. $k \times w$ 协同（bluechip 实证）

与 REPORT §2.5–§2.6 规律 **一致**：

1. **低 $k$ + 长窗口无效**：$k=0.5$ 时 $w=40 \to 80$ 仅 +1.3 pp；$k=1.0$ 时同跨度 +5.5 pp。
2. **高 $k$ 需长上下文**：$k=1.2$ 在 $w=96 \to 192$ 提升 **+5.5 pp**（0.680 → 0.735）。
3. **样本 trade-off 可接受**：$k=1.2$ 时 test_n ≈ 191k（约为 $k=0.5$ 的 35%），仍远高于可信阈值 80k。
4. **std(AUC) 随 peak 升高**：最优组 std=0.065，高于 $k=0.5$ 区域（~0.016），反映 held-out 间方差增大，需在扩展扫描中关注。

---

## 5. 扩展峰值搜索（Phase B，进行中）

**动机**：基线精扫在 $k=1.2, w=192$ 处为矩阵内最高，用户要求向 **$k=2.5$、$w=250$** 延伸以确认峰值可信。

**矩阵**（16 组，跳过已有 1.2:192）：

| k | w（bars） | tag |
| ---: | ---: | --- |
| 1.2 | 208, 224, 240, 250 | label_k12 |
| 1.5 | 208, 224, 240, 250 | label_k15 |
| 1.8 | 224, 240, 250 | label_k18 |
| 2.0 | 224, 240, 250 | label_k20 |
| 2.5 | 240, 250 | label_k25 |

**进度**（2026-05-28 更新）：

| 项目 | 状态 |
| --- | --- |
| 已完成 summary | 0 / 16（扩展组） |
| 当前运行 | `k=1.2, w=208`（`bluechip_kw_extended_ratio.log`） |
| interim 可信峰值 | **k=1.2, w=192**（AUC 0.7352，local peak=True） |

**复现**：

```bash
bash scripts/run_category_kw_extended.sh bluechip
python3 scripts/category_kw_extended_compare.py --category bluechip
```

**预期产物**：

- `outputs/metrics/category_kw_extended_optimal.json`
- `outputs/figures/bluechip_kw_extended_heatmap_ratio.png`

---

## 6. 其他四大类（未开始）

| 类别 | 基线 9 组 | 扩展 16 组 | 备注 |
| --- | ---: | ---: | --- |
| midcap | 0/9 | — | 待 bluechip 定峰后复制流程 |
| solana_fast | 0/9 | — | 先验：可能偏短 $w$ |
| base_eco | 0/9 | — | Binance 代理宇宙，与 Base 链有偏差 |
| micro_cap | 0/9 | — | 先验：可能偏大 $k$ |

**策略**（[advice/规划(动态调整的).txt](advice/规划(动态调整的).txt)）：bluechip 确定可信 $(k,w)$ 后，再为其余四类 **各自** 跑 Phase A →（可选）Phase B → Phase C strict。

---

## 7. 与 REPORT.md 主结论的对齐说明

| 主题 | REPORT.md（meme8） | 本报告（bluechip） |
| --- | --- | --- |
| **严格 calendar 主结论** | AUC ≈ **0.510** | **尚未复测** |
| **开发协议探索** | AUC ≈ **0.743** @ k=1.2,w=192 | AUC ≈ **0.735** @ 同配置、16 币 |
| 泄漏审计 | PASS | 未对新宇宙重跑 audit |
| PnL / τ 扫描 | 开发协议 +43% 等 | 未做 |
| 论文叙事 | 双协议 + 负结果 | 分域超参仍在 **开发协议** 内 |

**重要**：bluechip 0.735 **不能**直接替代 REPORT 的严格外推结论；下一步应对 bluechip 冻结 $(k,w)$ 跑 `--split-mode calendar`，再与 meme8 的 0.510 对照。

---

## 8. 下一步

1. **完成 bluechip 扩展扫描**（16 组）→ 更新 §5 表格与 `category_kw_extended_optimal.json`。
2. **冻结 bluechip $(k,w)$** → 严格 calendar LOSO 16 轮 + leakage audit（可选）。
3. **按类复制**：`run_category_kw_scan.sh` + 视情况 `run_category_kw_extended.sh`。
4. **跨类对比**：`scripts/category_loso_compare.py` → `categories_loso_comparison.json`。
5. **回填 REPORT.md §18** 表格（或保持本报告为分域专用入口）。

---

## 附录 A：产物索引

| 类型 | 路径 |
| --- | --- |
| bluechip 基线汇总 | `outputs/metrics/category_kw_optimal.json` |
| bluechip 扩展汇总 | `outputs/metrics/category_kw_extended_optimal.json` |
| bluechip 基线日志 | `outputs/metrics/bluechip_kw_scan_ratio.log` |
| bluechip 扩展日志 | `outputs/metrics/bluechip_kw_extended_ratio.log` |
| 基线热力图 | `outputs/figures/category_kw_heatmap_ratio.png` |
| 扩展热力图 | `outputs/figures/bluechip_kw_extended_heatmap_ratio.png` |
| 类别币单 | `config/categories.yaml` |

## 附录 B：复现命令

```bash
# bluechip 基线 9 组（已完成）
bash scripts/run_category_kw_scan.sh bluechip

# bluechip 扩展峰值（进行中）
bash scripts/run_category_kw_extended.sh bluechip

# 汇总
python3 scripts/category_kw_compare.py --category bluechip
python3 scripts/category_kw_extended_compare.py --category bluechip

# 严格验证（待冻结 k,w 后）
python3 main.py --mode loso --category bluechip --stage all --model mlp \
  --label-k 1.2 --window-size 192 --split-mode calendar --ablation-tag label_k12
```

---

*数值以 `outputs/metrics/*_summary.json` 为准；本报告随扩展扫描完成持续更新。*
