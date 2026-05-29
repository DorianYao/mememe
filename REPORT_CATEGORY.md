# 五大类分域实验报告（承接 REPORT.md）

**项目**：15 分钟 K 线下一根涨跌二分类（MLP）  
**前置报告**：[REPORT.md](REPORT.md)（legacy **meme8** × 8 币、双协议主结论、严格 calendar AUC ≈ 0.51）  
**本报告范围**：2026-05-28 起的 **五大研究类 × 16 币** 独立 LOSO、类内 $k \times w$ 调参  
**报告更新**：2026-05-29  

> 完整 JSON / 日志见 `outputs/metrics/`。  
> **与 REPORT.md 的关系**：§1–§18 的 meme8 主结论（尤其严格协议 ≈ 随机）**不变**；本报告记录 **分大类后的开发协议超参探索**，尚未对任何类别跑严格 calendar 验证。

---

## 1. 摘要

### 1.1 本阶段做了什么

| 里程碑 | 状态 | 说明 |
| --- | --- | --- |
| 41 币 OHLCV 下载（五大类并集） | ✅ | `scripts/download_all_category_data.py` |
| **五类 Phase A** 基线 $k \times w$ 精扫（各 5 组） | ✅ | ratio-LOSO，16 held-out / 组 |
| **base_eco Phase B** 扩展峰值搜索（$k \to 2.5$, $w \to 288$） | 🔄 **当前主线** | 26 combo；远程 Pod 执行 |
| bluechip / midcap / solana_fast / micro_cap Phase B | ⏸ 暂停 | Phase A 已定各类 interim 最优，扩展扫留待 base_eco 定峰后 |
| 五大类 strict calendar（wf7085） | ⏳ 未开始 | 待 base_eco $(k,w)$ 冻结后再验证 |

**策略调整（2026-05-29）**：五类 Phase A 完成后，**base_eco 在开发协议 AUC 最高（0.715）**，故收敛算力至 **仅训练 base_eco**，优先完成 $k \in [1.2, 2.5]$ 扩展峰值搜索。

### 1.2 五类 Phase A 横向对比（开发协议 ratio-LOSO）

| 类别 | 中文 | interim 最优 $(k,w)$ | mean AUC | mean MCC | mean F1 | test_n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **base_eco** | 新生态 Meme | **1.2 / 192** | **0.7151** | **0.3081** | **0.6535** | 202,152 |
| bluechip | 蓝筹 Meme | 1.2 / 128 | 0.6744 | 0.2473 | 0.6227 | 192,576 |
| solana_fast | Solana 高频 Meme | 1.2 / 192 | 0.6434 | 0.2023 | 0.5988 | 180,104 |
| midcap | 中盘趋势 Meme | 1.2 / 96 | 0.6378 | 0.1921 | 0.5956 | 180,878 |
| micro_cap | 超小盘 Meme | 1.2 / 96 | 0.6332 | 0.1860 | 0.5921 | 178,801 |

汇总 JSON：`outputs/metrics/category_kw_optimal.json`

**解读**：

- **base_eco 显著领先**（比第二名 bluechip 高 **+4.1 pp AUC**），且 MCC / F1 同步最高。
- 五类 interim 最优均为 **k=1.2**，但最优窗口因类而异：base_eco / solana_fast 偏长（192），midcap / micro_cap 偏短（96），bluechip 居中（128）。
- base_eco 在 $w=128 \to 192$ 处出现 **+7.6 pp** 跃升（0.640 → 0.715），$k/w$ 协同效应最强，值得向更高 $k$、更长 $w$ 延伸探测。

### 1.3 推荐配置速查（base_eco，开发协议）

| 用途 | k | w | 协议 | mean AUC |
| --- | ---: | ---: | --- | ---: |
| **当前 interim 最优** | 1.2 | 192 | ratio-LOSO | **0.7151** |
| 短窗口对照 | 1.2 | 128 | ratio-LOSO | 0.6395 |
| 低 $k$ 对照 | 1.0 | 128 | ratio-LOSO | 0.6056 |

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
| **C 严格验证** | calendar wf7085 | 冻结 $(k,w)$ 后单点复跑 | 待 B 完成后 |

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
**日志**：`outputs/metrics/base_eco_kw_scan_ratio.log`  
**汇总**：`outputs/metrics/category_kw_optimal.json` → `base_eco` 段

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

**汇总 JSON**：`outputs/metrics/base_eco16_15m_w192_loso_ratio_label_k12_summary.json`

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

## 4. 扩展峰值搜索（Phase B，base_eco，进行中）

**动机**：Phase A 在 $k=1.2, w=192$ 处为矩阵内最高；需向 **$k=2.5$、$w=288$** 延伸以确认峰值可信，并补充 **$k \in [1.2, 2.5]$** 全段测试。

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
| Phase B（26 组） | 🔄 远程执行中 |
| interim 可信峰值 | **k=1.2, w=192**（AUC 0.7151，local peak=True） |

**复现**（远程 Pod）：

```bash
bash scripts/run_category_kw_extended.sh base_eco
python3 scripts/category_kw_extended_compare.py --category base_eco
python3 scripts/category_kw_extended_compare.py --category base_eco --plot
```

**预期产物**：

- `outputs/metrics/category_kw_extended_optimal.json`
- `outputs/metrics/base_eco_kw_extended_ratio.log`
- `outputs/figures/base_eco_kw_extended_heatmap_ratio.png`

---

## 5. 其他四类（Phase A 完成，Phase B 暂停）

| 类别 | Phase A 最优 | AUC | Phase B |
| --- | ---: | ---: | --- |
| bluechip | k=1.2, w=128 | 0.6744 | ⏸ 本地已终止，待 base_eco 定峰 |
| solana_fast | k=1.2, w=192 | 0.6434 | ⏸ |
| midcap | k=1.2, w=96 | 0.6378 | ⏸ |
| micro_cap | k=1.2, w=96 | 0.6332 | ⏸ |

各类 Phase A 日志：`outputs/metrics/{category}_kw_scan_ratio.log`

---

## 6. 与 REPORT.md 主结论的对齐说明

| 主题 | REPORT.md（meme8） | 本报告（base_eco） |
| --- | --- | --- |
| **严格 calendar 主结论** | AUC ≈ **0.510** | **尚未复测** |
| **开发协议探索** | AUC ≈ **0.743** @ k=1.2,w=192 | AUC ≈ **0.715** @ 同配置、16 币 L2 代理 |
| 泄漏审计 | PASS | 未对新宇宙重跑 audit |
| PnL / τ 扫描 | 开发协议 +43% 等 | 未做 |

**重要**：base_eco 0.715 **不能**直接替代 REPORT 的严格外推结论；下一步应对 base_eco 冻结 $(k,w)$ 跑 `--split-mode calendar`，再与 meme8 的 0.510 对照。

---

## 7. 下一步

1. **完成 base_eco Phase B 扩展扫描**（26 组）→ 更新 §4 表格与 `category_kw_extended_optimal.json`。
2. **冻结 base_eco $(k,w)$** → 严格 calendar LOSO 16 轮 + leakage audit（可选）。
3. **视 base_eco 定峰结果**，决定是否对其余四类复制 Phase B。
4. **跨类对比**：`scripts/category_loso_compare.py` → `categories_loso_comparison.json`。

---

## 附录 A：产物索引

| 类型 | 路径 |
| --- | --- |
| 五类 Phase A 汇总 | `outputs/metrics/category_kw_optimal.json` |
| base_eco 扩展汇总 | `outputs/metrics/category_kw_extended_optimal.json` |
| base_eco 基线日志 | `outputs/metrics/base_eco_kw_scan_ratio.log` |
| base_eco 扩展日志 | `outputs/metrics/base_eco_kw_extended_ratio.log` |
| 基线热力图 | `outputs/figures/category_kw_heatmap_ratio.png` |
| 扩展热力图 | `outputs/figures/base_eco_kw_extended_heatmap_ratio.png` |
| 类别币单 | `config/categories.yaml` |
| 远程操作手册 | [docs/remote_workflow.md](docs/remote_workflow.md) |

## 附录 B：远程复现命令（base_eco Phase B）

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

严格验证（待冻结 k,w 后）：

```bash
python3 main.py --mode loso --category base_eco --stage all --model mlp \
  --label-k 1.2 --window-size 192 --split-mode calendar --ablation-tag label_k12
```

---

*数值以 `outputs/metrics/*_summary.json` 为准；本报告随 base_eco 扩展扫描完成持续更新。*
