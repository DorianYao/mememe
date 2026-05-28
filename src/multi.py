"""Multi-symbol pooling + Leave-One-Symbol-Out (LOSO) data orchestration.

LOSO 是这个项目最重要的实验设计：
- 训练: N-1 个 meme 币的池化数据（含时间序内部 val 切分）
- 测试: 第 N 个 meme 币的全部数据（模型从未见过）

这样得到的指标才是「跨币种普适规律」的真正度量。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data import load_ohlcv
from .features import (
    FeatureDataset,
    SplitDataset,
    _scale_inplace,
    build_windows,
    save_processed,
)


PerSymbol = dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def build_per_symbol(
    symbols: list[str],
    config: ExperimentConfig,
    cache: PerSymbol | None = None,
) -> PerSymbol:
    """对每个币种单独建窗口；可选传入缓存避免重复计算。"""
    per_symbol: PerSymbol = dict(cache) if cache else {}
    for sym in symbols:
        key = sym.upper()
        if key in per_symbol:
            continue
        try:
            ohlcv = load_ohlcv(config, symbol=sym)
        except FileNotFoundError as exc:
            print(f"[multi] WARN: missing data for {sym}: {exc}")
            continue
        try:
            x, y, ts, sy = build_windows(ohlcv, config, symbol=key)
        except ValueError as exc:
            print(f"[multi] WARN: cannot build windows for {sym}: {exc}")
            continue
        per_symbol[key] = (x, y, ts, sy)
        positive = float(y.mean()) if len(y) else float("nan")
        print(
            f"[multi] {key:<14} samples={len(y):>6}  "
            f"positive_rate={positive:.4f}"
        )
    if not per_symbol:
        raise RuntimeError("No symbols produced valid samples.")
    return per_symbol


_build_per_symbol = build_per_symbol  # backward-compatible alias


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty(0)
    return np.concatenate(arrays, axis=0)


def _time_sort(
    x: np.ndarray, y: np.ndarray, ts: np.ndarray, syms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按时间戳升序排列（跨币种合并时必要）。"""
    if len(ts) == 0:
        return x, y, ts, syms
    order = np.argsort(ts)
    return x[order], y[order], ts[order], syms[order]


def _ts_max(ts: np.ndarray) -> str:
    if len(ts) == 0:
        return ""
    return str(max(ts.tolist()))


def _ts_min(ts: np.ndarray) -> str:
    if len(ts) == 0:
        return ""
    return str(min(ts.tolist()))


def _mask_by_time(
    ts: np.ndarray,
    *,
    upper: str | None = None,
    lower: str | None = None,
    strict_upper: bool = False,
    strict_lower: bool = False,
) -> np.ndarray:
    """按 ISO 时间戳字符串筛选样本。"""
    if len(ts) == 0:
        return np.zeros(0, dtype=bool)
    arr = np.asarray(ts)
    mask = np.ones(len(arr), dtype=bool)
    if upper is not None:
        mask &= arr < upper if strict_upper else arr <= upper
    if lower is not None:
        mask &= arr > lower if strict_lower else arr >= lower
    return mask


def _apply_mask(
    x: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    syms: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mask.sum() == 0:
        empty_sy = syms[:0] if syms is not None and len(syms) else syms
        return x[:0], y[:0], ts[:0], empty_sy
    return x[mask], y[mask], ts[mask], syms[mask] if syms is not None and len(syms) else syms


def compute_calendar_boundaries(
    per_symbol: PerSymbol,
    symbols: list[str],
    config: ExperimentConfig,
) -> tuple[str, str]:
    """根据全局时间轴分位得到 train_end、val_end（ISO 字符串）。"""
    all_ts: list[str] = []
    for sym in symbols:
        if sym in per_symbol:
            all_ts.extend(per_symbol[sym][2].tolist())
    if not all_ts:
        raise ValueError("No timestamps available for calendar boundaries.")
    sorted_ts = np.sort(np.asarray(all_ts))
    n = len(sorted_ts)
    i_train = max(0, min(n - 1, int(n * config.time_train_fraction) - 1))
    i_val = max(i_train + 1, min(n - 1, int(n * config.time_val_fraction) - 1))
    train_end = str(sorted_ts[i_train])
    val_end = str(sorted_ts[i_val])
    if train_end >= val_end:
        raise ValueError(
            f"Invalid calendar boundaries: train_end={train_end} >= val_end={val_end}. "
            "Adjust time_train_fraction / time_val_fraction."
        )
    return train_end, val_end


def build_loso_splits(
    held_out: str,
    config: ExperimentConfig,
    symbols: list[str] | None = None,
    per_symbol_cache: PerSymbol | None = None,
) -> SplitDataset:
    """LOSO 切分：测试集 = 整个 held_out 币；train+val = 其它币池化后时间切分。

    val 取池化训练集的最后 val_ratio 部分（按时间排序后切尾），保证
    val 在训练集之后、test 币种完全未见。

    `per_symbol_cache` 可复用同一 universe 多轮 LOSO 时已经构造好的特征矩阵。

    split_mode=calendar（walk-forward）：
    - train: 非 held-out 币，timestamp <= train_end
    - val:   非 held-out 币，train_end < timestamp <= val_end
    - test:  held-out 币，timestamp > val_end（严格未来片段）
    """
    universe = [s.upper() for s in (symbols or config.symbols)]
    held_out = held_out.upper()
    if held_out not in universe:
        raise ValueError(
            f"held_out {held_out} not in universe {universe}. Provide --symbols if needed."
        )
    per_symbol = build_per_symbol(universe, config, cache=per_symbol_cache)
    if held_out not in per_symbol:
        raise ValueError(f"held_out symbol {held_out} has no samples")

    train_symbols = [s for s in universe if s != held_out and s in per_symbol]
    if not train_symbols:
        raise ValueError("No training symbols left after holding out.")

    if config.split_mode == "calendar":
        train_end, val_end = compute_calendar_boundaries(per_symbol, universe, config)
        pool_x = _concat([per_symbol[s][0] for s in train_symbols])
        pool_y = _concat([per_symbol[s][1] for s in train_symbols])
        pool_ts = _concat([per_symbol[s][2] for s in train_symbols])
        pool_sy = _concat([per_symbol[s][3] for s in train_symbols])
        pool_x, pool_y, pool_ts, pool_sy = _time_sort(pool_x, pool_y, pool_ts, pool_sy)

        m_train = _mask_by_time(pool_ts, upper=train_end)
        m_val = _mask_by_time(pool_ts, lower=train_end, upper=val_end, strict_lower=True)
        raw_train = _apply_mask(pool_x, pool_y, pool_ts, pool_sy, m_train)
        raw_val = _apply_mask(pool_x, pool_y, pool_ts, pool_sy, m_val)

        tx, ty, tts, tsy = per_symbol[held_out]
        m_test = _mask_by_time(tts, lower=val_end, strict_lower=True)
        test_x, test_y, test_ts, test_sy = _apply_mask(tx, ty, tts, tsy, m_test)

        if len(raw_train[1]) == 0 or len(raw_val[1]) == 0 or len(test_y) == 0:
            raise ValueError(
                f"Calendar LOSO split empty for {held_out}: "
                f"train={len(raw_train[1])} val={len(raw_val[1])} test={len(test_y)} "
                f"(train_end={train_end}, val_end={val_end})"
            )
        if _ts_max(raw_train[2]) >= _ts_min(test_ts):
            raise ValueError(
                "Calendar invariant failed: max(train_ts) >= min(test_ts). "
                f"train_max={_ts_max(raw_train[2])} test_min={_ts_min(test_ts)}"
            )
    else:
        train_x = _concat([per_symbol[s][0] for s in train_symbols])
        train_y = _concat([per_symbol[s][1] for s in train_symbols])
        train_ts = _concat([per_symbol[s][2] for s in train_symbols])
        train_sy = _concat([per_symbol[s][3] for s in train_symbols])
        train_x, train_y, train_ts, train_sy = _time_sort(
            train_x, train_y, train_ts, train_sy
        )

        n = len(train_y)
        val_size = int(n * config.val_ratio)
        if val_size <= 0:
            raise ValueError(f"val_ratio too small: only {n} pooled training samples.")
        cutoff = n - val_size

        raw_train = (
            train_x[:cutoff],
            train_y[:cutoff],
            train_ts[:cutoff],
            train_sy[:cutoff],
        )
        raw_val = (
            train_x[cutoff:],
            train_y[cutoff:],
            train_ts[cutoff:],
            train_sy[cutoff:],
        )

        test_x, test_y, test_ts, test_sy = per_symbol[held_out]

    scaler = StandardScaler()
    scaler.fit(raw_train[0].reshape(-1, raw_train[0].shape[-1]))

    return SplitDataset(
        train=FeatureDataset(
            _scale_inplace(scaler, raw_train[0]), raw_train[1], raw_train[2], raw_train[3]
        ),
        val=FeatureDataset(
            _scale_inplace(scaler, raw_val[0]), raw_val[1], raw_val[2], raw_val[3]
        ),
        test=FeatureDataset(
            _scale_inplace(scaler, test_x), test_y, test_ts, test_sy
        ),
        scaler=scaler,
        feature_columns=list(config.feature_columns),
        universe=universe,
        held_out=held_out,
    )


def build_pooled_splits(
    config: ExperimentConfig,
    symbols: list[str] | None = None,
    per_symbol_cache: PerSymbol | None = None,
) -> SplitDataset:
    """池化模式：所有币种数据合并后，整体按时间切 train/val/test。"""
    universe = [s.upper() for s in (symbols or config.symbols)]
    per_symbol = build_per_symbol(universe, config, cache=per_symbol_cache)

    x_all = _concat([per_symbol[s][0] for s in per_symbol])
    y_all = _concat([per_symbol[s][1] for s in per_symbol])
    ts_all = _concat([per_symbol[s][2] for s in per_symbol])
    sy_all = _concat([per_symbol[s][3] for s in per_symbol])
    x_all, y_all, ts_all, sy_all = _time_sort(x_all, y_all, ts_all, sy_all)

    n = len(y_all)
    test_size = int(n * config.test_ratio)
    val_size = int(n * config.val_ratio)
    train_size = n - val_size - test_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(f"Not enough pooled samples ({n}) for split ratios.")
    bounds = [0, train_size, train_size + val_size, n]

    scaler = StandardScaler()
    scaler.fit(x_all[: bounds[1]].reshape(-1, x_all.shape[-1]))

    parts: list[FeatureDataset] = []
    for i in range(3):
        s, e = bounds[i], bounds[i + 1]
        parts.append(
            FeatureDataset(
                _scale_inplace(scaler, x_all[s:e]),
                y_all[s:e],
                ts_all[s:e],
                sy_all[s:e],
            )
        )

    return SplitDataset(
        train=parts[0],
        val=parts[1],
        test=parts[2],
        scaler=scaler,
        feature_columns=list(config.feature_columns),
        universe=list(per_symbol.keys()),
        held_out=None,
    )


def loso_processed_path(config: ExperimentConfig, held_out: str) -> Path:
    return config.processed_dir / f"{config.universe_tag('loso', held_out)}.npz"


def pooled_processed_path(config: ExperimentConfig) -> Path:
    return config.processed_dir / f"{config.universe_tag('pooled')}.npz"


def build_and_save_loso(
    config: ExperimentConfig,
    held_out: str,
    symbols: list[str] | None = None,
    per_symbol_cache: PerSymbol | None = None,
) -> tuple[SplitDataset, Path]:
    config.ensure_dirs()
    splits = build_loso_splits(
        held_out, config, symbols, per_symbol_cache=per_symbol_cache
    )
    path = loso_processed_path(config, held_out)
    save_processed(splits, path)
    return splits, path


def build_and_save_pooled(
    config: ExperimentConfig,
    symbols: list[str] | None = None,
    per_symbol_cache: PerSymbol | None = None,
) -> tuple[SplitDataset, Path]:
    config.ensure_dirs()
    splits = build_pooled_splits(config, symbols, per_symbol_cache=per_symbol_cache)
    path = pooled_processed_path(config)
    save_processed(splits, path)
    return splits, path
