from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


@dataclass
class FeatureDataset:
    """单个数据子集（train / val / test）。

    x: (samples, window_size, num_features)
    y: (samples,) binary
    timestamps: (samples,)
    symbols: optional (samples,) - 用于按币种聚合指标
    """

    x: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    symbols: np.ndarray | None = None


@dataclass
class SplitDataset:
    train: FeatureDataset
    val: FeatureDataset
    test: FeatureDataset
    scaler: StandardScaler
    feature_columns: list[str]
    universe: list[str] = field(default_factory=list)
    held_out: str | None = None


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """生成完全去币种化的特征。

    所有特征都是「相对量 / 比率 / z-score / 归一化指标」，跨币种可直接拼接。
    """
    out = df.copy()
    open_ = out["open"]
    high = out["high"]
    low = out["low"]
    close = out["close"]
    volume = out["volume"]

    # ---- 价格 / K 线形态（全部相对化） ----
    out["log_return"] = np.log(close / close.shift(1))

    upper_body = np.maximum(open_, close)
    lower_body = np.minimum(open_, close)
    out["upper_wick_ratio"] = (high - upper_body) / close
    out["lower_wick_ratio"] = (lower_body - low) / close
    out["body_ratio"] = (close - open_) / open_
    out["body_abs_ratio"] = (close - open_).abs() / close

    # ---- 成交量：rolling z-score（关键：跨币种可比） ----
    vw = max(config.volume_z_window, 5)
    safe_volume = volume.replace(0, np.nan)
    log_volume = np.log1p(safe_volume)
    vol_mean = log_volume.rolling(vw).mean()
    vol_std = log_volume.rolling(vw).std().replace(0, np.nan)
    out["volume_z_50"] = (log_volume - vol_mean) / vol_std

    # 备用：原始 log volume（量级在 BTC vs PEPE 差很多，专门留给消融实验）
    out["volume_log"] = log_volume
    out["volume_change"] = volume.pct_change()

    # ---- 波动率（特征用窗口 vs 标签用窗口可解耦） ----
    rw = max(config.volatility_window, 5)
    out["volatility_50"] = out["log_return"].rolling(rw).std()
    lw = max(config.label_volatility_window, 5)
    out["volatility_label"] = out["log_return"].rolling(lw).std()

    # ATR / close 是天然币种无关的
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14_ratio"] = true_range.rolling(14).mean() / close

    # ---- 趋势 ----
    ma_10 = close.rolling(10).mean()
    ema_10 = close.ewm(span=10, adjust=False).mean()
    out["ma_10_ratio"] = close / ma_10 - 1
    out["ema_10_ratio"] = close / ema_10 - 1

    # ---- 动量 ----
    out["rsi_14"] = compute_rsi(close, period=14) / 100.0

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - macd_signal) / close

    # ---- 波动通道 ----
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_position"] = (close - bb_lower) / bb_range

    # ---- 买卖强度（taker buy 比率） ----
    if "taker_buy_base" in out.columns:
        out["taker_buy_ratio"] = out["taker_buy_base"] / safe_volume
        out["taker_buy_ratio_change"] = out["taker_buy_ratio"].diff()
    else:
        out["taker_buy_ratio"] = 0.5
        out["taker_buy_ratio_change"] = 0.0

    return out.replace([np.inf, -np.inf], np.nan)


def _decide_label(
    future_return: float,
    sigma: float,
    config: ExperimentConfig,
) -> int | None:
    """返回 1 / 0 / None（None 表示要丢弃的模糊样本）。"""
    if not np.isfinite(future_return):
        return None
    if config.label_mode == "volatility":
        if not np.isfinite(sigma) or sigma <= 0:
            return None
        threshold = config.label_k * sigma
        if future_return > threshold:
            return 1
        if future_return < -threshold:
            return 0
        return None
    # fixed mode
    if abs(future_return) < config.label_epsilon:
        return None
    return 1 if future_return > 0 else 0


def build_windows(
    ohlcv: pd.DataFrame,
    config: ExperimentConfig,
    symbol: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成 (samples, window, features) 滑窗 + 二分类标签。

    标签使用动态波动率阈值（默认）或固定 epsilon。
    返回: x, y, timestamps, symbols 四个数组（长度一致）。
    """
    df = add_indicators(ohlcv, config)
    feature_cols = list(config.feature_columns)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns after indicator computation: {missing}"
        )
    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].astype(str).to_numpy()
    if config.label_mode == "volatility":
        sigma = df["volatility_label"].to_numpy(dtype=np.float32)
    else:
        sigma = df["volatility_50"].to_numpy(dtype=np.float32)

    window = config.window_size
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    ts_list: list[str] = []
    dropped_ambiguous = 0
    skipped_nan = 0

    for end_idx in range(window - 1, len(df) - 1):
        start_idx = end_idx - window + 1
        window_features = feature_matrix[start_idx : end_idx + 1]
        if np.isnan(window_features).any():
            skipped_nan += 1
            continue
        cur_close = close[end_idx]
        next_close = close[end_idx + 1]
        if not np.isfinite(cur_close) or cur_close <= 0 or not np.isfinite(next_close):
            skipped_nan += 1
            continue
        future_return = float(np.log(next_close / cur_close))
        label = _decide_label(future_return, float(sigma[end_idx]), config)
        if label is None:
            dropped_ambiguous += 1
            continue
        x_list.append(window_features)
        y_list.append(label)
        ts_list.append(timestamps[end_idx])

    if not x_list:
        raise ValueError(
            f"No valid samples for {symbol or 'data'}. "
            "Check data length, indicators, or label settings."
        )

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    ts = np.asarray(ts_list)
    syms = np.full(len(y), symbol or "", dtype=object) if symbol else np.empty(0)

    return x, y, ts, syms if symbol else np.array([], dtype=object)


def _scale_inplace(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    flat = x.reshape(-1, x.shape[-1])
    return scaler.transform(flat).reshape(x.shape).astype(np.float32)


def time_split(
    x: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    syms: np.ndarray,
    config: ExperimentConfig,
) -> SplitDataset:
    """单币时间序顺切分。test_ratio + val_ratio < 1。"""
    n = len(y)
    test_size = int(n * config.test_ratio)
    val_size = int(n * config.val_ratio)
    train_size = n - val_size - test_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(f"Not enough samples ({n}) for split ratios.")

    bounds = [0, train_size, train_size + val_size, n]
    sub_x = [x[bounds[i] : bounds[i + 1]] for i in range(3)]
    sub_y = [y[bounds[i] : bounds[i + 1]] for i in range(3)]
    sub_ts = [ts[bounds[i] : bounds[i + 1]] for i in range(3)]
    sub_syms = (
        [syms[bounds[i] : bounds[i + 1]] for i in range(3)]
        if syms.size
        else [None, None, None]
    )

    scaler = StandardScaler()
    scaler.fit(sub_x[0].reshape(-1, sub_x[0].shape[-1]))
    scaled = [_scale_inplace(scaler, arr) for arr in sub_x]

    return SplitDataset(
        train=FeatureDataset(scaled[0], sub_y[0], sub_ts[0], sub_syms[0]),
        val=FeatureDataset(scaled[1], sub_y[1], sub_ts[1], sub_syms[1]),
        test=FeatureDataset(scaled[2], sub_y[2], sub_ts[2], sub_syms[2]),
        scaler=scaler,
        feature_columns=list(config.feature_columns),
    )


def build_and_save(
    ohlcv: pd.DataFrame, config: ExperimentConfig
) -> tuple[SplitDataset, Path]:
    """Single-symbol convenience: build windows for `config.symbol` and save NPZ."""
    config.ensure_dirs()
    x, y, ts, syms = build_windows(ohlcv, config, symbol=config.symbol)
    splits = time_split(x, y, ts, syms, config)
    save_processed(splits, config.processed_npz_path)
    return splits, config.processed_npz_path


def save_processed(splits: SplitDataset, path: Path) -> Path:
    payload = {
        "x_train": splits.train.x,
        "y_train": splits.train.y,
        "ts_train": splits.train.timestamps,
        "x_val": splits.val.x,
        "y_val": splits.val.y,
        "ts_val": splits.val.timestamps,
        "x_test": splits.test.x,
        "y_test": splits.test.y,
        "ts_test": splits.test.timestamps,
        "feature_columns": np.array(splits.feature_columns),
        "scaler_mean": splits.scaler.mean_,
        "scaler_scale": splits.scaler.scale_,
        "universe": np.array(splits.universe),
        "held_out": np.array(splits.held_out or ""),
    }
    if splits.train.symbols is not None:
        payload["sym_train"] = splits.train.symbols
    if splits.val.symbols is not None:
        payload["sym_val"] = splits.val.symbols
    if splits.test.symbols is not None:
        payload["sym_test"] = splits.test.symbols
    np.savez_compressed(path, **payload)
    return path


def load_processed(path: Path) -> SplitDataset:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}.")
    with np.load(path, allow_pickle=True) as data:
        scaler = StandardScaler()
        scaler.mean_ = data["scaler_mean"].astype(np.float64)
        scaler.scale_ = data["scaler_scale"].astype(np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)

        def _syms(key: str) -> np.ndarray | None:
            return data[key] if key in data.files else None

        held = str(data["held_out"]) if "held_out" in data.files else ""
        return SplitDataset(
            train=FeatureDataset(
                data["x_train"].astype(np.float32),
                data["y_train"].astype(np.float32),
                data["ts_train"],
                _syms("sym_train"),
            ),
            val=FeatureDataset(
                data["x_val"].astype(np.float32),
                data["y_val"].astype(np.float32),
                data["ts_val"],
                _syms("sym_val"),
            ),
            test=FeatureDataset(
                data["x_test"].astype(np.float32),
                data["y_test"].astype(np.float32),
                data["ts_test"],
                _syms("sym_test"),
            ),
            scaler=scaler,
            feature_columns=list(data["feature_columns"]),
            universe=list(data["universe"]) if "universe" in data.files else [],
            held_out=held or None,
        )
