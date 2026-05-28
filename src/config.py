from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    """单次实验的全部超参数。"""

    # Single-symbol legacy / default symbol (used by single-mode)
    symbol: str = "BTCUSDT"
    # Multi-symbol meme universe used by pooled / LOSO modes
    symbols: tuple[str, ...] = (
        "DOGEUSDT",
        "SHIBUSDT",
        "PEPEUSDT",
        "WIFUSDT",
        "BONKUSDT",
        "FLOKIUSDT",
        "BOMEUSDT",
        "1000SATSUSDT",
    )
    # Research category (bluechip, midcap, ...); None = legacy meme8-style tag
    category_id: str | None = None
    interval: str = "15m"
    lookback_days: int = 720

    window_size: int = 20

    # 所有特征都做过去币种化处理（相对量 / z-score / 比率）。
    feature_columns: tuple[str, ...] = (
        "log_return",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "body_ratio",
        "body_abs_ratio",
        "volume_z_50",
        "volatility_50",
        "atr_14_ratio",
        "ma_10_ratio",
        "ema_10_ratio",
        "rsi_14",
        "macd_hist",
        "bb_position",
        "taker_buy_ratio",
        "taker_buy_ratio_change",
    )

    # 滚动窗口（用于 volume_z 与 volatility）
    volume_z_window: int = 50
    volatility_window: int = 50
    # 标签用 σ 的滚动窗口（可与特征 volatility_50 解耦）
    label_volatility_window: int = 50

    # LOSO 切分：ratio=池化样本序 85/15；calendar=全局日历 walk-forward
    split_mode: str = "ratio"  # "ratio" | "calendar"
    # calendar 模式下训练/验证边界（占全局时间轴分位）
    time_train_fraction: float = 0.70
    time_val_fraction: float = 0.85

    # 标签策略
    label_mode: str = "volatility"  # "fixed" | "volatility"
    label_epsilon: float = 0.0005    # fixed 模式使用
    label_k: float = 0.3             # volatility 模式使用：|r_{t+1}| > k * sigma_t

    # 训练 / 验证 / 测试切分
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 训练超参数
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    early_stopping_patience: int = 8
    random_state: int = 42
    # DataLoader workers (0 = main process only; 4 is good on CPU for large LOSO sets)
    dataloader_workers: int = 4

    # 网络结构
    mlp_hidden: tuple[int, ...] = (256, 128, 64)
    cnn_channels: tuple[int, ...] = (32, 64)
    cnn_kernel_size: int = 3
    cnn_fc_hidden: int = 64

    # 路径
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    # 消融实验：所有输出文件名追加此后缀，方便对比
    ablation_tag: str = ""

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_csv_path_for(self.symbol)

    def raw_csv_path_for(self, symbol: str) -> Path:
        return (
            self.raw_dir
            / f"{symbol.upper()}_{self.interval}_{self.lookback_days}d.csv"
        )

    @property
    def processed_npz_path(self) -> Path:
        return (
            self.processed_dir
            / f"{self.symbol.upper()}_{self.interval}_w{self.window_size}.npz"
        )

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)

    def ensure_dirs(self) -> None:
        for path in [
            self.raw_dir,
            self.processed_dir,
            self.model_dir,
            self.metrics_dir,
            self.figures_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _ablation_suffix(self) -> str:
        return f"_{self.ablation_tag}" if self.ablation_tag else ""

    def run_tag(self, model_name: str, suffix: str | None = None) -> str:
        base = f"{self.symbol.upper()}_{self.interval}_w{self.window_size}_{model_name}"
        if suffix:
            base = f"{base}_{suffix}"
        return f"{base}{self._ablation_suffix()}"

    @property
    def split_tag(self) -> str:
        if self.split_mode == "calendar":
            t = int(self.time_train_fraction * 100)
            v = int(self.time_val_fraction * 100)
            return f"wf{t}{v}"
        return "ratio"

    def universe_prefix(self) -> str:
        """Stable run-id prefix: category16_* or legacy meme{n}_*."""
        n = len(self.symbols)
        if self.category_id:
            return f"{self.category_id.lower()}{n}"
        symbols_signature = "-".join(s[:4] for s in self.symbols)[:40]
        return f"meme{n}_{symbols_signature}"

    def universe_tag(self, mode: str, held_out: str | None = None) -> str:
        prefix = self.universe_prefix()
        split_suffix = f"_{self.split_tag}" if mode == "loso" else ""
        if mode == "loso":
            held_part = f"_{held_out.upper()}" if held_out is not None else ""
            base = (
                f"{prefix}_{self.interval}_w{self.window_size}"
                f"_loso{held_part}{split_suffix}"
            )
        else:
            base = (
                f"{prefix}_{self.interval}_w{self.window_size}_{mode}"
            )
        return f"{base}{self._ablation_suffix()}"


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def config_from_yaml(path: Path | None = None) -> ExperimentConfig:
    """从 YAML 创建配置；未提供路径或缺字段则用默认值。"""
    default_path = PROJECT_ROOT / "config" / "default.yaml"
    raw = load_yaml_config(path or default_path)
    if not raw:
        return ExperimentConfig()

    tuple_fields = {"feature_columns", "mlp_hidden", "cnn_channels", "symbols"}
    # category_id loaded from YAML if present (optional)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in tuple_fields and isinstance(value, list):
            kwargs[key] = tuple(value)
        elif key in {"data_dir", "output_dir"} and isinstance(value, str):
            kwargs[key] = Path(value)
        else:
            kwargs[key] = value
    return ExperimentConfig(**kwargs)
