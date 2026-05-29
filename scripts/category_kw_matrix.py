"""Single source of truth for five-category k×w scan grids.

Phase A (base): k∈{1.0,1.2}, w∈{96,128,192} — drop low-k/low-w after bluechip Phase A.
Phase B (extended): k∈{1.0,1.2,1.5,1.8,2.0,2.5}, w up to 288 — expand high-w peak search.

Usage:
  python3 scripts/category_kw_matrix.py base
  python3 scripts/category_kw_matrix.py extended
"""

from __future__ import annotations

import sys

# Phase A — 5 pruned combos (was 9: k≥1.0, w≥96)
KW_MATRIX: list[tuple[float, int, str]] = [
    (1.0, 96, "label_k10"),
    (1.0, 128, "label_k10"),
    (1.2, 96, "label_k12"),
    (1.2, 128, "label_k12"),
    (1.2, 192, "label_k12"),
]

# Phase B — 22 pruned combos (k=1.0–2.5, w=192–288; skip duplicates with Phase A)
EXTENDED_KW_MATRIX: list[tuple[float, int, str]] = [
    (1.0, 192, "label_k10"),
    (1.2, 208, "label_k12"),
    (1.2, 224, "label_k12"),
    (1.2, 240, "label_k12"),
    (1.2, 256, "label_k12"),
    (1.2, 272, "label_k12"),
    (1.5, 208, "label_k15"),
    (1.5, 224, "label_k15"),
    (1.5, 240, "label_k15"),
    (1.5, 256, "label_k15"),
    (1.8, 224, "label_k18"),
    (1.8, 240, "label_k18"),
    (1.8, 256, "label_k18"),
    (1.8, 272, "label_k18"),
    (2.0, 224, "label_k20"),
    (2.0, 240, "label_k20"),
    (2.0, 256, "label_k20"),
    (2.0, 272, "label_k20"),
    (2.5, 240, "label_k25"),
    (2.5, 256, "label_k25"),
    (2.5, 272, "label_k25"),
    (2.5, 288, "label_k25"),
]

K_TO_TAG: dict[float, str] = {
    1.0: "label_k10",
    1.2: "label_k12",
    1.5: "label_k15",
    1.8: "label_k18",
    2.0: "label_k20",
    2.5: "label_k25",
}


def combo_lines(matrix: list[tuple[float, int, str]]) -> list[str]:
    return [f"{k}:{w}:{tag}" for k, w, tag in matrix]


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"base", "extended"}:
        print("Usage: category_kw_matrix.py base|extended", file=sys.stderr)
        raise SystemExit(1)
    matrix = KW_MATRIX if sys.argv[1] == "base" else EXTENDED_KW_MATRIX
    print("\n".join(combo_lines(matrix)))


if __name__ == "__main__":
    main()
