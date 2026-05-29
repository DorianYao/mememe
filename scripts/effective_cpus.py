#!/usr/bin/env python3
"""Print cgroup-aware CPU count for worker sizing in shell scripts."""
from __future__ import annotations

import os


def effective_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, NotImplementedError, OSError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except (OSError, ValueError, IndexError):
        pass
    return max(1, (os.cpu_count() or 4))


if __name__ == "__main__":
    print(effective_cpus())
