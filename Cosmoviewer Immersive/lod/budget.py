"""
How many particles are drawn, and how many are kept per snapshot in memory.

DEFAULT_BUDGET (X) is the number of particles on screen for the current
snapshot. 2,000,000 is GadgetViewer's default display sample and, measured
on this code (PERFORMANCE.md), keeps a full refresh (selection, colours,
upload to the GPU) to a fraction of a second on an M-series MacBook while
giving more than one particle per screen pixel of a 1600x1000 view, so the
view does not look sparse at any zoom. Set it with --max-particles or the
View > Particle budget menu.

Snapshots other than the current one (playback) keep a smaller base sample so
that loading hundreds of snapshots stays within BASE_MEMORY_BYTES.
"""

DEFAULT_BUDGET = 2_000_000
BUDGET_CHOICES = (500_000, 1_000_000, 2_000_000, 5_000_000)
BASE_MEMORY_BYTES = 1_500_000_000      # all base samples together
BYTES_PER_PARTICLE = 41                # c, v (float32 x3), m, t, id, row
MIN_BASE_SAMPLE = 50_000


def base_sample_size(budget: int, n_snapshots: int) -> int:
    """Per-snapshot base sample: the budget, unless that would exceed the memory allowance."""
    per_snap = BASE_MEMORY_BYTES // (BYTES_PER_PARTICLE * max(n_snapshots, 1))
    return int(max(MIN_BASE_SAMPLE, min(budget, per_snap)))


def background_workers() -> int:
    """Half the cores, at most 4: the rest of the laptop (and the window) stays responsive."""
    import os
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def lower_priority():
    """Run a worker process below the window and the rest of the system."""
    import os
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
