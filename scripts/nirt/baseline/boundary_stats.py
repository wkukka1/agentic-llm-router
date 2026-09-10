"""Report the graded-score boundary statistics (y==0 / y==1 / interior).

    python scripts/boundary_stats.py

Writes artifacts/phase2/response_boundary_statistics.json. Run before ZOIB.
"""

from __future__ import annotations

import json
import sys

from router.config import load_config
from router.nirt.baseline.continuous_eval import boundary_statistics


def main() -> int:
    stats = boundary_statistics(load_config())
    print(json.dumps(stats["by_split"], indent=2))
    print("\nper test model (y==0 / y==1 / interior):")
    for m, f in stats["by_model"].items():
        print(f"  {m:26s} {f['y==0']:.3f} / {f['y==1']:.3f} / {f['0<y<1']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
