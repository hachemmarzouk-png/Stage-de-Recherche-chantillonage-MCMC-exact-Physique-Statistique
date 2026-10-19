"""Small reproducible CLI around the original SOS research implementation.

Example: python scripts/run_cftp.py --N 2 --beta 1.0 --seed 0
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sos_cftp as sos  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Exact CFTP sampling for truncated mixed-boundary SOS")
    parser.add_argument("--N", type=int, default=2, help="Lattice side is 2*N+1 (default 2)")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["local", "line"], default="local",
                        help="Local Metropolis CFTP or experimental line-bombardment dynamics")
    parser.add_argument("--k", type=int, default=20, help="Extra moves per bombardment in line mode")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs_demo")
    parser.add_argument("--plot", action="store_true", help="Save the visualisations from the original script")
    args = parser.parse_args()
    if args.N < 1:
        parser.error("N must be >= 1")
    if args.beta < 0:
        parser.error("beta must be non-negative")

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "local":
        grid, horizon, elapsed = sos.propp_wilson_mixed(
            N=args.N, beta=args.beta, seed=args.seed)
        summary = {"mode": args.mode, "horizon_local_steps": horizon}
    else:
        grid, horizon, n_bomb, n_proposals, elapsed = sos.propp_wilson_mixed_with_line_bombardments(
            N=args.N, beta=args.beta, seed=args.seed, k=args.k)
        summary = {"mode": args.mode, "horizon_ordinary_steps": horizon,
                   "line_bombardments": n_bomb, "local_proposals": n_proposals, "k": args.k}

    summary.update({"N": args.N, "L": 2 * args.N + 1, "beta": args.beta,
                    "seed": args.seed, "time_minutes": elapsed,
                    "min_height": int(grid.min()), "max_height": int(grid.max()),
                    "energy": sos.total_energy(grid)})
    np.save(args.outdir / "sample.npy", grid)
    (args.outdir / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        sos.OUTPUT_DIR = args.outdir
        sos.plot_outputs(grid, horizon, args.N, args.beta, show=False)
    print("Saved sample.npy and run.json in", args.outdir.resolve())


if __name__ == "__main__":
    main()
