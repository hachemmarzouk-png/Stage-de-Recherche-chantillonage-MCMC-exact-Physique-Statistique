"""Quick checks of the original research code; run with python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sos_cftp import apply_step, make_max_mixed, make_min_mixed, propp_wilson_mixed, total_energy


class TestSOS(unittest.TestCase):
    def test_extreme_states_and_fixed_boundaries(self):
        n, k = 2, 3
        lo, hi = make_min_mixed(n, k), make_max_mixed(n, k)
        self.assertEqual(lo.shape, (5, 5))
        self.assertTrue(np.all(lo <= hi))
        for grid in (lo, hi):
            self.assertTrue(np.all(grid[0] == -n))
            self.assertTrue(np.all(grid[-1] == n))
            self.assertTrue(np.all(np.abs(grid) <= k))

    def test_energy_local_move(self):
        grid = make_min_mixed(1, 2)
        before = total_energy(grid)
        apply_step(grid, 1, 1, +1, 0., K=2, beta=0.)
        self.assertEqual(grid[1, 1], -1)
        self.assertEqual(total_energy(grid) - before, 0)

    def test_small_cftp_reproducibility(self):
        sample1, t1, _ = propp_wilson_mixed(N=1, beta=0.5, seed=0, heartbeat_seconds=1000.)
        sample2, t2, _ = propp_wilson_mixed(N=1, beta=0.5, seed=0, heartbeat_seconds=1000.)
        np.testing.assert_array_equal(sample1, sample2)
        self.assertEqual(t1, t2)
        self.assertEqual(t1, 48)
        self.assertTrue(np.all(sample1[0] == -1))
        self.assertTrue(np.all(sample1[-1] == +1))


if __name__ == "__main__":
    unittest.main()
