"""Tests for sampling invariants independent of model downloads."""
import unittest
import numpy as np
from experiment_support import sample_splits

class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.labels=np.repeat(np.arange(10),30)
        self.eligible=np.arange(300)
    def test_balanced_disjoint_repeatable(self):
        a,b=sample_splits(self.labels,self.eligible,4,6,2026)
        c,d=sample_splits(self.labels,self.eligible,4,6,2026)
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)
        np.testing.assert_array_equal(np.bincount(self.labels[a]),np.full(10,4))
        np.testing.assert_array_equal(np.bincount(self.labels[b]),np.full(10,6))
        self.assertFalse(set(a)&set(b))
    def test_excluded_rows_never_sampled(self):
        excluded=np.arange(0,300,3)
        allowed=np.setdiff1d(self.eligible,excluded)
        a,b=sample_splits(self.labels,allowed,4,6,2026)
        self.assertFalse((set(a)|set(b))&set(excluded))
    def test_insufficient_class_fails(self):
        with self.assertRaises(ValueError):
            sample_splits(self.labels,self.eligible[self.labels!=9],4,6,2026)

if __name__=='__main__':unittest.main()
