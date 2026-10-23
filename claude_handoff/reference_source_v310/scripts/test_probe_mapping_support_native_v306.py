"""Pure cost-estimator and admission schema tests; no remote/GPU invocation."""
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
from scripts import probe_mapping_support_native_v306 as p


class ProbeTests(unittest.TestCase):
    def rows(self):
        return [dict(arm=a,size=h,agents=n,measured_seconds=[h/128.,h/128.])
            for a in ('support_local','cross_address') for h in (128,256,512) for n in (8,16)]

    def test_all_cohorts_equally_weighted(self):
        r=p.estimate(self.rows(),2160,900)
        self.assertEqual(r['gradient_training_seconds']['support_local'],5040)
        self.assertEqual(r['gradient_training_seconds']['cross_address'],5040)
        self.assertEqual(r['estimated_pair_seconds'],16020)
        self.assertTrue(r['estimate_not_measured_training_duration'])

    def test_missing_duplicate_nonfinite_forbidden(self):
        with self.assertRaises(ValueError):p.estimate(self.rows()[:-1],2160,900)
        rows=self.rows();rows[-1]=copy.deepcopy(rows[-2])
        with self.assertRaises(ValueError):p.estimate(rows,2160,900)
        rows=self.rows();rows[0]['measured_seconds']=[float('nan')]
        with self.assertRaises(ValueError):p.estimate(rows,2160,900)
        for count in (0,-1,True):
            with self.assertRaises(ValueError):p.estimate(self.rows(),count,900)


if __name__=='__main__':unittest.main()
