"""Numerical tests for Experiment 0.5 morphology, controls and signed BGS."""
import importlib.util
import io
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

spec=importlib.util.spec_from_file_location("specificity",Path(__file__).with_name("boundary_specificity_audit.py"))
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SpecificityAuditTests(unittest.TestCase):
    def cube_regions(self):
        label=torch.zeros(1,1,17,17,17)
        label[:,:,4:13,4:13,4:13]=1
        return audit.make_regions(label,label.shape[2:])

    def test_near_far_core_ring_counts_and_disjointness(self):
        regions=self.cube_regions()
        expected=dict(near_in=9**3-7**3,near_out=11**3-9**3,
                      far_in=5**3-3**3,far_out=15**3-13**3,
                      foreground_core=3**3,background_core=17**3-15**3)
        for name,count in expected.items():
            self.assertEqual(int(regions[name].sum()),count,name)
        self.assertFalse(bool((regions["boundary"] & regions["nonboundary"]).any()))
        self.assertFalse(bool((audit.morphology(regions["boundary"],1) & regions["nonboundary"]).any()))

    def test_strict_protocol_exposes_insufficient_core(self):
        indices,counts,n,reason=audit.matched_indices(self.cube_regions(),np.random.default_rng(42),"strict_four_bands",8)
        self.assertIsNone(indices)
        self.assertEqual(n,98)
        self.assertEqual(counts["foreground_core"],27)
        self.assertEqual(reason,"global_core_cannot_match_four_band_min")

    def test_all_six_matching_is_equal_unique_seeded_and_inside_region(self):
        regions=self.cube_regions()
        first,counts,n,reason=audit.matched_indices(regions,np.random.default_rng(42),"all_six_min",8)
        second,_,_,_=audit.matched_indices(regions,np.random.default_rng(42),"all_six_min",8)
        self.assertEqual(n,27)
        self.assertEqual(reason,"")
        for name,indices in first.items():
            self.assertEqual(len(indices),n)
            self.assertEqual(len(set(indices.tolist())),n)
            self.assertTrue(bool(regions[name].reshape(-1)[indices].all()))
            np.testing.assert_array_equal(indices,second[name])

    def test_empty_core_is_reported(self):
        label=torch.zeros(1,1,9,9,9)
        label[:,:,3:6,3:6,3:6]=1
        indices,_,n,reason=audit.matched_indices(audit.make_regions(label,label.shape[2:]),np.random.default_rng(42),"all_six_min",8)
        self.assertIsNone(indices)
        self.assertEqual(n,0)
        self.assertEqual(reason,"insufficient_region_voxels")

    def test_standardized_transition_and_moments(self):
        feature=torch.tensor([[[[[1.,3.,5.,7.]]],[[[7.,5.,3.,1.]]]]])
        d,moments=audit.standardized_samples(feature,np.array([0,1]),np.array([2,3]))
        np.testing.assert_allclose(d,np.array([-4.,4.])/np.sqrt(1+audit.EPS))
        np.testing.assert_array_equal(moments["var_in"],[1.,1.])
        np.testing.assert_array_equal(moments["var_out"],[1.,1.])

    def test_gradient_alignment_preserves_linear_ramp_and_constant(self):
        d=torch.arange(17)[:,None,None]
        h=torch.arange(17)[None,:,None]
        w=torch.arange(17)[None,None,:]
        ramp=(d+2*h+3*w).float()[None,None]
        g=audit.spatial_gradient(ramp)
        self.assertEqual(tuple(g.shape),tuple(ramp.shape))
        self.assertTrue(torch.equal(g,torch.full_like(g,2)))
        bgs,gb,gn=audit.bgs_vector(ramp,self.cube_regions())
        np.testing.assert_array_equal(bgs,[0.])
        np.testing.assert_array_equal(gb,gn)
        bgs,_,_=audit.bgs_vector(torch.ones_like(ramp),self.cube_regions())
        np.testing.assert_array_equal(bgs,[0.])

    def test_boundary_step_has_positive_gradient_selectivity(self):
        label=torch.zeros(1,1,17,17,17)
        label[:,:,4:13,4:13,4:13]=1
        bgs,gb,gn=audit.bgs_vector(label,audit.make_regions(label,label.shape[2:]))
        self.assertGreater(bgs[0],0.99)
        self.assertGreater(gb[0],0)
        self.assertEqual(gn[0],0)

    def test_bgs_ranking_is_signed(self):
        scores=np.array([-10.,1.,2.,0.])
        self.assertEqual(audit.score_topk(scores,.25),{2})
        self.assertAlmostEqual(audit.signed_spearman(scores,scores),1)
        self.assertAlmostEqual(audit.signed_spearman(scores,-scores),-1)
        self.assertTrue(np.isnan(audit.signed_spearman(np.ones(4),scores)))
        self.assertLess(audit.cosine(scores,-scores),-.999)

    def test_primary_cosine_averages_patches_before_cases(self):
        patches=[dict(case_id="case",patch_id=1,sample_n=8,near=np.array([10.,0.]),far=np.array([10.,0.]),global_=None),
                 dict(case_id="case",patch_id=2,sample_n=8,near=np.array([0.,1.]),far=np.array([0.,1.]),global_=None)]
        patches[0]["global"]=np.array([0.,1.]);patches[1]["global"]=np.array([1.,0.])
        case_vectors={kind:{"case":np.mean([p[kind] for p in patches],axis=0)} for kind in ("near","far","global")}
        _,summary=audit.compare_transitions(["case"],case_vectors,patches,SimpleNamespace(bootstrap_samples=20),
                                           np.random.default_rng(42),logging.getLogger("test"))
        primary=next(r for r in summary if r["metric"]=="near_global_cosine" and r["scope"]=="transition_case_mean_of_patch_cosines")
        secondary=next(r for r in summary if r["metric"]=="near_global_cosine" and r["scope"]=="transition_cosine_of_case_mean_vectors")
        self.assertEqual(primary["mean"],0)
        self.assertGreater(secondary["mean"],.7)

    def test_nonfinite_vector_reports_measurement_and_channel(self):
        stream=io.StringIO();logger=logging.getLogger("specificity_nonfinite")
        logger.addHandler(logging.StreamHandler(stream));logger.setLevel(logging.INFO)
        with self.assertRaises(FloatingPointError):
            audit.finite_vector(np.array([0.,np.inf]),"near","case",1,"identity",logger)
        self.assertIn("kind=near case=case",stream.getvalue())
        self.assertIn("channel=1",stream.getvalue())

    def test_source_is_read_only_analysis(self):
        audit.check_source()
        audit.base.check_analysis_source()


if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
