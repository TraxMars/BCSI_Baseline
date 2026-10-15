"""R1-D1 scientific invariants; CPU tests, no historical result mutations."""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import experiment_R1_D1 as a
from model.vnet import VNet


class D1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_frozen_forward_leaf_gradient_and_other_features(self):
        torch.manual_seed(42)
        model=VNet(n_channels=1,n_classes=2,n_filters=2).eval()
        for p in model.parameters():p.requires_grad_(False)
        state=a.u.snapshot_state(model)
        with torch.no_grad():
            image=torch.randn(1,1,32,32,32);fs=model.encoder(image)
            self.assertTrue(torch.equal(model(image),model.decoder(fs)))
        y=torch.zeros(1,32,32,32,dtype=torch.long);y[:,8:24,8:24,8:24]=1
        inner,outer=a.u.boundary_sides(y)
        g,loss=a.gradient_at_x3(model,fs,y,inner,outer)
        self.assertEqual(g.shape,tuple(fs[2][0].shape));self.assertGreater(np.linalg.norm(g),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not f.requires_grad for f in fs))
        self.assertTrue(a.u.verify_state(model,state)['weights_and_buffers_exactly_unchanged'])
        self.assertTrue(np.isfinite(loss))

    def test_direction_sign_and_undefined_zero_norm(self):
        g=np.ones((3,2,2,2));delta=-g.copy();delta[:,0]=0
        aa,defined,dot,_,_=a.alignment(g,delta)
        self.assertTrue(np.all(aa[defined]>0));self.assertFalse(defined[0].any())
        self.assertLess(dot.sum(),0)
        z=a.alignment_stats(g,np.zeros_like(g),np.ones((2,2,2),bool))
        self.assertIsNone(z['alignment_mean']);self.assertIsNotNone(z['undefined_reason'])
        z=a.alignment_stats(np.zeros_like(g),delta,np.ones((2,2,2),bool))
        self.assertIsNone(z['aligned_fraction'])

    def test_finite_difference_analytic_boundary_loss(self):
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__()
            def decoder(self,features):
                x=features[2][:,:1];return torch.cat([-x,x],1)
        model=Tiny().eval();x=torch.linspace(-.3,.3,64).reshape(1,1,4,4,4)
        fs=[x,x,x,x,x];y=torch.zeros(1,4,4,4,dtype=torch.long);y[:,:2]=1
        inner,outer=a.u.boundary_sides(y)
        g,lb=a.gradient_at_x3(model,fs,y,inner,outer)
        delta=-g/np.max(np.abs(g))*.1
        expected=float((g.astype(np.float64)*delta).sum())
        self.assertGreater(abs(expected),1e-4)
        rows=a.fd_check(model,fs,y,inner,outer,delta,expected,lb)
        self.assertTrue(all(r['central_pass'] for r in rows))

    def test_precision_flags_and_restoration_after_exception(self):
        previous=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        with self.assertRaises(RuntimeError):
            with a.computation_precision(True) as state:
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertEqual(state['precision'],'strict_fp32')
                raise RuntimeError('planned failure')
        self.assertEqual(previous,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))

    def test_source_resolution_never_changes_candidate_path_or_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'old_source';source.write_bytes(b'old source')
            candidate=p/'candidate';candidate.write_bytes(b'unchanged prediction')
            old_code='/original/code.py';code_sha=a.r0.sha256(source)
            seal=dict(phaseA_complete=True,artifacts=[dict(path=old_code,sha256=code_sha),a.s.artifact(candidate,'candidate')])
            meta=dict(code_sha256={old_code:code_sha})
            audit=dict(source_snapshots=[dict(original_path=old_code,snapshot_path=str(source),sha256=code_sha)])
            resolved=a.resolve_phase_a_seal(seal,meta,audit)
            self.assertEqual(resolved['artifacts'][1],seal['artifacts'][1])
            self.assertEqual(seal['artifacts'][0]['path'],old_code)
            self.assertEqual(resolved['artifacts'][0]['path'],str(source))
            source.write_bytes(b'wrong snapshot')
            with self.assertRaises(RuntimeError):a.resolve_phase_a_seal(seal,meta,audit)

    def test_global_GT_gate_and_tamper_detection(self):
        r=a.Reader(dict(data_sources=[],sampling=dict(patch_coordinates=[])))
        r.start_fold(0,'r',list('abcdefg'))
        r.sealed=True
        with self.assertRaises(RuntimeError):r.read('r',0,'label')
        with self.assertRaises(RuntimeError):r.evaluation_fold(0,'r',list('abcdefg'))
        with self.assertRaises(RuntimeError):r.unlock(dict(phaseA_complete=False,artifacts=[]))
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'candidate';f.write_bytes(b'fixed')
            seal=dict(phaseA_complete=True,artifacts=[a.s.artifact(f,'candidate')])
            r.unlock(seal);self.assertTrue(r.global_sealed)
            f.write_bytes(b'tampered')
            with self.assertRaises(RuntimeError):r.unlock(seal)
        with self.assertRaises(ValueError):r.start_fold(0,'r',['r']+list('abcdef'))

    def test_phase_A_no_receiver_GT_or_oracle_API(self):
        for fn in (a.phase_a_patch,a.match_bank,a.build_banks):
            self.assertFalse(set(inspect.signature(fn).parameters)&{'gt','receiver_gt','gradient','labels'})
        source=inspect.getsource(a.phase_a_patch)
        self.assertNotIn('autograd',source);self.assertNotIn("read(case,patch,'label')",source)
        self.assertNotIn('gradient_at_x3',source)

    def test_bank_class_identity_and_global_equivalence(self):
        rng=np.random.RandomState(42);x=rng.randn(4,3,3,3).astype(np.float32)
        p=rng.randn(2,4).astype(np.float32);mean=np.zeros(4,np.float32);scale=np.ones(4,np.float32)
        b=a.match_bank(x,p,np.array([0,1]),mean,scale);g=a.s.match(x,p,mean,scale)
        self.assertTrue(np.array_equal(b['chosen'],g['chosen']))
        np.testing.assert_allclose(b['q'],g['q'],atol=1e-6)
        np.testing.assert_array_equal(b['target'],g['target'])
        with self.assertRaises(ValueError):a.match_bank(x,p,np.array([0,0]),mean,scale)

    def test_local_bank_donor_identity_determinism_and_cluster_inverse(self):
        rng=np.random.RandomState(9);ids=list('abcdefgh');libs={};xs={}
        shape=(4,4,4)
        for case in ids[1:]:
            grids=[]
            for p in range(3):
                xs[case,p]=rng.randn(4,*shape).astype(np.float32)
                fg=np.zeros(shape,bool);fg[:2]=True
                grids.append(dict(boundary_fg=fg,boundary_bg=~fg))
            libs[case]=dict(grids=grids)
        donor=dict(mean=np.array([1,2,3,4],np.float32),scale=np.array([2,3,4,5],np.float32))
        b,rows=a.build_banks(xs,libs,donor,'a',ids,0)
        b2,_=a.build_banks(xs,libs,donor,'a',ids,0)
        self.assertTrue(np.array_equal(b['multi']['raw'],b2['multi']['raw']))
        self.assertEqual(b['multi']['raw'].shape,(8,4))
        self.assertEqual(np.bincount(b['multi']['classes']).tolist(),[4,4])
        for vec,cls,src in zip(b['local']['raw'],b['local']['classes'],b['local']['source']):
            case=ids[src[0]];self.assertNotEqual(case,'a')
            np.testing.assert_array_equal(vec,xs[case,src[1]].reshape(4,-1)[:,src[2]])
            self.assertTrue(libs[case]['grids'][src[1]]['boundary_'+('fg' if cls==0 else 'bg')].ravel()[src[2]])
        with self.assertRaises(ValueError):a.build_banks(xs,dict(libs,a={}),donor,'a',ids,0)

    def test_q_bins_include_one_exactly_once(self):
        q=np.array([0,.249,.25,.5,.75,1.],np.float32)
        bins=a.q_groups(q)
        self.assertTrue(np.all(sum(v.astype(int) for _,v in bins)==1))
        self.assertTrue(bins[-1][1][-1])

    def test_mixed_GT_not_forced_class_and_correction_sign(self):
        gt=np.zeros((8,8,8),int);gt[:4]=1
        gt[4:6,:4,:4]=1  # one .5 mixed grid
        p0=gt.astype(bool).copy();p0[0,0,0]=False
        p1=gt.astype(bool).copy();p1[7,7,7]=True
        shape=(2,2,2);m=dict(chosen=np.ones(shape,int),q=np.ones(shape),valid=np.ones(shape,bool))
        m['chosen'][0]=0
        sem,con,qrows,types=a.semantic_metrics(m,np.ones(shape,bool),gt,p0,p1,np.ones(gt.shape)*.001)
        self.assertEqual(sem[0]['mixed_positions'],1)
        self.assertEqual(sum(r['corrected'] for r in qrows if r['group']!='continuous'),1)
        self.assertEqual(sum(r['harmed'] for r in qrows if r['group']!='continuous'),1)
        self.assertEqual(sum(r['positions'] for r in types),7)

    def test_actual_residual_cap_mask_and_energy(self):
        rng=np.random.RandomState(42);x=rng.rand(4,4,4,4).astype(np.float32)
        mask=np.zeros((4,4,4),bool);mask[1]=True
        q=np.ones(mask.shape,np.float32)
        new,st=a.s.apply_residual(x,np.ones_like(x)*100,mask,q,.02)
        self.assertTrue(np.array_equal(new[:,~mask],x[:,~mask]))
        self.assertLessEqual(st['maximum_position_relative_l2'],.100002)
        equal,es=a.s.apply_residual(x,np.zeros_like(x),mask,q,.02,st['residual_l2']*.5)
        self.assertLess(es['energy_match_relative_error'],1e-5)
        zero,_=a.s.apply_residual(x,np.zeros_like(x),mask,q,0)
        np.testing.assert_array_equal(zero,x)

    def test_case_aggregation_not_random_or_patch_independence(self):
        rows=[dict(case_id='a',patch_index=0,method='m',repetition=i,value=float(i)) for i in range(20)]
        rows += [dict(case_id='a',patch_index=1,method='m',value=0.),dict(case_id='b',patch_index=0,method='m',value=100.)]
        cases,_=a.group_case(rows,['method'])
        self.assertEqual(len(cases),2);self.assertEqual(cases[0]['value'],4.75)
        self.assertEqual(a.s.summarize([r['value'] for r in cases])['mean'],52.375)

    def test_lossless_archive_and_refuse_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz';x=np.random.RandomState(42).randn(3,4,5).astype(np.float32)
            a.save_npz(p,x=x)
            with np.load(p) as z:np.testing.assert_array_equal(z['x'].view(np.uint32),x.view(np.uint32))
            with self.assertRaises(FileExistsError):a.save_npz(p,x=x)
        with self.assertRaises(ValueError):a.create_output('/tmp/not_a_run')


if __name__=='__main__':unittest.main(verbosity=2)
