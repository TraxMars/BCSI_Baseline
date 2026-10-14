"""R1-S0 protocol checks: GT gate, frozen replay, geometry and residual signs."""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import experiment_R1_S0 as a
from model.vnet import VNet


class R1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_original_forward_replay_frozen_and_alpha_zero(self):
        model=VNet(n_channels=1,n_classes=2,n_filters=2).eval()
        for p in model.parameters(): p.requires_grad_(False)
        state=a.u.snapshot_state(model)
        with torch.no_grad():
            image=torch.randn(1,1,32,32,32)
            features=model.encoder(image)
            original=model(image)
            self.assertTrue(torch.equal(original,model.decoder(features)))
            x=features[2][0].numpy()
            new,stats=a.apply_residual(x,np.ones_like(x),np.ones(x.shape[1:],bool),np.ones(x.shape[1:]),0.)
            self.assertTrue(np.array_equal(x,new))
            modified=a.replay_features(features,new)
            self.assertTrue(torch.equal(original,model.decoder(modified)))
            self.assertTrue(all(modified[i] is features[i] for i in (0,1,3,4)))
            self.assertEqual(stats['residual_l2'],0)
        self.assertTrue(a.u.verify_state(model,state)['weights_and_buffers_exactly_unchanged'])

    def test_receiver_GT_gate_and_seal(self):
        reader=a.FoldReader(dict(data_sources=[],sampling=dict(patch_coordinates=[])))
        with self.assertRaises(RuntimeError): reader.read('receiver',0,'label')
        with self.assertRaises(ValueError): reader.start_fold(0,'receiver',['receiver']+list('abcdef'))
        reader.start_fold(0,'receiver',list('abcdefg'))
        with self.assertRaises(RuntimeError): reader.read('receiver',0,'label')
        with self.assertRaises(RuntimeError): reader.seal([],0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prediction.bin';path.write_bytes(b'fixed predictions')
            art=a.artifact(path,'prediction')
            with self.assertRaises(RuntimeError): reader.seal([art],2)
            reader.seal([art],1);self.assertTrue(reader.sealed)
            reader.start_fold(1,'b',['receiver']+list('acdefg'))
            with self.assertRaises(RuntimeError): reader.read('b',0,'label')
            path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError): reader.seal([art],1)

    def test_phaseA_api_does_not_accept_receiver_labels(self):
        for fn in (a.match,a.apply_residual,a.risk_mask,a.random_mask,a.self_prototypes):
            self.assertFalse(any(k in inspect.signature(fn).parameters for k in ('gt','label','labels','receiver_gt')))
        source=inspect.getsource(a.receiver_phaseA)
        self.assertNotIn('"label"',source)
        self.assertNotIn("'label'",source)
        with patch.object(a.FoldReader,'read',side_effect=AssertionError('GT forbidden')):
            x=np.ones((4,4,4,4),np.float32)
            a.match(x,np.array([[2]*4,[0]*4],np.float32),np.zeros(4),np.ones(4))
            a.self_prototypes(x,np.ones((16,16,16),np.float32)*.7)

    def test_gt_to_grid_purity_and_boundary_geometry(self):
        mask=np.zeros((32,32,32),bool);mask[:16]=True
        c=a.donor_candidates(mask)
        self.assertTrue(np.all(c['foreground_fraction'][:4]==1))
        self.assertTrue(np.all(c['foreground_fraction'][4:]==0))
        self.assertTrue(np.all(c['boundary_fg'][3]))
        self.assertTrue(np.all(c['boundary_bg'][4]))
        self.assertFalse(c['boundary_bg'][:4].any())
        self.assertTrue(c['nonboundary_fg'][0].all())
        labels=torch.from_numpy(mask.astype(np.int64))[None]
        inner,outer=a.u.boundary_sides(labels)
        self.assertTrue(np.array_equal(a.r0.boundary(mask),(inner|outer)[0].numpy()))
        self.assertTrue(np.array_equal(a.r05.lift_grid(c['foreground_fraction']>0),mask))

    def test_donor_equal_case_weight_and_receiver_exclusion(self):
        libs={str(i):dict(case_id=str(i),mean=np.ones(4)*i,second=np.ones(4)*(i*i+1),
                          prototypes={'boundary':np.ones((2,4))*i}) for i in range(7)}
        d=a.combine_libraries(libs,'held')
        self.assertTrue(np.all(d['prototypes']['boundary']==3))
        self.assertTrue(np.all(d['mean']==3))
        self.assertTrue(np.allclose(d['scale'],np.sqrt(5)))
        with self.assertRaises(ValueError): a.combine_libraries(libs,'0')

    def test_matching_raw_space_label_swap_and_opposite(self):
        x=np.ones((4,2,2,2),np.float32);x[0]=3
        p=np.array([[3,1,1,1],[1,3,1,1]],np.float32)
        mean=np.zeros(4,np.float32);scale=np.array([2,1,3,1],np.float32)
        m=a.match(x,p,mean,scale);sw=a.match(x,p[::-1],mean,scale)
        self.assertTrue(np.array_equal(m['target'],sw['target']))
        self.assertTrue(np.array_equal(m['q'],sw['q']))
        self.assertTrue(np.all(m['chosen']==0))
        self.assertTrue(np.allclose(m['target'][:,0,0,0],p[0]))
        opposite=p[1-m['chosen']].transpose(3,0,1,2)
        self.assertFalse(np.array_equal(m['target'],opposite))
        self.assertTrue(np.isfinite(m['q']).all())

    def test_residual_shape_mask_cap_and_energy_fairness(self):
        x=np.ones((4,4,4,4),np.float32)
        target=np.ones_like(x)*100
        mask=np.zeros((4,4,4),bool);mask[1,1,1]=True;mask[2,2,2]=True
        q=np.ones(mask.shape,np.float32);q[2,2,2]=0
        changed,stats=a.apply_residual(x,target,mask,q,.05)
        self.assertEqual(changed.shape,x.shape)
        self.assertTrue(np.array_equal(changed[:,~mask],x[:,~mask]))
        self.assertTrue(np.array_equal(changed[:,2,2,2],x[:,2,2,2]))
        self.assertLessEqual(stats['maximum_position_relative_l2'],a.CAP+1e-6)
        self.assertGreater(stats['clipped_fraction'],0)
        equal,ss=a.apply_residual(x,-target,mask,q,.02,stats['residual_l2'])
        self.assertTrue(ss['energy_match_feasible'])
        self.assertLess(ss['energy_match_relative_error'],1e-5)
        partial,sp=a.apply_residual(x,target,mask,q,.02,channels=[0])
        self.assertTrue(np.array_equal(partial[1:],x[1:]))
        with self.assertRaises(ValueError): a.apply_residual(x,target,mask[:-1],q,.02)

    def test_invalid_or_tied_match_allows_no_correction(self):
        x=np.ones((4,2,2,2),np.float32)
        for p in (None,np.ones((2,4),np.float32)):
            m=a.match(x,p,np.zeros(4),np.ones(4))
            changed,stats=a.apply_residual(x,m['target'],np.ones((2,2,2),bool),m['q'],.05)
            self.assertTrue(np.array_equal(changed,x))
            self.assertEqual(stats['changed_positions'],0)

    def test_random_reproducibility_independence_and_budget(self):
        before=torch.get_rng_state().clone();npbefore=np.random.get_state()
        masks=[a.random_mask((28,28,20),314,42+i*101) for i in range(20)]
        self.assertTrue(all(m.sum()==314 for m in masks))
        self.assertTrue(np.array_equal(masks[0],a.random_mask((28,28,20),314,42)))
        self.assertFalse(np.array_equal(masks[0],masks[1]))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(np.array_equal(npbefore[1],np.random.get_state()[1]))

    def test_historical_top2_tie_rule_and_flip_inverse(self):
        score=np.zeros((4,4,4),np.float32);m=a.risk_mask(score,2,1)
        seed=42+2*1009+9176+20000003
        ties=np.random.RandomState(seed).permutation(score.size)
        expected=np.zeros(score.size,bool)
        expected[a.r0.ordered_indices(score,np.ones(score.shape,bool),ties)[:a.r0.budget_count(score.size,.02)]]=True
        self.assertTrue(np.array_equal(m,expected.reshape(score.shape)))
        x=torch.arange(8*12*16).reshape(8,12,16)
        for axis in range(3): self.assertTrue(torch.equal(x,torch.flip(torch.flip(x,[axis]),[axis])))
        large=np.arange(16**3,dtype=np.float32).reshape(16,16,16)
        self.assertEqual(a.r05.map_to_grid(large,(4,4,4),'max')[0,0,0],large[:4,:4,:4].max())

    def test_gain_correction_harm_sign_and_loss_consistency(self):
        gt=np.zeros((16,16,16),np.int64);gt[:8]=1
        pred=gt.astype(bool);pred[0,0,0]=False
        p=np.where(pred,.8,.2).astype(np.float32)
        after=gt.astype(bool);after[-1,-1,-1]=True
        pa=np.where(after,.8,.2).astype(np.float32)
        logits=np.stack([np.zeros_like(pa),np.log(pa/(1-pa))]).astype(np.float32)
        m=dict(chosen=np.zeros((4,4,4),np.uint8),q=np.ones((4,4,4),np.float32))
        result=a.evaluate_prediction(logits,pa,after,gt,pred,p,np.ones((4,4,4),bool),m)
        self.assertEqual(result['corrected'],1)
        self.assertEqual(result['harmed'],1)
        self.assertEqual(result['net_correction'],0)
        self.assertAlmostEqual(result['boundary_ce'],.5*(result['inner_ce']+result['outer_ce']))
        self.assertAlmostEqual(result['ce_plus_dice_sum'],2*result['supervised_ce_dice'])
        self.assertTrue(all(np.isfinite(v) for v in result.values() if isinstance(v,(int,float))))

    def test_lossless_prediction_storage(self):
        logits=torch.randn(1,2,8,8,8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'z.npz';a.save_prediction(path,logits)
            z,p,pred=a.read_prediction(path)
            self.assertTrue(np.array_equal(z,logits[0].numpy()))
            self.assertTrue(np.array_equal(pred,(torch.softmax(logits,1)[0,1]>.5).numpy()))
            self.assertTrue(np.allclose(p,torch.softmax(logits,1)[0,1].numpy(),atol=2e-7))
            changed=logits.clone();changed[:,:,2:4,2:4,2:4]+=.003
            delta_path=Path(tmp)/'delta.npz'
            a.save_prediction(delta_path,changed,logits[0].numpy(),path)
            zz,pp,bits=a.read_prediction(delta_path)
            self.assertTrue(np.array_equal(zz.view(np.uint32),changed[0].numpy().view(np.uint32)))
            self.assertTrue(np.array_equal(bits,(torch.softmax(changed,1)[0,1]>.5).numpy()))

    def test_case_statistics_not_patch_or_random_independence(self):
        rows=[]
        for case in ('a','b'):
            for patch in range(3):
                for rep in range(20):
                    rows.append(dict(case_id=case,patch_index=patch,repetition=rep,mapping='max',method='random_mask',alpha=.02,
                        boundary_ce=1.,full_ce=1.,dice=.5,net_correction=0.,boundary_ce_gain=1. if case=='a' else 3.))
        rr,summary,_=a.aggregate(rows,['a','b'])
        self.assertEqual(len(rr),2)
        s=summary['max|random_mask|0.02']['boundary_ce_gain']
        self.assertEqual(s['n_cases'],2);self.assertEqual(s['mean'],2.)
        self.assertTrue(all(r['n_raw_results']==60 for r in rr))

    def test_existing_output_rejected(self):
        with self.assertRaises(ValueError): a.create_output(a.ROOT)


if __name__=='__main__': unittest.main(verbosity=2)
