"""CPU checks for ablation signs, preservation, probe and LOCO isolation."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import inspect
import numpy as np
import torch
import experiment_donor_D0 as a


class ToyDecoder(torch.nn.Module):
    def forward(self,features):
        f=features[2][:,:1]
        return torch.cat([torch.zeros_like(f),f],dim=1)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=ToyDecoder()


class D0Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        a.threadpool_limits(limits=2)

    def test_interventions_preserve_inputs_other_layers_and_means(self):
        features=[torch.arange(32.).reshape(1,4,2,2,2).clone() for _ in range(5)]
        preserved=[f.clone() for f in features]
        for mode in a.MODES:
            changed,relative=a.intervene(features,[1],mode)
            self.assertTrue(all(changed[i] is features[i] for i in (0,1,3,4)))
            self.assertTrue(torch.equal(changed[2][:,0],features[2][:,0]))
            if mode=="spatial_mean":
                self.assertEqual(changed[2][:,1].std(),0)
                self.assertEqual(changed[2][:,1].mean(),features[2][:,1].mean())
            else: torch.testing.assert_allclose(changed[2][:,1],features[2][:,1]*a.FACTORS[mode])
            self.assertGreater(relative,0)
            self.assertTrue(all(torch.equal(f,p) for f,p in zip(features,preserved)))
        with self.assertRaises(ValueError): a.intervene(features,[1,1],"zero")

    def test_contribution_sign_and_first_order_identity(self):
        y=torch.tensor([[[[1,1],[1,1]],[[0,0],[0,0]]]])
        f=torch.zeros(1,16,2,2,2); f[:,0]=torch.where(y>0,2.,-2.)
        features=[f.clone() for _ in range(5)]
        model=ToyModel().eval()
        base=a.loss_values(model.decoder(features),y,y>0,y==0)
        rr=a.evaluate_interventions(model,features,y,y>0,y==0,[dict(channels=[0],mode=m) for m in a.MODES],2)
        self.assertTrue(all(r["boundary_ce"]>base["boundary_ce"] for r in rr))
        utility=2/(1+np.exp(2))  # U=-dL/da; C(.05) ~= .05 U.
        self.assertLess(abs((rr[0]["boundary_ce"]-base["boundary_ce"])/(.05*utility)-1),.08)
        self.assertAlmostEqual(base["boundary_ce"],.5*(base["inner_ce"]+base["outer_ce"]))
        self.assertAlmostEqual(base["ce_plus_dice_sum"],2*base["supervised_ce_dice"])

    def test_per_intervention_loss_never_pooled_dice(self):
        y=torch.tensor([[[[1,0],[1,0]],[[1,0],[1,0]]]])
        f=torch.randn(1,16,2,2,2); features=[f.clone() for _ in range(5)]
        ss=[dict(channels=[0],mode=m) for m in a.MODES]
        single=a.evaluate_interventions(ToyModel(),features,y,y>0,y==0,ss,1)
        batched=a.evaluate_interventions(ToyModel(),features,y,y>0,y==0,ss,4)
        for aa,bb in zip(single,batched):
            for key in ("boundary_ce","full_ce","supervised_ce_dice","inner_ce","outer_ce"):
                self.assertAlmostEqual(aa[key],bb[key],places=7)
        with self.assertRaises(ValueError): a.loss_values(torch.randn(2,2,2,2,2),y,y>0,y==0)

    def test_loco_selection_excludes_held_out_and_randoms_complete(self):
        fields=("boundary_contribution","bgs","full_ce_contribution","energy","utility")
        dev={str(i):{k:np.arange(16,dtype=float) for k in fields} for i in range(7)}
        state=np.random.get_state(); selections,_=a.fold_selection(dev,"held",42)
        self.assertEqual(len(selections),48)
        self.assertTrue(np.array_equal(state[1],np.random.get_state()[1]))
        for k in a.KS:
            rr=[r for r in selections if r["method"]=="random" and r["k"]==k]
            self.assertEqual(len(rr),20)
            self.assertTrue(all(len(set(r["channels"]))==k for r in rr))
        for rep in range(20):
            aa=next(r for r in selections if r["method"]=="random" and r["k"]==8 and r["repetition"]==rep)
            bb=next(r for r in selections if r["method"]=="random" and r["k"]==16 and r["repetition"]==rep)
            self.assertEqual(aa["channels"],bb["channels"][:8])
        with self.assertRaises(ValueError): a.fold_selection({**dev,"held":dev["0"]},"held",42)

    def test_role_GT_gate(self):
        reader=a.RoleReader(dict(data_sources=[],sampling=dict(patch_coordinates=[])))
        reader.start_fold(0,"held")
        with self.assertRaises(RuntimeError): reader.read("held",0,"label")
        with tempfile.TemporaryDirectory() as tmp:
            plan=Path(tmp)/"plan.json";plan.write_text("fixed developer-only plan")
            with self.assertRaises(RuntimeError): reader.seal_plan(plan,"wrong")
            reader.seal_plan(plan,a.r0.sha256(plan));self.assertTrue(reader.plan_sealed)

    def test_developer_only_standardization_and_equal_case_weights(self):
        cache={}
        for case in range(7):
            for p in range(3):
                cache[str(case),p]=dict(sample_valid=True,sample_x=np.ones((10+2*case,16))*case,
                   sample_y=np.r_[np.ones(5+case,int),np.zeros(5+case,int)])
        cache["held",0]=dict(sample_x=np.ones((10,16))*1e9)
        x,y,w,mean,scale,_=a.development_matrix(cache,[str(i) for i in range(7)],"held")
        np.testing.assert_allclose(mean,3)
        np.testing.assert_allclose(np.average(x,axis=0,weights=w),0,atol=1e-12)
        with self.assertRaises(ValueError): a.development_matrix(cache,["held"]+[str(i) for i in range(6)],"held")

    def test_balanced_samples_and_probe_prediction_without_GT(self):
        feature=np.random.RandomState(42).normal(size=(16,8,8,8)).astype(np.float32)
        inner=np.zeros((8,8,8),bool);inner[:4]=True;outer=~inner
        x,y,indices,status=a.sampled_probe_data(feature,inner,outer,42)
        self.assertEqual((y==1).sum(),(y==0).sum())
        self.assertEqual(len(indices),len(np.unique(indices)))
        self.assertEqual(set(inspect.signature(a.predict_probe).parameters),{"feature","channels","mean","scale","probe"})
        x[:,0]=2*y-1; probe=a.fit_probe(x,y,np.ones(len(y)),[0])
        feature[0]=np.where(inner,1.,-1.)
        p=a.predict_probe(feature,[0],np.zeros(16),np.ones(16),probe)
        metrics=a.probe_metrics(p,inner,outer)
        self.assertEqual(metrics["auroc"],1);self.assertEqual(metrics["balanced_accuracy"],1)
        self.assertIsNone(a.probe_metrics(p,np.zeros_like(inner),np.zeros_like(inner))["auroc"])

    def test_output_refusal_and_absolute_code_paths(self):
        self.assertEqual(a.parse_args([]).decoder_batch_size,1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(a,"ROOT",Path(tmp)):
                out=a.create_output(Path(tmp)/"analysis/results/experiment_donor_D0/run_20261013_000000")
                f=out/"keep.txt";f.write_text("keep")
                with self.assertRaises(FileExistsError): a.create_output(out)
                self.assertEqual(f.read_text(),"keep")
        self.assertTrue(all(p.is_absolute() and p.is_file() for p in a.NEW_CODE))

    def test_frozen_real_VNet_batch_eval(self):
        model=a.u.VNet(n_channels=1,n_classes=2,n_filters=2).eval()
        for p in model.parameters(): p.requires_grad_(False)
        before=a.u.snapshot_state(model)
        with torch.no_grad():
            x=torch.randn(1,1,32,32,32); features=model.encoder(x); base=model(x)
            torch.testing.assert_allclose(model.decoder(features),base,rtol=0,atol=0)
            batch=model.decoder([f.expand(2,-1,-1,-1,-1) for f in features])
            torch.testing.assert_allclose(batch,base.expand_as(batch),rtol=1e-5,atol=2e-5)
            changed,_=a.intervene(features,[0],"zero");self.assertEqual(model.decoder(changed).shape,base.shape)
        self.assertTrue(a.u.verify_state(model,before)["weights_and_buffers_exactly_unchanged"])

    def test_bootstrap_units_and_missing_values(self):
        stat=a.bootstrap([1,2,None,3]);self.assertEqual(stat["n_cases"],3)
        self.assertEqual(stat["unit"],"case");self.assertEqual(stat["repetitions"],10000)
        self.assertIsNone(a.correlation(np.ones(64),np.ones(64))[0]);self.assertIsNone(a.bootstrap([None])["mean"])

    def test_complete_LOCO_aggregation_preserves_random20_and_pairs(self):
        ids=[str(i) for i in range(8)]
        fields=("boundary_contribution","bgs","full_ce_contribution","energy","utility")
        all_rows=[]
        for held in ids:
            dev={i:{k:np.arange(16,dtype=float) for k in fields} for i in ids if i!=held}
            selections,_=a.fold_selection(dev,held,42)
            for s in selections:
                for p in range(3):
                    value=.8 if s["method"]=="boundary_contribution" else .6
                    all_rows.append(dict(case_id=held,k=s["k"],method=s["method"],repetition=s["repetition"],patch_index=p,
                        delta_boundary_ce=value,delta_full_ce=value,delta_supervised_ce_dice=value,
                        relative_feature_change=.2,auroc=value,balanced_accuracy=value,ap_foreground_side=value,undefined_reason=""))
        repetitions,cases,paired,summary=a.aggregate_loco(all_rows,ids)
        self.assertEqual(len(all_rows),1152)
        self.assertEqual(len(cases),8*2*5)
        self.assertTrue(all(r["n_repetitions"]==20 for r in cases if r["method"]=="random"))
        self.assertAlmostEqual(summary["8/boundary_minus_random"]["auroc"]["mean"],.2)
        self.assertEqual(summary["8/boundary_minus_random"]["auroc"]["n_cases"],8)
        with self.assertRaises(AssertionError):a.aggregate_loco(all_rows[:-1],ids)


if __name__=="__main__": unittest.main(verbosity=2)
