"""A/B ablation correctness, isolation, legacy equivalence and factorial summaries."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from src.augmentation import appearance_transform
from src.dataset import PolypDataset
from src.metrics import boundary, boundary_target, segmentation_loss, training_objective
from src.model import UNet, build_model
from scripts.run_baselines import aggregate, sha, validate_protocol, write

ROOT = Path(__file__).resolve().parents[1]


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(91)
        self.config = json.loads((ROOT/'configs/ablation_pilot/AB_seed20261020.json').read_text())

    def test_shared_initialization_rng_and_legacy_logits(self):
        baseline = UNet(2)
        state = torch.get_rng_state()
        torch.manual_seed(91)
        aux = UNet(2, boundary_head=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, aux.state_dict()[key]))
        x = torch.randn(2,3,33,49)
        self.assertTrue(torch.equal(baseline(x), aux(x)))
        # The actual frozen v1 implementation is independent evidence of compatibility.
        source=ROOT/'results/unet_baseline_v1/source/src/model.py'
        if source.exists():
            spec=importlib.util.spec_from_file_location('frozen_v1_model',source)
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            torch.manual_seed(91); old=module.UNet(2)
            self.assertTrue(torch.equal(old(x), baseline(x)))

    def test_boundary_target_empty_full_border_and_width(self):
        masks=torch.zeros(3,1,16,16)
        masks[1]=1; masks[2,0,4:12,4:12]=1
        one=boundary_target(masks,1)
        for index in range(3):
            self.assertTrue(np.array_equal(one[index,0].numpy().astype(bool),boundary(masks[index,0].numpy())))
        two=boundary_target(masks,2)
        self.assertEqual(two[0].sum(),0)
        self.assertEqual(two[1].sum(),16*16-12*12)
        self.assertEqual(two[2].sum(),8*8-4*4)
        self.assertTrue((two<=masks).all())
        with self.assertRaises(ValueError): boundary_target(masks,0)

    def test_auxiliary_gradients_loss_and_inference_skip(self):
        config={**self.config,'base_channels':2}
        model=build_model(config)
        x=torch.randn(2,3,33,49); target=(torch.rand(2,1,33,49)>0.7).float()
        outputs=model(x,return_boundary=True)
        total,parts=training_objective(outputs,target,config['boundary_supervision'])
        self.assertTrue(torch.allclose(total,parts['segmentation']+0.2*parts['boundary']))
        total.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(model.boundary_head.weight.grad.abs().sum()),0)
        calls=[]
        hook=model.boundary_head.register_forward_hook(lambda *args: calls.append(True))
        model.eval()
        self.assertEqual(model(x).shape,(2,1,33,49))
        self.assertEqual(calls,[])
        hook.remove()
        with self.assertRaises(ValueError): UNet(2)(x,return_boundary=True)

    def test_disabled_objective_is_exact_legacy_loss(self):
        x=torch.randn(2,1,32,32,requires_grad=True); target=torch.zeros_like(x)
        actual,_=training_objective(x,target,{'enabled':False})
        expected=segmentation_loss(x,target)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0]))

    def test_appearance_repeatable_and_preserves_rng_geometry_and_masks(self):
        array=np.arange(32*48*3,dtype=np.uint8).reshape(32,48,3)
        config={**self.config['appearance_augmentation'],'color_probability':1,'blur_probability':1,'jpeg_probability':1}
        image=torch.from_numpy(array.transpose(2,0,1).copy()).float()/255
        state=torch.get_rng_state()
        first=appearance_transform(image,config); second=appearance_transform(image,config)
        self.assertTrue(torch.equal(first,second)); self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertFalse(torch.equal(first,image))
        self.assertTrue(torch.isfinite(first).all() and first.min()>=0 and first.max()<=1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); mask=np.zeros((32,48),dtype=np.uint8); mask[3:15,11:36]=255
            Image.fromarray(array).save(root/'image.png'); Image.fromarray(mask).save(root/'mask.png')
            manifest=root/'manifest.csv'
            with manifest.open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=['id','image','mask']);writer.writeheader()
                writer.writerow({'id':'one','image':root/'image.png','mask':root/'mask.png'})
            plain=PolypDataset(manifest,(48,32),augment=True)
            changed=PolypDataset(manifest,(48,32),augment=True,appearance=config)
            for seed in range(10):
                torch.manual_seed(seed); p=plain[0]; state=torch.get_rng_state()
                torch.manual_seed(seed); q=changed[0]
                self.assertTrue(torch.equal(p['mask'],q['mask']))
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
            val=PolypDataset(manifest,(48,32),appearance=config)
            self.assertTrue(torch.equal(val[0]['image'],image))
            self.assertTrue(torch.equal(val[0]['image'],val[0]['image']))

    def test_condition_seed_validation_and_factorial_aggregation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'source').mkdir()
            baseline={'runs':[{'seed':s,'dice':0.5+0.1*i,'iou':0.5+0.1*i,'boundary_f1':0.5+0.1*i} for i,s in enumerate([1,2])]}
            write(root/'source/baseline.json',baseline)
            runs=[]
            for condition,delta in [('A',0.05),('B',0.10),('AB',0.18)]:
                for i,seed in enumerate([1,2]):
                    name=f'{condition}_{seed}'; directory=root/name
                    (directory/'validation').mkdir(parents=True)
                    (directory/'best.pt').write_bytes(b'fake hash fixture, not loaded')
                    write(directory/'validation/metrics.json',{'count':2,**{k:0.5+0.1*i+delta for k in ['dice','iou','boundary_f1']}})
                    write(directory/'validation/provenance.json',{'epoch':1,'manifest_sha256':'validation','checkpoint_sha256':sha(directory/'best.pt')})
                    runs.append({'name':name,'condition':condition,'seed':seed,'epochs':1,'config':'config.json'})
            protocol={'protocol_id':'test','status':'training_frozen','evaluate_test':False,'runs':runs,
                      'files_sha256':{'config.json':'stub','baseline.json':sha(root/'source/baseline.json')},
                      'validation_manifest_sha256':'validation','baseline_reference':'baseline.json'}
            validate_protocol(protocol)
            result=aggregate(protocol,root)
            self.assertNotIn('metrics',result)
            self.assertEqual(set(result['by_condition']),{'A','B','AB'})
            effect=result['validation_factorial_effects']['interaction_AB_minus_A_minus_B_plus_baseline']
            self.assertAlmostEqual(effect['metrics']['dice']['mean'],0.03)
            self.assertAlmostEqual(result['by_condition']['A']['metrics']['dice']['mean'],0.60)
            protocol['runs'].append(runs[0].copy())
            with self.assertRaises(ValueError): validate_protocol(protocol)


if __name__=='__main__': unittest.main()
