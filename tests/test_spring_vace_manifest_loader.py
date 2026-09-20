"""Independent regression cases for explicit Spring indices and export commits."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

LOADER = Path('/music-3d-shared-disk/user/KAIST/MK/foundation_models/DiffSynth-Studio-ref_keyframes_fixed/diffsynth/core/data/dataset_spring.py')
spec = importlib.util.spec_from_file_location('spring_manifest_regression', LOADER)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SpringManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample_dir = self.root/'samples'/'arbitrary_window_name'
        self.sample_dir.mkdir(parents=True)
        self.ctx = list(range(100,123,2))
        self.target = list(range(100,123))
        paths = []
        for i in range(23):
            p = self.root/f'asset_{chr(65+i)}.png'
            Image.fromarray(np.full((8,8,3),i*10,dtype=np.uint8)).save(p)
            paths.append(p.name)
        (self.root/'cameras.npz').write_bytes(b'camera provenance')
        self.sample = {'schema_version':1,'complete':True,'index_base':0,'sample_id':'arbitrary_window_name',
                       'scene':'synthetic','context_indices':self.ctx,'target_indices':self.target,
                       'prediction_indices':self.target,'context_image_paths':paths[::2],
                       'encoder_input_image_paths':paths[::2],'prediction_paths':paths,
                       'target_image_paths':paths,'camera_path':'cameras.npz',
                       'context_source_paths':['source']*12,'target_source_paths':['source']*23,
                       'context_source_file_numbers':[i+1 for i in self.ctx],
                       'target_source_file_numbers':[i+1 for i in self.target],
                       'context_camera':'left','target_camera':'left'}
        self.spec = {'schema_version':1,'dataset':'spring','index_base':0,'planned_samples':1,
                     'context_count':12,'target_count':23}
        self.write('dataset.json',self.spec)
        self.write('status.json',{'state':'complete'})
        self.write('samples/arbitrary_window_name/sample.json',self.sample)

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value))

    def load(self, **kwargs):
        return module.DatasetSpringDeblur(module.DatasetSpringDeblurCfg(
            dataset_root=str(self.root),image_height=8,image_width=8,**kwargs))

    def test_explicit_nonzero_indices_and_padding(self):
        x=self.load()[0]
        self.assertEqual(x['context_idx_orig'].tolist(),self.ctx)
        self.assertEqual(x['target_idx_orig'].tolist(),self.target)
        self.assertEqual(tuple(x['vace_video'].shape),(37,3,8,8))
        self.assertTrue(torch.equal(x['vace_video'][-3],x['vace_video'][-1]))
        self.assertAlmostEqual(float(x['vace_video'][12,0,0,0]),0.)
        self.assertAlmostEqual(float(x['vace_video'][-1,0,0,0]),220/255,places=6)
        self.assertEqual(x['vace_video_mask'].tolist(),[0.]*12+[1.]*25)

    def test_prediction_order_must_match_gt(self):
        self.sample['prediction_indices']=list(reversed(self.target))
        self.write('samples/arbitrary_window_name/sample.json',self.sample)
        with self.assertRaisesRegex(ValueError,'Prediction and GT'):
            self.load()

    def test_missing_file_fails_without_resampling(self):
        (self.root/self.sample['prediction_paths'][0]).unlink()
        with self.assertRaises(FileNotFoundError): self.load()

    def test_partial_export_is_explicit(self):
        self.write('status.json',{'state':'running'})
        with self.assertRaises(RuntimeError): self.load()
        self.assertEqual(len(self.load(require_complete=False)),1)

    def test_uncommitted_window_is_not_loaded(self):
        self.sample['complete']=False
        self.write('samples/arbitrary_window_name/sample.json',self.sample)
        with self.assertRaisesRegex(ValueError,'No committed'): self.load()

    def test_source_one_based_mapping_is_checked(self):
        self.sample['target_source_file_numbers']=self.target
        self.write('samples/arbitrary_window_name/sample.json',self.sample)
        with self.assertRaisesRegex(ValueError,'index\\+1'): self.load()

    def test_caption_sample_override(self):
        self.write('captions.json',{'synthetic':'scene caption','arbitrary_window_name':'window caption'})
        self.assertEqual(self.load()[0]['caption'],'window caption')

    def test_asset_cannot_escape_export_root(self):
        self.sample['prediction_paths'][0]='../unexpected.png'
        self.write('samples/arbitrary_window_name/sample.json',self.sample)
        with self.assertRaisesRegex(ValueError,'relative to dataset root'): self.load()

if __name__=='__main__': unittest.main()
