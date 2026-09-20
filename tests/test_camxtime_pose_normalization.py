"""Regression cases for stationary-center cameras and unchanged moving-camera scaling."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from camxtime_pose_normalization import normalize_camxtime_poses

class CameraNormalizationTests(unittest.TestCase):
    def source(self):
        poses=np.repeat(np.eye(4)[None],33,axis=0)
        poses[:,:3,3]=[10.,20.,30.]
        return poses
    def normalize(self,poses):
        return normalize_camxtime_poses(poses,range(0,33,2),range(33),.001,1000.)
    def test_stationary_camera_does_not_divide_by_zero(self):
        poses,scale,info=self.normalize(self.source())
        self.assertEqual(float(scale),1.)
        self.assertEqual(info['mode'],'stationary_centers_unit_scale')
        torch.testing.assert_close(poses,torch.eye(4).repeat(33,1,1))
    def test_rotation_only_preserves_rotations(self):
        source=self.source()
        for i,theta in enumerate(np.linspace(0,.8,33)):
            c,s=np.cos(theta),np.sin(theta)
            source[i,:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
        poses,scale,_=self.normalize(source)
        self.assertEqual(float(scale),1.)
        self.assertFalse(torch.allclose(poses[-1,:3,:3],torch.eye(3)))
        torch.testing.assert_close(poses[:,:3,3],torch.zeros(33,3),atol=1e-5,rtol=0)
    def test_moving_camera_matches_previous_implementation_exactly(self):
        source=self.source();source[:,0,3]+=np.linspace(0,3,33)
        old=torch.tensor(source@np.diag([1,-1,-1,1]),dtype=torch.float32)
        scale=(old[list(range(2,33,2)),:3,3]-old[0,:3,3]).norm(dim=1).max()
        old[:,:3,3]/=scale;old=torch.linalg.inv(old[:1])@old
        new,new_scale,info=self.normalize(source)
        self.assertTrue(torch.equal(old,new));self.assertTrue(torch.equal(scale,new_scale))
        self.assertEqual(info['mode'],'context_baseline')
    def test_unobserved_translation_is_not_silently_unit_scaled(self):
        source=self.source();source[1,0,3]+=1
        with self.assertRaisesRegex(ValueError,'requested cameras move'):
            self.normalize(source)

if __name__=='__main__':unittest.main()
