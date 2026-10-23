"""Small CPU fixture checks, not a dataset/model/communication benchmark."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.ndimage import label

spec=importlib.util.spec_from_file_location('multiroom_data_v293_under_test',Path(__file__).with_name('mapping_multiroom_data_v293.py'))
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)


class MultiroomDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.small=data.make_fixture('TRAIN',0,128,6,8)

    def test_cross_resolution_world_pose_radio_identity_not_resized_private(self):
        ref=self.small
        for size in (256,512):
            f=data.make_fixture('TRAIN',0,size,6,8)
            for key in ('geometry_sha256','trajectory_sha256','adjacency_sha256'):self.assertEqual(ref['manifest'][key],f['manifest'][key])
            np.testing.assert_array_equal(ref['inputs']['trajectory_m'],f['inputs']['trajectory_m'])
            self.assertEqual(f['inputs']['private'].shape,(8,size,size));self.assertEqual(f['inputs']['private'].dtype,np.int8)
            self.assertLessEqual(f['manifest']['max_sensor_snap_error_m'],f['manifest']['snap_error_bound_m']+1e-12)
            self.assertNotEqual(ref['manifest']['private_sha256'],f['manifest']['private_sha256'])
            self.assertTrue(all(n>0 for n in f['manifest']['observed_cells']))

    def test_connected_unequal_room_geometry_and_doors_all_native_sizes(self):
        for rooms in data.ROOM_COUNTS:
            world=data.world_spec('TRAIN',1,rooms)
            self.assertEqual(len(world['rooms']),rooms)
            areas=[(r['rectangle_m'][2]-r['rectangle_m'][0])*(r['rectangle_m'][3]-r['rectangle_m'][1]) for r in world['rooms']]
            self.assertGreater(np.ptp(areas),.1)
            for size in data.SIZES:
                g=data.rasterize(world,size);components,count=label(g['walls']==0)
                self.assertEqual(count,1);self.assertTrue(g['door_mask'].any());self.assertTrue(np.all(g['walls'][g['door_mask']]==0))
                for room in world['rooms']:
                    x0,y0,x1,y1=room['rectangle_m'];row,col=np.floor(np.array([(y0+y1)/2,(x0+x1)/2])/(20/size)).astype(int)
                    self.assertEqual(components[row,col],1)

    def test_original_sensing_consistent_cumulative_and_metric_range(self):
        f=self.small;own=f['inputs']['private'];truth=f['evaluation']['walls'];trajectory=f['inputs']['trajectory_m']
        self.assertTrue(np.all(own[own>=0]==np.broadcast_to(truth,own.shape)[own>=0]))
        axis=(np.arange(128)+.5)*20/128
        for node in range(8):
            in_range=np.zeros((128,128),bool)
            for x,y in trajectory[:,node]:in_range|=(axis[None]-x)**2+(axis[:,None]-y)**2<=25
            self.assertFalse(np.any((own[node]>=0)&~in_range))
        first,_,_=data.observe_trajectory(truth,trajectory[:1])
        self.assertTrue(np.array_equal(own[first>=0],first[first>=0]))
        self.assertGreater(np.count_nonzero(own<0),0)

    def test_first_hit_occlusion_no_wall_leak(self):
        truth=np.zeros((128,128),np.uint8);truth[:,64]=1
        pose=np.array([[[50.5*20/128,64.5*20/128]]],dtype=np.float64)
        own,_,_=data.observe_trajectory(truth,pose)
        self.assertEqual(own[0,64,64],1)
        self.assertTrue(np.all(own[0,:,65:]==-1))
        self.assertTrue(np.any(own[0,:,:64]==0))

    def test_physical_short_motion_stays_in_assigned_room_or_corridor(self):
        world=data.world_spec('TRAIN',0,20);trajectory,assigned=data.metric_trajectory(world,16)
        self.assertEqual(trajectory.shape,(3,16,2));self.assertGreater(len(set(assigned[assigned>=0])),1)
        self.assertEqual(int(np.sum(assigned==-1)),4)
        for node,rid in enumerate(assigned):
            x0,y0,x1,y1=world['corridor_rectangle_m'] if rid<0 else world['rooms'][rid]['rectangle_m']
            xy=trajectory[:,node];self.assertTrue(np.all((xy[:,0]>x0)&(xy[:,0]<x1)&(xy[:,1]>y0)&(xy[:,1]<y1)))
        self.assertLessEqual(np.linalg.norm(np.diff(trajectory,axis=0),axis=-1).max(),np.sqrt(2)*.1+1e-12)

    def test_static_hops_no_same_round_shortcut_unreached_is_unknown(self):
        own=np.full((4,2,2),-1,np.int8);own[0,0,0]=0;own[2,1,1]=1;own[3,0,1]=0
        edges=((0,1),(1,0),(1,2),(2,1))
        one=data.hop_targets(own,edges,1);two=data.hop_targets(own,edges,2)
        self.assertEqual(one[0,1,1],-1);self.assertEqual(two[0,1,1],1)
        self.assertEqual(two[0,0,1],-1);np.testing.assert_array_equal(two[3],own[3])
        f=self.small
        for k in data.HOPS:
            reach=f['evaluation']['reachable_sources'][k]
            for node in range(8):
                np.testing.assert_array_equal(f['evaluation']['targets'][k][node],f['inputs']['private'][reach[node]].max(0))
        np.testing.assert_array_equal(f['evaluation']['targets'][0],f['inputs']['private'])

    def test_radius_from_metric_poses_keeps_cycles_and_disconnections(self):
        f=self.small;p=f['inputs']['positions_m'];expected=np.linalg.norm(p[:,None]-p[None,:],axis=-1)<=8.;np.fill_diagonal(expected,False)
        np.testing.assert_array_equal(expected,f['inputs']['adjacency'])
        pos=np.array([[0.,0.],[1.,0.],[.5,1.],[19.,19.]])
        graph=data.radius_adjacency(pos,2.);diagnostic=data.graph_diagnostics(graph)
        self.assertEqual(diagnostic['components'],2);self.assertEqual(diagnostic['cycle_rank'],1)
        self.assertIsNone(diagnostic['diameter']);self.assertTrue(diagnostic['disconnected_or_cyclic_retained'])

    def test_local_permissions_and_no_alias_to_evaluator(self):
        f=self.small;local=data.per_agent_inputs(f,0)
        self.assertEqual(set(local),{'private','position_m','trajectory_m','neighbor_ids','public_frame'})
        self.assertEqual(local['private'].ndim,2);self.assertNotIn('walls',f['inputs']);self.assertNotIn('targets',f['inputs'])
        before=f['inputs']['private'].copy();local['private'][:]=1;local['public_frame']['shape'][0]=1
        np.testing.assert_array_equal(f['inputs']['private'],before);self.assertEqual(f['inputs']['public_frame']['shape'],[128,128])
        np.testing.assert_array_equal(f['evaluation']['targets'][0],before)
        self.assertFalse(f['manifest']['model_executed']);self.assertFalse(f['manifest']['TEST_opened'])

    def test_determinism_split_boundaries_and_invalid_inputs(self):
        spec=data.world_spec('TRAIN',0,6);self.assertEqual(spec,data.world_spec('TRAIN',0,6))
        self.assertNotEqual(spec['world_id'],data.world_spec('VALIDATION',0,6)['world_id'])
        self.assertNotEqual(data._json_hash(spec),data._json_hash(data.world_spec('VALIDATION',0,6)))
        for args in [('TEST',0,6),('TRAIN',True,6),('TRAIN',-1,6),('TRAIN',0,7)]:
            with self.assertRaises(ValueError):data.world_spec(*args)
        with self.assertRaises(ValueError):data.make_fixture('TRAIN',0,64)
        with self.assertRaises(ValueError):data.make_fixture('TRAIN',0,128,6,2)
        with self.assertRaises(ValueError):data.per_agent_inputs(self.small,8)
        invalid=copy.deepcopy(self.small);invalid['inputs']['walls']=invalid['evaluation']['walls']
        with self.assertRaises(ValueError):data.per_agent_inputs(invalid,0)


if __name__=='__main__':unittest.main()
