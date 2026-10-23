"""Temporary CPU fixtures only; no remote/full-bank/model evaluation."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np

spec=importlib.util.spec_from_file_location('native_bank306_test',Path(__file__).with_name('mapping_native_bank_v306.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def scene(split='TRAIN',size=128,index=0):
    return dict(split=split,index=index,rooms=6,agents=8,size=size)


def old_fixture(folder,fixtures):
    folder.mkdir();pins={}
    for split in ('TRAIN','VALIDATION'):
        s=scene(split);f=fixtures[split];key=m.old_key(s)
        arrays={k:f['inputs'][k] for k in m.INPUT_ARRAYS};arrays['door_mask']=f['evaluation']['door_mask']
        file=folder/(key+'.npz');np.savez_compressed(file,**arrays)
        meta=dict(spec=s,manifest=f['manifest'],public_frame=f['inputs']['public_frame'],cache_sha256=m.sha(file),
            arrays_sha256={k:m.data.array_hash(v) for k,v in arrays.items()})
        m.write(file.with_suffix('.json'),meta)
        manifest=dict(scenes={key:dict(spec=s,manifest=f['manifest'],cache_sha256=m.sha(file))})
        path=folder/(split+'-manifest.json');m.write(path,manifest);pins[split]=m.sha(path)
    return pins


class NativeBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures={split:m.data.make_fixture(**scene(split)) for split in ('TRAIN','VALIDATION')}

    def make_bank(self,tmp):
        old=Path(tmp)/'old';pins=old_fixture(old,self.fixtures)
        bank=m.NativeBank.create(Path(tmp)/'new',old,pins)
        return bank,old,pins

    def test_size_qualified_keys_and_closed_schema(self):
        self.assertEqual(len({m.scene_key(scene(size=h)) for h in m.SIZES}),3)
        for s in (scene('TEST'),scene(index=256),scene('VALIDATION',index=8),scene(size=64),dict(scene(),index=True),dict(scene(),mask=1)):
            with self.assertRaises(ValueError):m.scene_key(s)
        with tempfile.TemporaryDirectory() as tmp:
            bank,_,_=self.make_bank(tmp)
            with self.assertRaises(ValueError):bank.prepare(scene())
            with self.assertRaises(ValueError):bank.register128(scene(size=256))

    def test_fixed_prefix_macrobalance_and_global_rng_untouched(self):
        state=np.random.get_state();a=m.draw_plan(55)
        self.assertEqual(a[:19],m.draw_plan(19));self.assertEqual(a,m.draw_plan(55))
        after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
        for start in (0,18,36):
            block=a[start:start+18]
            self.assertEqual({(v['size'],v['rooms'],v['agents']) for v in block},{(h,r,n) for h in m.SIZES for r in m.ROOMS for n in m.AGENTS})
            for r in m.ROOMS:self.assertEqual(len({v['index'] for v in block if v['rooms']==r}),1)
        for u in (-1,True,1.5):
            with self.assertRaises(ValueError):m.draw_plan(u)
        with self.assertRaises(ValueError):m.draw_plan(1,306102)

    def test_old128_reference_is_read_only_no_sensor_regeneration_no_eval_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,old,_=self.make_bank(tmp);before={f.name:m.sha(f) for f in old.iterdir()}
            with mock.patch.object(m.data,'make_fixture',side_effect=AssertionError('No generation')):
                bank.register128(scene());a=bank.get(scene())
            self.assertEqual(before,{f.name:m.sha(f) for f in old.iterdir()})
            np.testing.assert_array_equal(a['inputs']['private'],self.fixtures['TRAIN']['inputs']['private'])
            self.assertEqual(set(a['inputs']),set(m.INPUT_ARRAYS)|{'public_frame'});self.assertEqual(set(a['evaluation']),{'door_mask'})
            self.assertNotIn('targets',a);self.assertFalse((bank.root/m.scene_key(scene())/'inputs.npz').exists())
            a['inputs']['private'][:]=1;a['evaluation']['door_mask'][:]=False
            self.assertTrue(np.any(bank.get(scene())['inputs']['private']==-1))
            pin=m.sha(bank.root/'manifest.json');readonly=m.NativeBank(bank.root,pin)
            with self.assertRaises(ValueError):readonly.register128(scene('VALIDATION'))

    def test_true_native_prepare_and_cross_size_physical_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,_,_=self.make_bank(tmp);bank.register128(scene());base=bank.get(scene())
            for size in (256,512):
                s=scene(size=size);bank.prepare(s);entry=bank.get(s)
                self.assertEqual(entry['inputs']['private'].shape,(8,size,size))
                self.assertEqual(entry['metadata']['physical_geometry_sha256'],base['metadata']['physical_geometry_sha256'])
                for k in ('geometry_sha256','trajectory_sha256','adjacency_sha256'):
                    self.assertEqual(entry['metadata']['manifest'][k],base['metadata']['manifest'][k])
                folder=bank.root/m.scene_key(s)
                with np.load(folder/'inputs.npz') as saved:self.assertEqual(set(saved.files),set(m.INPUT_ARRAYS))
                with np.load(folder/'evaluation.npz') as saved:self.assertEqual(saved.files,['door_mask'])
                expected=m.data.rasterize(m.data.world_spec('TRAIN',0,6),size)['walls']
                own=entry['inputs']['private'];observed=own>=0
                np.testing.assert_array_equal(own[observed],np.broadcast_to(expected,own.shape)[observed])
                repeated=np.repeat(np.repeat(base['inputs']['private'],size//128,1),size//128,2)
                self.assertFalse(np.array_equal(own,repeated))
            self.assertEqual(len(m.read(bank.root/'manifest.json')['cases']),3)

    def test_missing_partial_existing_no_implicit_get_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,old,pins=self.make_bank(tmp);s=scene(size=256)
            with mock.patch.object(m.data,'make_fixture',side_effect=AssertionError('No generation')):
                with self.assertRaises(FileNotFoundError):bank.get(s)
                (bank.root/m.scene_key(s)).mkdir()
                with self.assertRaises(ValueError):bank.prepare(s)
            with self.assertRaises(ValueError):m.NativeBank.create(bank.root,old,pins)
            with self.assertRaises(ValueError):m.NativeBank.create(old/'nested',old,pins)
            bank.register128(scene())
            with self.assertRaises(ValueError):bank.register128(scene())

    def test_manifest_cache_and_metadata_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,old,_=self.make_bank(tmp);bank.register128(scene());path=old/(m.old_key(scene())+'.npz')
            with path.open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):bank.get(scene())
        with tempfile.TemporaryDirectory() as tmp:
            bank,_,_=self.make_bank(tmp);bank.prepare(scene(size=256));folder=bank.root/m.scene_key(scene(size=256))
            with (folder/'evaluation.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):bank.get(scene(size=256))
        with tempfile.TemporaryDirectory() as tmp:
            bank,_,_=self.make_bank(tmp);bank.register128(scene());folder=bank.root/m.scene_key(scene())
            meta=m.read(folder/'metadata.json');meta['public_frame']['shape']=[512,512]
            m.write(folder/'metadata.json',meta,replace=True)
            with self.assertRaises(ValueError):bank.get(scene())
            manifest=m.read(bank.root/'manifest.json');entry=manifest['cases'].pop(m.scene_key(scene()));manifest['cases']['../bad']=entry
            m.write(bank.root/'manifest.json',manifest,replace=True)
            with self.assertRaises(ValueError):bank.get(scene())

    def test_bad_old_manifest_pin_rejects_before_target_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old';pins=old_fixture(old,self.fixtures);pins['TRAIN']='0'*64;new=Path(tmp)/'new'
            with self.assertRaises(ValueError):m.NativeBank.create(new,old,pins)
            self.assertFalse(new.exists())

    def test_physical_hash_not_identity_and_split_collision(self):
        w=m.data.world_spec('TRAIN',0,6);same=copy.deepcopy(w);same.update(world_id='other',split='VALIDATION',index=8)
        self.assertEqual(m.physical_geometry_hash(w),m.physical_geometry_hash(same))
        same['door_rectangles_m'][0][0]+=.01
        self.assertNotEqual(m.physical_geometry_hash(w),m.physical_geometry_hash(same))
        with tempfile.TemporaryDirectory() as tmp:
            bank,_,_=self.make_bank(tmp);bank.register128(scene());s=scene('VALIDATION');f=self.fixtures['VALIDATION']
            with mock.patch.object(m,'physical_geometry_hash',return_value=bank.get(scene())['metadata']['physical_geometry_sha256']):
                with self.assertRaises(ValueError):bank._commit(s,f['inputs'],{'door_mask':f['evaluation']['door_mask']},f['manifest'],'old128',{})

    def test_forest_exact_v295_and_not_radius_shortcuts_with_isolates(self):
        from scripts import mapping_multiroom_forest_v295 as reference
        positions=np.array([[0.,0.],[1.,0.],[.5,.5],[10.,10.]],dtype=np.float64)
        adjacency=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)<=2.;np.fill_diagonal(adjacency,False)
        a=m.select_forest(positions,adjacency,2.);b=reference.select_forest(positions,2.,adjacency)
        self.assertEqual(a['edges'],b['edges']);self.assertEqual(a['components'],b['components'])
        own=np.full((4,2,2),-1,np.int8)
        for node in range(4):own[node].flat[node]=node%2
        inputs=dict(private=own,positions_m=positions,trajectory_m=np.stack([positions]*3),adjacency=adjacency,public_frame={'communication_radius_m':2.})
        for k in (0,1,2,4,8,16):
            targets,reached=m.forest_targets(inputs,k);expected=reference.reachability(b['edges'],4,k)
            np.testing.assert_array_equal(reached,expected)
            np.testing.assert_array_equal(targets,np.stack([own[s].max(0) for s in expected]))
        one,reach=m.forest_targets(inputs,1)
        self.assertFalse(reach[1,2]);self.assertEqual(one[1].flat[2],-1)
        self.assertTrue(adjacency[1,2]);np.testing.assert_array_equal(one[3],own[3])
        bad=dict(inputs,private=own.copy());bad['private'][1,0,0]=1
        with self.assertRaises(ValueError):m.forest_targets(bad,2)
        with self.assertRaises(ValueError):m.forest_targets(dict(inputs,door_mask=np.zeros((2,2),bool)),2)
        with self.assertRaises(ValueError):m.forest_targets(dict(inputs,positions_m=positions[:3]),2)
        with self.assertRaises(ValueError):m.select_forest(positions,np.zeros_like(adjacency),2.)


if __name__=='__main__':unittest.main()
