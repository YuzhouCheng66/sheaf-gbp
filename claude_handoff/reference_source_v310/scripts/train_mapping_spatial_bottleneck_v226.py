"""Identity-schema repair only; preserve v225 failure before first update.

The generator stores world_id under metadata. Expose that validated TRAIN ID
to the unchanged feature precompute. Raster contents and training unchanged.
"""
import argparse
from pathlib import Path
import shutil
import unittest
from scripts import train_mapping_spatial_bottleneck_v225 as stage
from scripts import train_mapping_implicit64_v192 as base

PARENT=Path('/home/yuzhou/Desktop/sheaf-gbp/mapping-spatial-bottleneck-v225-20261016')
FILE='scripts/train_mapping_spatial_bottleneck_v226.py'
ORIGINAL_MAKE_BANK=base.make_bank


def expose_identity(item):
    metadata=item['metadata'];world_id=metadata['world_id']
    if metadata['split'].lower()!='train' or not world_id.lower().startswith('train:'):
        raise ValueError('TRAIN-only feature bank')
    if 'world_id' in item and item['world_id']!=world_id:raise ValueError('Conflicting ID')
    return dict(item,world_id=world_id)


def validated_bank(p):
    bank,digest=ORIGINAL_MAKE_BANK(p)
    return [expose_identity(item) for item in bank],digest


class Tests(unittest.TestCase):
    def test_native_generator_contract(self):
        item=base.data.sample_world('TRAIN',0,128,'base');wrapped=expose_identity(item)
        self.assertEqual(wrapped['world_id'],'train:0:base')
        for key in ('walls','private','union','door_mask','masks'):self.assertIs(item[key],wrapped[key])
        self.assertNotIn('world_id',item)
        with self.assertRaises(ValueError):expose_identity(base.data.sample_world('VALIDATION',0,128,'base'))


def prepare(root):
    if root.exists():raise FileExistsError('No overwrite')
    failed=base.read(PARENT/'status.json')
    if failed['state']!='failed_spatial_bottleneck' or failed['step']!=0 or "KeyError: 'world_id'" not in failed['error']:
        raise RuntimeError('Unexpected failure; re-diagnose')
    manifest=base.read(PARENT/'source-manifest.json')
    for rel,digest in manifest['files'].items():
        if base.sha(PARENT/'source'/rel)!=digest:raise RuntimeError('Parent source changed')
    root.mkdir();shutil.copytree(PARENT/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__),root/'source'/FILE);shutil.copy2(PARENT/'preserved-prior.json',root/'preserved-prior.json')
    p=base.read(PARENT/'registered-protocol.json');p.update(version=226,preserved_pretraining_failure=str(PARENT),
        repair='Expose existing metadata.world_id after TRAIN split validation. No geometry, data, model, optimizer, loss, budget or wire change.')
    base.write(root/'registered-protocol.json',p)
    files={v.relative_to(root/'source').as_posix():base.sha(v) for v in (root/'source').rglob('*') if v.is_file()}
    base.write(root/'source-manifest.json',dict(files=files,protocol_sha256=base.sha(root/'registered-protocol.json'),preserved_sha256=base.sha(root/'preserved-prior.json')))
    print(str(root),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(root)
    elif args.launch:stage.FILE=FILE;stage.launch(root)
    else:
        if not unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful():raise RuntimeError('Schema test failed')
        base.make_bank=validated_bank;stage.run(root)
