"""Size-qualified original-sensor cache; explicit preparation, read-only get.

v293 is deliberately unchanged, including its inherited transitive imports.
This module imports no model/trainer, executes no neural code, and stores no
learned factors or targets. Evaluation-only doors never enter input dictionaries.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from scripts import mapping_multiroom_data_v293 as data

SIZES=(128,256,512)
ROOMS=(6,12,20)
AGENTS=(8,16)
SEED=306101
GENERATOR_SHA='6e4bc0766e22b3b720332fde59bfcf84c2284d27f1067357835c31df2cfd5445'
INPUT_ARRAYS=('private','positions_m','trajectory_m','adjacency')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def json_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value,*,replace=False):
    path=Path(path)
    if not replace and path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
    tmp.replace(path)


def checked_spec(spec):
    require(type(spec) is dict and set(spec)=={'split','index','rooms','agents','size'},'Exact scene specification required')
    require(spec['split'] in ('TRAIN','VALIDATION'),'TEST is closed')
    for name,allowed in (('rooms',ROOMS),('agents',AGENTS),('size',SIZES)):
        require(type(spec[name]) is int and spec[name] in allowed,'Unregistered '+name)
    limit=256 if spec['split']=='TRAIN' else 8
    require(type(spec['index']) is int and 0<=spec['index']<limit,'Unregistered world index')
    return dict(spec)


def scene_key(spec):
    s=checked_spec(spec)
    return f"{s['split']}-H{s['size']:03d}-r{s['rooms']:02d}-n{s['agents']:02d}-i{s['index']:03d}"


def old_key(spec):
    s=checked_spec(spec)
    return f"{s['split']}-r{s['rooms']:02d}-n{s['agents']:02d}-i{s['index']:03d}"


def draw_plan(updates,seed=SEED):
    require(type(updates) is int and updates>=0 and type(seed) is int and seed==SEED,'Fixed nonnegative budget and seed306101 required')
    rng=np.random.default_rng(seed);result=[]
    while len(result)<updates:
        block=[]
        for rooms in ROOMS:
            index=int(rng.integers(256))
            block.extend(dict(split='TRAIN',index=index,rooms=rooms,agents=n,size=h) for h in SIZES for n in AGENTS)
        result.extend(block[int(i)] for i in rng.permutation(18))
    return result[:updates]


def physical_geometry_hash(world):
    """Do not hash world/split IDs when checking physical cross-split copies."""
    fields=dict(extent_m=world['extent_m'],corridor_rectangle_m=world['corridor_rectangle_m'],
        room_rectangles_m=[r['rectangle_m'] for r in world['rooms']],door_rectangles_m=world['door_rectangles_m'])
    require(all(np.isfinite(np.asarray(v,dtype=float)).all() for v in fields.values()),'Finite physical geometry required')
    return json_hash(fields)


def select_forest(positions_m,adjacency,radius_m=8.):
    """Pure NumPy reproduction of v295 public sorted minimum-root BFS."""
    p=np.asarray(positions_m);a=np.asarray(adjacency);n=len(p)
    require(p.shape==(n,2) and 1<=n<=16 and np.issubdtype(p.dtype,np.floating) and np.isfinite(p).all(),'Finite metric positions required')
    require(type(radius_m) in (float,int) and np.isfinite(radius_m) and radius_m>0,'Positive public radius required')
    physical=((p[:,None]-p[None,:])**2).sum(-1)<=float(radius_m)**2;np.fill_diagonal(physical,False)
    require(a.dtype==bool and a.shape==(n,n) and np.array_equal(a,physical),'Physical radius adjacency mismatch')
    visited=set();edges=[];components=[]
    for root in range(n):
        if root in visited:continue
        queue=deque([root]);visited.add(root);component=[]
        while queue:
            i=queue.popleft();component.append(i)
            for j in np.flatnonzero(a[i]).tolist():
                if j not in visited:visited.add(j);queue.append(j);edges.append((min(i,j),max(i,j),1.))
        components.append(sorted(component))
    selected=np.zeros_like(a)
    for i,j,_ in edges:selected[i,j]=selected[j,i]=True
    return dict(edges=sorted(edges),physical_adjacency=a.copy(),selected_adjacency=selected,components=components,
        component_roots=[c[0] for c in components],offline_public_schedule=True,distributed_forest_construction=False,
        schedule_setup_measured=False,uses_map_or_target=False)


def forest_targets(inputs,rounds):
    require(type(rounds) is int and 0<=rounds<=16,'Rounds0..16 required')
    require(set(inputs)==set(INPUT_ARRAYS)|{'public_frame'},'Source inputs only, not evaluation')
    own=np.asarray(inputs['private'])
    require(own.ndim==3 and min(own.shape)>0 and own.dtype==np.int8 and np.isin(own,[-1,0,1]).all(),'Original ternary sensors required')
    n=len(own);require(np.asarray(inputs['positions_m']).shape==(n,2),'Source and public graph node counts differ')
    require(not np.any(np.any(own==0,axis=0)&np.any(own==1,axis=0)),'Conflicting original observations')
    forest=select_forest(inputs['positions_m'],inputs['adjacency'],inputs['public_frame']['communication_radius_m'])
    reached=np.eye(n,dtype=bool)
    for _ in range(rounds):
        previous=reached;reached=previous.copy()
        for i,j,_ in forest['edges']:reached[i]|=previous[j];reached[j]|=previous[i]
    return np.stack([own[s].max(0) for s in reached]),reached


def expected_frame(size):
    return dict(origin_m=[0.,0.],extent_m=[20.,20.],shape=[size,size],cell_size_m=20./size,
        coordinate_convention='pose=(x,y); raster=[y,x]',sensor_range_m=5.,sensor_beams=256,
        communication_radius_m=8.,static_snapshot=True,radio_occlusion=False)


def checked_arrays(spec,inputs,evaluation,manifest):
    s=checked_spec(spec);n=s['agents'];h=s['size']
    require(set(inputs)==set(INPUT_ARRAYS)|{'public_frame'} and set(evaluation)=={'door_mask'},'Strict input/evaluation separation required')
    own=inputs['private'];door=evaluation['door_mask']
    require(own.shape==(n,h,h) and own.dtype==np.int8 and np.isin(own,[-1,0,1]).all(),'Native int8 original sensors required')
    require(door.shape==(h,h) and door.dtype==bool and not np.any((own==1)&door[None]),'Observed doors must be free')
    require((own>=0).any((1,2)).all() and not np.any(np.any(own==0,axis=0)&np.any(own==1,axis=0)),'Empty or conflicting original sources')
    p=inputs['positions_m'];t=inputs['trajectory_m']
    require(p.dtype==np.float64 and t.dtype==np.float64 and t.shape==(3,n,2) and p.shape==(n,2)
            and np.isfinite(t).all() and (t>=0).all() and (t<20).all() and np.array_equal(p,t[-1]),'Exact metric public poses required')
    require(inputs['public_frame']==expected_frame(h),'Public frame/sensor changed')
    world=data.world_spec(s['split'],s['index'],s['rooms'])
    require(all(manifest[k]==s[k] for k in ('split','index','rooms','agents','size')) and manifest['world_id']==world['world_id']
        and manifest['geometry_sha256']==data._json_hash(world),'Original geometry identity changed')
    for name,array in (('private_sha256',own),('trajectory_sha256',t),('adjacency_sha256',inputs['adjacency'])):
        require(manifest[name]==data.array_hash(array),'Original '+name+' mismatch')
    require(manifest['source_sha256']==[data.array_hash(v) for v in own],'Original source hashes differ')
    return select_forest(p,inputs['adjacency']),physical_geometry_hash(world)


class NativeBank:
    """Explicit old128 references and new native caches; no implicit get writes."""
    @classmethod
    def create(cls,root,old_bank,old_manifest_pins):
        root=Path(root).resolve();old=Path(old_bank).resolve()
        require(not root.exists() and not root.is_relative_to(old) and not old.is_relative_to(root),'Fresh external bank directory required')
        require(set(old_manifest_pins)=={'TRAIN','VALIDATION'},'Both explicit original manifest pins required')
        for split,digest in old_manifest_pins.items():require(sha(old/(split+'-manifest.json'))==digest,'Original manifest pin mismatch')
        require(sha(data.__file__)==GENERATOR_SHA,'Original native sensor generator changed')
        root.mkdir(parents=True)
        config=dict(version=306,old_bank=str(old),old_manifest_pins=dict(old_manifest_pins),generator_sha256=GENERATOR_SHA,
            targets_cached=False,learned_factors_cached=False,TEST_opened=False,old_sensors_regenerated=False,
            source_inputs_are_central_simulation_storage_not_global_agent_authority=True)
        write(root/'config.json',config);write(root/'manifest.json',dict(version=306,config_sha256=sha(root/'config.json'),cases={}))
        return cls(root)

    def __init__(self,root,expected_manifest_sha256=None):
        self.root=Path(root).resolve();self.expected_manifest_sha256=expected_manifest_sha256
        self._manifest()

    def _manifest(self):
        path=self.root/'manifest.json'
        if self.expected_manifest_sha256 is not None:require(sha(path)==self.expected_manifest_sha256,'Pinned native manifest changed')
        m=read(path);c=read(self.root/'config.json')
        require(m['version']==306 and c['version']==306 and m['config_sha256']==sha(self.root/'config.json')
            and c['TEST_opened'] is False and c['generator_sha256']==GENERATOR_SHA and sha(data.__file__)==GENERATOR_SHA,'Bank configuration/source changed')
        for key,entry in m['cases'].items():require(scene_key(entry['spec'])==key,'Invalid manifest scene identity')
        return m,c

    def _old(self,spec,config):
        old=Path(config['old_bank']);split=spec['split'];mp=old/(split+'-manifest.json')
        require(sha(mp)==config['old_manifest_pins'][split],'Original manifest changed')
        entry=read(mp)['scenes'][old_key(spec)];path=old/(old_key(spec)+'.npz');meta_path=path.with_suffix('.json');meta=read(meta_path)
        require(entry['spec']==spec==meta['spec'] and entry['manifest']==meta['manifest']
            and sha(path)==entry['cache_sha256']==meta['cache_sha256'],'Original128 cache mismatch')
        with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        require(set(arrays)==set(INPUT_ARRAYS)|{'door_mask'} and {k:data.array_hash(v) for k,v in arrays.items()}==meta['arrays_sha256'],'Original128 arrays mismatch')
        inputs={k:arrays[k] for k in INPUT_ARRAYS};inputs['public_frame']=meta['public_frame']
        return inputs,{'door_mask':arrays['door_mask']},meta['manifest'],dict(cache_sha256=sha(path),metadata_sha256=sha(meta_path))

    def _commit(self,spec,inputs,evaluation,manifest,kind,origin=None):
        m,c=self._manifest();key=scene_key(spec);folder=self.root/key
        require(self.expected_manifest_sha256 is None,'Pinned read-only bank cannot prepare')
        require(key not in m['cases'] and not folder.exists(),'Existing/partial cache cannot be overwritten')
        forest,physical=checked_arrays(spec,inputs,evaluation,manifest)
        # AllN/size aliases may share geometry, but not TRAIN/VALIDATION.
        for entry in m['cases'].values():
            require(not (entry['physical_geometry_sha256']==physical and entry['spec']['split']!=spec['split']),'Cross-split duplicate physical geometry')
            if entry['world_id']==manifest['world_id'] and entry['spec']['agents']==spec['agents']:
                require(entry['geometry_sha256']==manifest['geometry_sha256'] and entry['trajectory_sha256']==manifest['trajectory_sha256']
                    and entry['adjacency_sha256']==manifest['adjacency_sha256'],'Cross-size physical source identity changed')
        folder.mkdir()
        try:
            hashes={}
            if kind=='native':
                np.savez_compressed(folder/'inputs.npz',**{k:inputs[k] for k in INPUT_ARRAYS})
                np.savez_compressed(folder/'evaluation.npz',**evaluation)
                hashes={name:sha(folder/name) for name in ('inputs.npz','evaluation.npz')}
            meta=dict(spec=spec,kind=kind,public_frame=inputs['public_frame'],manifest=manifest,origin=origin,files=hashes,
                input_array_sha256={k:data.array_hash(inputs[k]) for k in INPUT_ARRAYS},door_sha256=data.array_hash(evaluation['door_mask']),
                forest_sha256=json_hash(forest['edges']),physical_geometry_sha256=physical,targets_cached=False,
                inherited_manifest_targets_are_all_radius_references_not_supervision=True,
                supervision_rule='forest_targets(inputs, actual_synchronous_rounds) from original private maps only')
            write(folder/'metadata.json',meta)
            m['cases'][key]=dict(spec=spec,metadata_sha256=sha(folder/'metadata.json'),physical_geometry_sha256=physical,
                world_id=manifest['world_id'],geometry_sha256=manifest['geometry_sha256'],trajectory_sha256=manifest['trajectory_sha256'],adjacency_sha256=manifest['adjacency_sha256'])
            write(self.root/'manifest.json',m,replace=True)
        except BaseException:
            write(folder/'failure.json',dict(complete=False,error=traceback.format_exc()));raise
        return key

    def register128(self,spec):
        spec=checked_spec(spec);require(spec['size']==128,'Only existing native128 reference registration')
        _,c=self._manifest();i,e,m,origin=self._old(spec,c)
        return self._commit(spec,i,e,m,'old128',origin)

    def prepare(self,spec):
        spec=checked_spec(spec);require(spec['size'] in (256,512),'Native128 must reuse immutable old cache')
        m,_=self._manifest();key=scene_key(spec)
        require(self.expected_manifest_sha256 is None and key not in m['cases'] and not (self.root/key).exists(),'Existing/partial/pinned cache cannot prepare')
        fixture=data.make_fixture(**spec)
        return self._commit(spec,fixture['inputs'],{'door_mask':fixture['evaluation']['door_mask']},fixture['manifest'],'native')

    def get(self,spec):
        spec=checked_spec(spec);m,c=self._manifest();key=scene_key(spec)
        if key not in m['cases']:raise FileNotFoundError('No admitted cache: '+key)
        folder=self.root/key;entry=m['cases'][key]
        require(folder.is_dir() and not (folder/'failure.json').exists() and sha(folder/'metadata.json')==entry['metadata_sha256'],'Incomplete/tampered native cache')
        meta=read(folder/'metadata.json');require(meta['spec']==spec,'Wrong stored scene')
        if meta['kind']=='old128':
            require(spec['size']==128,'Invalid old-size reference');inputs,evaluation,manifest,origin=self._old(spec,c)
            require(origin==meta['origin'],'Original source bytes changed')
        else:
            require(meta['kind']=='native' and spec['size'] in (256,512) and set(meta['files'])=={'inputs.npz','evaluation.npz'},'Unknown cache kind')
            for name,digest in meta['files'].items():require(sha(folder/name)==digest,'Native array file changed')
            with np.load(folder/'inputs.npz',allow_pickle=False) as saved:inputs={k:saved[k].copy() for k in saved.files}
            with np.load(folder/'evaluation.npz',allow_pickle=False) as saved:evaluation={k:saved[k].copy() for k in saved.files}
            inputs['public_frame']=meta['public_frame'];manifest=meta['manifest']
        require(manifest==meta['manifest'] and {k:data.array_hash(inputs[k]) for k in INPUT_ARRAYS}==meta['input_array_sha256']
            and data.array_hash(evaluation['door_mask'])==meta['door_sha256'],'Stored source/evaluation hashes changed')
        forest,physical=checked_arrays(spec,inputs,evaluation,manifest)
        require(physical==entry['physical_geometry_sha256']==meta['physical_geometry_sha256'] and json_hash(forest['edges'])==meta['forest_sha256'],'Geometry/schedule changed')
        return dict(inputs=inputs,evaluation=evaluation,metadata=meta,forest=forest)
