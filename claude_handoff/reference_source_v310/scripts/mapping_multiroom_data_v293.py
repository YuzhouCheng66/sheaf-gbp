"""CPU-only metric multiroom sensor fixture; no model or artifact generation.

Only ``inputs`` may reach the inference harness; each agent receives only its
``per_agent_inputs`` slice. Hidden floorplan, room IDs and reachable unions are
in ``evaluation``. Static hop labels assume reliable simultaneous sends on all
radius edges; they do not assert that a compressed protocol retains the facts.
"""
import copy
import hashlib
import json
from numbers import Integral

import numpy as np

from scripts.prepare_houseexpo_pilot_v121 import make_rays, raycast_observe, radius_adjacency
from scripts.mapping_hop_targets_v277 import reachable_sources, hop_targets

SIZES=(128,256,512)
ROOM_COUNTS=(6,12,20)
AGENT_COUNTS=(8,16)
HOPS=(0,1,2,4,8)
EXTENT=20.
SENSOR_RANGE=5.
RADIO_RADIUS=8.
INPUT_FIELDS=('private','positions_m','trajectory_m','adjacency','public_frame')


def _integer(value,name,allowed=None):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<0:
        raise ValueError(name+' must be a nonnegative integer')
    value=int(value)
    if allowed is not None and value not in allowed:raise ValueError('Unsupported '+name)
    return value


def _seed(text):return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')


def _json_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256(json.dumps([str(a.dtype),list(a.shape)],separators=(',',':')).encode());h.update(a.tobytes());return h.hexdigest()


def world_spec(split,index,rooms=6):
    """Fresh continuous geometry; no resolution, agent count or sensor fields."""
    split=str(split).upper()
    if split not in ('TRAIN','VALIDATION'):raise ValueError('Only TRAIN/VALIDATION, never TEST')
    index=_integer(index,'world index');rooms=_integer(rooms,'rooms',ROOM_COUNTS)
    if index>=1_000_000_000:raise ValueError('World index outside registered range')
    identity=f'v293:{split}:{index}:corridor-spine:{rooms}'
    rng=np.random.default_rng(_seed(identity));count=rooms//2
    corridor=[9.2,.6,10.8,19.4];regions=[];doors=[]
    for side in ('left','right'):
        # Strict positive minimum leaves >1m interior height even for20 rooms.
        extra=rng.uniform(.5,1.5,count);heights=1.4+(18.8-1.4*count)*extra/extra.sum()
        edges=np.r_[.6,.6+np.cumsum(heights)];edges[-1]=19.4
        for k in range(count):
            low,high=float(edges[k]+.15),float(edges[k+1]-.15)
            width=float(rng.uniform(4.4,8.1));center=float(rng.uniform(low+.45,high-.45))
            if side=='left':rect=[8.9-width,low,8.9,high];door=[8.9,center-.40,9.2,center+.40]
            else:rect=[11.1,low,11.1+width,high];door=[10.8,center-.40,11.1,center+.40]
            regions.append(dict(room_id=len(regions),side=side,rectangle_m=rect));doors.append(door)
    return dict(version=293,family='corridor-spine',split=split,index=index,world_id=identity,extent_m=[20.,20.],
                nominal_rooms=rooms,corridor_rectangle_m=corridor,rooms=regions,door_rectangles_m=doors,
                occupancy='Union of corridor, room interiors and door connectors is free; complement is occupied',
                coordinate_convention='metric(x,y); raster[row=y,column=x], cell-centre inclusion')


def _rect_mask(rect,size):
    axis=(np.arange(size,dtype=np.float64)+.5)*(EXTENT/size);x0,y0,x1,y1=rect
    return ((axis[None]>=x0)&(axis[None]<x1)&(axis[:,None]>=y0)&(axis[:,None]<y1))


def rasterize(spec,size):
    size=_integer(size,'size',SIZES)
    if spec.get('version')!=293 or spec.get('nominal_rooms') not in ROOM_COUNTS:raise ValueError('v293 geometry required')
    room_masks=np.stack([_rect_mask(r['rectangle_m'],size) for r in spec['rooms']])
    corridor=_rect_mask(spec['corridor_rectangle_m'],size)
    doors=np.zeros((size,size),bool)
    for rect in spec['door_rectangles_m']:doors|=_rect_mask(rect,size)
    walls=(~(room_masks.any(0)|corridor|doors)).astype(np.uint8)
    return dict(walls=walls,room_masks=room_masks,corridor_mask=corridor,door_mask=doors)


def metric_trajectory(spec,agents=8):
    """Three short poses within one sampled room/corridor, no GT path in inputs.

    Geometry-assisted feasible placement is an offline sampler, not an agent
    exploration policy. Sources are dispersed before any sensor/model result.
    """
    agents=_integer(agents,'agents',AGENT_COUNTS)
    rng=np.random.default_rng(_seed(spec['world_id']+f':poses:{agents}'))
    corridor_agents=agents//4;room_agents=agents-corridor_agents
    order=rng.permutation(len(spec['rooms']));assignment=[];centers=[];deltas=[]
    for node in range(room_agents):
        room_id=int(order[node%len(order)]);rect=spec['rooms'][room_id]['rectangle_m'];x0,y0,x1,y1=rect
        center=[rng.uniform(x0+.45,x1-.45),rng.uniform(y0+.40,y1-.40)]
        assignment.append(room_id);centers.append(center);deltas.append(rng.uniform(-.10,.10,2))
    for node in range(corridor_agents):
        center=[10.+rng.uniform(-.25,.25),.9+(node+.5)*(18.2/corridor_agents)+rng.uniform(-.15,.15)]
        assignment.append(-1);centers.append(center);deltas.append(rng.uniform(-.10,.10,2))
    # Agent IDs do not reveal room versus corridor placement.
    perm=rng.permutation(agents);centers=np.asarray(centers,dtype=np.float64)[perm];deltas=np.asarray(deltas,dtype=np.float64)[perm]
    trajectory=centers[None]+np.array([-1.,0.,1.])[:,None,None]*deltas[None]
    return trajectory,np.asarray(assignment,dtype=np.int64)[perm]


def observe_trajectory(walls,trajectory_m):
    """First-hit DDA rays from snapped cell centres, restricted to true5m.

    Public continuous poses are raster-snapped for the proven DDA routine.
    A per-pose metric distance mask removes any measured cell centre farther
    than5m from the actual continuous pose. No resizing or behind-hit filling.
    """
    walls=np.asarray(walls);trajectory=np.asarray(trajectory_m)
    if (walls.ndim!=2 or walls.shape[0]!=walls.shape[1] or walls.shape[0] not in SIZES
            or not np.isin(walls,(0,1)).all() or trajectory.ndim!=3 or trajectory.shape[-1]!=2
            or len(trajectory)<1 or trajectory.shape[1]<1 or not np.isfinite(trajectory).all()
            or (trajectory<0).any() or (trajectory>=EXTENT).any()):raise ValueError('Metric trajectory and native binary raster required')
    size=walls.shape[0];cell=EXTENT/size
    rc=np.floor(trajectory[...,::-1]/cell).astype(np.int64)
    if np.any(walls[rc[...,0],rc[...,1]]!=0):raise ValueError('Sensor pose must be free; no resampling')
    rays=make_rays(SENSOR_RANGE/cell,256)
    private=np.full((trajectory.shape[1],size,size),-1,np.int8)
    axis=(np.arange(size,dtype=np.float64)+.5)*cell
    for step in range(len(trajectory)):
        observed=np.full_like(private,-1);raycast_observe(walls,rc[step],observed,rays)
        for node,(x,y) in enumerate(trajectory[step]):
            inside=(axis[None]-x)**2+(axis[:,None]-y)**2<=SENSOR_RANGE**2
            observed[node,~inside]=-1
        private=np.maximum(private,observed)
    if not (private>=0).any(axis=(1,2)).all():raise AssertionError('All sources must have actual sensor evidence')
    snapped=(rc[...,::-1].astype(np.float64)+.5)*cell
    return private,rc,snapped


def graph_diagnostics(adjacency):
    adjacency=np.asarray(adjacency)
    if adjacency.ndim!=2 or adjacency.shape[0]!=adjacency.shape[1] or adjacency.dtype!=bool or np.diag(adjacency).any() or not np.array_equal(adjacency,adjacency.T):raise ValueError('Undirected radius graph without self edges required')
    n=len(adjacency);dist=np.where(adjacency,1.,np.inf);np.fill_diagonal(dist,0.)
    for k in range(n):dist=np.minimum(dist,dist[:,k,None]+dist[None,k,:])
    remaining=set(range(n));sizes=[];diameters=[]
    while remaining:
        node=min(remaining);component=set(np.flatnonzero(np.isfinite(dist[node])).tolist());remaining-=component
        ids=sorted(component);sizes.append(len(ids));diameters.append(int(dist[np.ix_(ids,ids)].max()))
    edges=int(adjacency.sum()//2)
    return dict(components=len(sizes),component_sizes=sizes,component_diameters=diameters,
                diameter=diameters[0] if len(sizes)==1 else None,undirected_edges=edges,
                cycle_rank=edges-n+len(sizes),degree=adjacency.sum(1).astype(int).tolist(),
                graph_resampled=False,disconnected_or_cyclic_retained=True)


def make_fixture(split,index,size=128,rooms=6,agents=8):
    """Return independent inference/evaluation dictionaries plus content hashes."""
    size=_integer(size,'size',SIZES);agents=_integer(agents,'agents',AGENT_COUNTS)
    spec=world_spec(split,index,rooms);geometry=rasterize(spec,size);trajectory,assignment=metric_trajectory(spec,agents)
    private,rc,snapped=observe_trajectory(geometry['walls'],trajectory)
    adjacency=radius_adjacency(trajectory[-1],RADIO_RADIUS);directed=tuple(map(tuple,np.argwhere(adjacency).tolist()))
    frame=dict(origin_m=[0.,0.],extent_m=[20.,20.],shape=[size,size],cell_size_m=EXTENT/size,
               coordinate_convention='pose=(x,y); raster=[y,x]',sensor_range_m=SENSOR_RANGE,sensor_beams=256,
               communication_radius_m=RADIO_RADIUS,static_snapshot=True,radio_occlusion=False)
    inputs=dict(private=private.copy(),positions_m=trajectory[-1].copy(),trajectory_m=trajectory.copy(),adjacency=adjacency.copy(),public_frame=frame)
    targets={k:hop_targets(private,directed,k) for k in HOPS};ancestry={k:reachable_sources(agents,directed,k) for k in HOPS}
    evaluation=dict(**geometry,geometry_spec=spec,source_room_id=assignment,pose_cells_rc=rc,snapped_sensor_trajectory_m=snapped,
                    targets=targets,reachable_sources=ancestry,directed_edges=directed)
    graph=graph_diagnostics(adjacency)
    manifest=dict(version=293,world_id=spec['world_id'],split=spec['split'],index=spec['index'],rooms=rooms,agents=agents,size=size,
        geometry_sha256=_json_hash(spec),trajectory_sha256=array_hash(trajectory),adjacency_sha256=array_hash(adjacency),
        private_sha256=array_hash(private),source_sha256=[array_hash(v) for v in private],walls_sha256=array_hash(geometry['walls']),
        target_sha256={str(k):array_hash(v) for k,v in targets.items()},input_fields=list(INPUT_FIELDS),graph=graph,
        max_sensor_snap_error_m=float(np.linalg.norm(snapped-trajectory,axis=-1).max()),
        snap_error_bound_m=float(np.sqrt(2.)*(EXTENT/size)/2),observed_cells=(private>=0).sum((1,2)).astype(int).tolist(),
        peer_known_cells_by_hop={str(k):((v>=0)&(private<0)).sum((1,2)).astype(int).tolist() for k,v in targets.items()},
        reachable_source_counts={str(k):v.sum(1).astype(int).tolist() for k,v in ancestry.items()},
        sampler='Geometry-assisted within-room/corridor metric motion; not learned exploration or inference GT',
        sensor='Three actual first-hit ray observations; strict5m cell-centre distance to continuous poses; no resized mask',
        graph_scope='Static metric radius; all loops/disconnections retained; no inference from room identities',
        targets_scope='Reliable simultaneous all-edge original-source information-access bounds, not compressed recovery claims',
        inference_authority='inputs is centralized experiment storage, NOT global knowledge for any agent; use per_agent_inputs',
        shared_frame_assumed=True,communication_setup_measured=False,communication_100x_claim=False,
        model_executed=False,TEST_opened=False,GPU_used=False,artifact_dataset_written=False)
    return dict(inputs=inputs,evaluation=evaluation,manifest=manifest)


def per_agent_inputs(fixture,node):
    """Own sensor/pose only plus local neighbour IDs; no labels or peer rasters."""
    values=fixture['inputs'];node=_integer(node,'node')
    if set(values)!=set(INPUT_FIELDS) or node>=len(values['private']):raise ValueError('Registered input dictionary and valid node required')
    return dict(private=values['private'][node].copy(),position_m=values['positions_m'][node].copy(),
                trajectory_m=values['trajectory_m'][:,node].copy(),neighbor_ids=np.flatnonzero(values['adjacency'][node]).astype(np.int64),
                public_frame=copy.deepcopy(values['public_frame']))
