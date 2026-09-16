"""Minimal interactive hero for the learned single-slit Sheaf-GBP skill.

The browser directly manipulates the physical slit and parametric-T geometry.
Every animated communication packet is emitted by the frozen policy on a real
radius edge; the page never fabricates decorative graph traffic.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass, replace
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse
import webbrowser

import torch

from .config import (
    HANDLE_LAYOUT_GLYPH_BOUNDARY_V4,
    HANDLE_LAYOUT_PARAMETRIC_T_V3,
)
from .env import contact_points, radius_edges
from .gbp import GBPMessageTrace, homogeneous_topology
from .models import SheafGBPPolicy
from .payload_shapes import payload_rectangles_for_config
from .single_slit_0720_eval import (
    STRICT_STABLE_STEPS,
    load_single_slit_checkpoint,
    strict_final_mask,
)
from .single_slit_contact_feedback import single_slit_contact_feedback_step_v1
from .single_slit_contact_observation import (
    initial_single_slit_contact_observation_context_v1,
    single_slit_contact_observation_context_from_result_v1,
    single_slit_contact_observations_v1,
)
from .single_slit_feedback import SOURCE_POSE, single_slit_scenario


ROLLOUT_LOCK = Lock()
DEFAULT_N_AGENTS = 12
MIN_N_AGENTS = 8
MAX_N_AGENTS = 20
HORIZON = 384
WORLD_Y_M = 1.08
MAX_WALL_LENGTH_M = 0.72
BASE_T_WIDTH_M = 0.930
BASE_T_HEIGHT_M = 0.920
BASE_T_STEM_WIDTH_M = 0.160
BASE_T_CAP_THICKNESS_M = 0.152
RANGES = {
    "upper_wall_length_m": (0.0, MAX_WALL_LENGTH_M),
    "lower_wall_length_m": (0.0, MAX_WALL_LENGTH_M),
    "length_scale": (0.9683, 1.0318),
    "thickness_scale": (0.6680, 1.3380),
}


@dataclass(frozen=True)
class DemoParameters:
    payload_shape: str = "T"
    n_agents: int = DEFAULT_N_AGENTS
    upper_wall_length_m: float = MAX_WALL_LENGTH_M
    lower_wall_length_m: float = MAX_WALL_LENGTH_M
    length_scale: float = 1.0
    thickness_scale: float = 1.0
    ink_seed: int = 20260839

    @property
    def upper_bottom_m(self) -> float:
        return WORLD_Y_M - self.upper_wall_length_m

    @property
    def lower_top_m(self) -> float:
        return -WORLD_Y_M + self.lower_wall_length_m

    @property
    def total_width_m(self) -> float:
        return BASE_T_WIDTH_M * self.length_scale

    @property
    def total_height_m(self) -> float:
        return BASE_T_HEIGHT_M * self.length_scale

    @property
    def stem_width_m(self) -> float:
        return BASE_T_STEM_WIDTH_M * self.thickness_scale

    @property
    def cap_thickness_m(self) -> float:
        return BASE_T_CAP_THICKNESS_M * self.thickness_scale

    @property
    def aperture_m(self) -> float:
        return self.upper_bottom_m - self.lower_top_m

    @property
    def gap_center_m(self) -> float:
        return (self.upper_bottom_m + self.lower_top_m) / 2.0

    def validate(self) -> None:
        if self.payload_shape != "T":
            raise ValueError("this audited public demo is restricted to the T payload")
        if (
            isinstance(self.n_agents, bool)
            or not isinstance(self.n_agents, int)
            or not MIN_N_AGENTS <= self.n_agents <= MAX_N_AGENTS
        ):
            raise ValueError(
                f"n_agents must be an integer in [{MIN_N_AGENTS}, {MAX_N_AGENTS}]"
            )
        for name, limits in RANGES.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not limits[0] <= value <= limits[1]:
                raise ValueError(f"{name} must lie in {limits}, got {value}")
        if self.aperture_m < 0.720 - 1.0e-9:
            raise ValueError("the interactive slit must retain at least 0.720 m aperture")
        if isinstance(self.ink_seed, bool) or not 0 <= self.ink_seed <= 2**31 - 1:
            raise ValueError("ink_seed must be a non-negative 31-bit integer")

    @classmethod
    def from_query(cls, query: dict[str, list[str]]) -> "DemoParameters":
        def number(name: str, default: float) -> float:
            raw = query.get(name, [str(default)])[0]
            return float(raw)

        agents = number("agents", cls.n_agents)
        if not agents.is_integer():
            raise ValueError("agents must be an integer")
        value = cls(
            payload_shape=query.get("shape", [cls.payload_shape])[0].strip().upper(),
            n_agents=int(agents),
            upper_wall_length_m=number("upperWall", cls.upper_wall_length_m),
            lower_wall_length_m=number("lowerWall", cls.lower_wall_length_m),
            length_scale=number("length", cls.length_scale),
            thickness_scale=number("thickness", cls.thickness_scale),
            ink_seed=int(number("seed", cls.ink_seed)),
        )
        value.validate()
        return value


def _tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _bind_environment(policy: torch.nn.Module, cfg: Any) -> None:
    target = getattr(policy, "policy", policy)
    bind = getattr(target, "bind_environment", None)
    if callable(bind):
        bind(cfg)
    elif hasattr(target, "env_cfg"):
        target.env_cfg = cfg
    else:
        raise TypeError("single-slit policy does not expose its environment")


def _trace_policy(policy, observation, edges, hidden, topology):
    target = policy if isinstance(policy, SheafGBPPolicy) else getattr(policy, "policy", None)
    if not isinstance(target, SheafGBPPolicy):
        raise TypeError("interactive hero requires a traceable SheafGBPPolicy")
    return target(
        observation,
        edges,
        hidden,
        topology=topology,
        trace_messages=True,
    )


class DemoEngine:
    def __init__(self, checkpoint: Path, device: str) -> None:
        self.device = torch.device(device)
        self.loaded = load_single_slit_checkpoint(checkpoint, device=self.device)
        self.loaded.policy.eval()
        self.cache: OrderedDict[DemoParameters, tuple[dict[str, Any], ...]] = OrderedDict()

    @torch.no_grad()
    def events(self, params: DemoParameters) -> Iterator[dict[str, Any]]:
        params.validate()
        cached = self.cache.get(params)
        if cached is not None:
            self.cache.move_to_end(params)
            yield from cached
            return
        recorded: list[dict[str, Any]] = []
        cfg = replace(
            self.loaded.cfg,
            n_agents=params.n_agents,
            handle_layout=(
                HANDLE_LAYOUT_PARAMETRIC_T_V3
                if params.payload_shape == "T"
                else HANDLE_LAYOUT_GLYPH_BOUNDARY_V4
            ),
            payload_shape=params.payload_shape,
            t_total_width_m=params.total_width_m,
            t_total_height_m=params.total_height_m,
            t_stem_width_m=params.stem_width_m,
            t_cap_thickness_m=params.cap_thickness_m,
            max_steps=HORIZON,
            device=str(self.device),
        )
        _bind_environment(self.loaded.policy, cfg)
        scenario = single_slit_scenario(
            1,
            cfg,
            visible_aperture_m=params.aperture_m,
        )
        scenario.gap_y[:, 0] = params.gap_center_m
        edges = radius_edges(cfg, device=self.device)
        handles = contact_points(cfg, self.device)
        topology = homogeneous_topology(edges, 1, cfg.n_agents, device=self.device)
        expected_sender = edges.reshape(-1)
        expected_receiver = edges.flip(-1).reshape(-1)
        body_rects = payload_rectangles_for_config(cfg, self.device)

        header = {
            "type": "header",
            "schema": "piano-movers.single-slit-ink-hero.v1",
            "checkpoint_sha256": self.loaded.checkpoint_sha256,
            "n_agents": cfg.n_agents,
            "dt": cfg.dt,
            "max_force": cfg.max_force,
            "world": {"x": 1.62, "y": 1.08, "wall_half_width": 0.060},
            "parameters": {
                **params.__dict__,
                "upper_bottom_m": params.upper_bottom_m,
                "lower_top_m": params.lower_top_m,
                "total_width_m": params.total_width_m,
                "total_height_m": params.total_height_m,
                "stem_width_m": params.stem_width_m,
                "cap_thickness_m": params.cap_thickness_m,
                "aperture_m": params.aperture_m,
                "gap_center_m": params.gap_center_m,
            },
            "ranges": {
                **RANGES,
                "n_agents": (MIN_N_AGENTS, MAX_N_AGENTS),
            },
            "source": list(SOURCE_POSE),
            "goal": _tensor_list(scenario.goal[0]),
            "body_rects": _tensor_list(body_rects),
            "handles": _tensor_list(handles),
            "edges": _tensor_list(edges),
            "sender": _tensor_list(expected_sender),
            "receiver": _tensor_list(expected_receiver),
        }
        recorded.append(header)
        yield header

        state = scenario.state0.clone()
        hidden = self.loaded.policy.init_hidden(1, self.device)
        context = initial_single_slit_contact_observation_context_v1(state, cfg)
        dwell = 0
        contacts = 0
        for tick in range(HORIZON):
            observation = single_slit_contact_observations_v1(state, scenario, cfg, context)
            output = _trace_policy(
                self.loaded.policy,
                observation,
                edges,
                hidden,
                topology,
            )
            if not bool(torch.isfinite(output.forces).all()):
                raise RuntimeError("interactive policy produced non-finite forces")
            transition = single_slit_contact_feedback_step_v1(
                state,
                output.forces,
                scenario,
                cfg,
            )
            trace = output.aux.get("gbp_message_trace")
            if not isinstance(trace, GBPMessageTrace):
                raise RuntimeError("interactive policy omitted its physical GBP trace")
            if not torch.equal(trace.sender, expected_sender) or not torch.equal(
                trace.receiver, expected_receiver
            ):
                raise RuntimeError("GBP trace endpoints changed from the radius graph")
            hit = bool(transition.attempted_hit[0])
            contacts += int(hit)
            dwell = (
                dwell + 1
                if not hit and bool(strict_final_mask(transition.resolved_state, scenario.goal)[0])
                else 0
            )
            success = dwell >= STRICT_STABLE_STEPS
            frame = {
                "type": "frame",
                "step": tick,
                "state": _tensor_list(state[0]),
                "next_state": _tensor_list(transition.resolved_state[0]),
                "hit": hit,
                "clearance_m": float(transition.attempted_clearance_m[0].item()),
                "intensity": _tensor_list(trace.intensity[0]),
                "success": success,
            }
            recorded.append(frame)
            yield frame
            context = single_slit_contact_observation_context_from_result_v1(
                transition,
                output.forces,
                cfg,
            )
            state = transition.resolved_state
            hidden = output.hidden
            if success:
                break

        done_event = {
            "type": "done",
            "success": dwell >= STRICT_STABLE_STEPS,
            "steps": tick + 1,
            "contact_events": contacts,
            "exact_penetration": False,
        }
        recorded.append(done_event)
        self.cache[params] = tuple(recorded)
        self.cache.move_to_end(params)
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        yield done_event


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Sheaf-GBP · Single Slit</title>
  <style>
    :root { color-scheme: light; --ink:#171717; --graphite:#505050; --paper:#e7e7e7; }
    * { box-sizing:border-box; }
    html,body { margin:0; width:100%; height:100%; overflow:hidden; background:#dcdcdc; }
    body { font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    #stage { position:relative; width:100vw; height:100vh; min-height:560px; isolation:isolate; }
    canvas { position:absolute; inset:0; display:block; width:100%; height:100%; touch-action:none; }
    #dock {
      position:absolute; z-index:3; left:50%; bottom:18px; transform:translateX(-50%);
      display:flex; align-items:center; gap:12px; height:48px; padding:0 12px;
      border:1px solid rgba(30,30,30,.16); border-radius:24px;
      background:rgba(243,243,243,.72); box-shadow:0 12px 34px rgba(28,28,28,.13);
      backdrop-filter:blur(16px) saturate(.8); transition:opacity .35s ease;
    }
    #dock.running .geometry { opacity:.15; pointer-events:none; }
    button { border:0; background:none; color:#242424; width:31px; height:31px; padding:5px;
      display:grid; place-items:center; border-radius:50%; cursor:pointer; }
    button:hover { background:rgba(32,32,32,.08); }
    button:focus-visible,input:focus-visible { outline:2px solid #555555; outline-offset:2px; }
    button svg { width:17px; height:17px; fill:none; stroke:currentColor; stroke-width:1.7; }
    select { width:39px; height:30px; padding:0 0 0 10px; border:0; border-radius:15px;
      color:#202020; background:rgba(32,32,32,.055); font:650 13px/1 ui-monospace,SFMono-Regular,Consolas,monospace;
      cursor:pointer; appearance:auto; }
    select:hover { background:rgba(32,32,32,.09); }
    .shape-mark { width:30px; height:30px; display:grid; place-items:center; border-radius:15px;
      color:#202020; background:rgba(32,32,32,.055); font:650 13px/1 ui-monospace,SFMono-Regular,Consolas,monospace; }
    #play { width:38px; height:38px; color:#f4f4f4; background:#1b1b1b; }
    #play.waiting { animation:pulse 1.15s ease-in-out infinite; }
    @keyframes pulse { 50% { box-shadow:0 0 0 8px rgba(72,72,72,.11); } }
    .divider { width:1px; height:22px; background:rgba(28,28,28,.13); }
    .geometry { display:flex; gap:10px; transition:opacity .3s ease; }
    label { width:70px; display:grid; grid-template-columns:13px 1fr; align-items:center; gap:5px;
      color:rgba(29,29,29,.58); font:600 11px/1 ui-monospace,SFMono-Regular,Consolas,monospace; }
    label.agents { width:91px; grid-template-columns:15px 48px 18px; gap:4px; }
    .ant-icon { width:15px; height:15px; overflow:visible; fill:#1b1b1b; stroke:#1b1b1b; stroke-width:1.15; }
    output { text-align:right; color:#252525; font-variant-numeric:tabular-nums; }
    input[type=range] { width:50px; height:18px; margin:0; appearance:none; background:transparent; cursor:ew-resize; }
    input[type=range]::-webkit-slider-runnable-track { height:1px; background:rgba(28,28,28,.28); }
    input[type=range]::-webkit-slider-thumb { appearance:none; width:8px; height:8px; margin-top:-3.5px;
      border-radius:50%; background:#1b1b1b; box-shadow:0 0 0 3px rgba(244,244,244,.8); }
    input[type=range]::-moz-range-track { height:1px; background:rgba(28,28,28,.28); }
    input[type=range]::-moz-range-thumb { width:8px; height:8px; border:0; border-radius:50%; background:#1b1b1b; }
    #mark { position:absolute; z-index:2; top:20px; right:22px; width:30px; height:30px;
      display:grid; place-items:center; opacity:0; border:1px solid rgba(68,68,68,.34); border-radius:50%;
      color:#4c4c4c; background:rgba(240,240,240,.56); backdrop-filter:blur(8px); transition:opacity .4s; }
    #mark.show { opacity:1; }
    #mark svg { width:17px; height:17px; fill:none; stroke:currentColor; stroke-width:2; }
    #buffer { position:absolute; z-index:2; left:22px; bottom:24px; display:flex; gap:4px; opacity:0; }
    #buffer.show { opacity:.7; }
    #buffer i { width:4px; height:4px; border-radius:50%; background:#5b5b5b; animation:b 1s infinite; }
    #buffer i:nth-child(2){animation-delay:.16s} #buffer i:nth-child(3){animation-delay:.32s}
    @keyframes b { 50% { transform:translateY(-5px); opacity:.35; } }
    @media(max-width:760px){ #dock{gap:7px;bottom:10px}.geometry{gap:4px}label{width:55px}label.agents{width:76px;grid-template-columns:14px 36px 18px}input[type=range]{width:36px} }
  </style>
</head>
<body>
<main id="stage">
  <canvas id="canvas" aria-label="Interactive single-slit Sheaf-GBP transport"></canvas>
  <div id="mark" aria-live="polite"><svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 7"/></svg></div>
  <div id="buffer" aria-hidden="true"><i></i><i></i><i></i></div>
  <div id="dock">
    <span class="shape-mark" title="T payload" aria-label="T payload">T</span>
    <input id="shape" type="hidden" value="T">
    <button id="random" title="New ink and geometry" aria-label="New ink and geometry">
      <svg viewBox="0 0 24 24"><path d="M4 7h3l10 10h3M17 7h3v3M4 17h3l3-3M14 10l3-3h3v3M17 17h3v-3"/></svg>
    </button>
    <div class="divider"></div>
    <div class="geometry">
      <label class="agents" title="Ant count">
        <svg class="ant-icon" viewBox="0 0 18 18" aria-hidden="true"><ellipse cx="4" cy="9" rx="2.2" ry="2.7"/><ellipse cx="8.3" cy="9" rx="1.8" ry="2.2"/><circle cx="12.3" cy="9" r="1.8"/><path d="M6.5 7 3.5 3M6.5 11l-3 4M9 7l2-4M9 11l2 4M13.5 8l2-3M13.5 10l2 3" fill="none"/></svg>
        <input id="agents" type="range" min="8" max="20" step="1" value="12">
        <output id="agents-value" for="agents">12</output>
      </label>
      <label class="shape-size" title="Rectangle length"><span>↗</span><input id="length" type="range" min="0.9683" max="1.0318" step="0.0005" value="1"></label>
      <label class="shape-size" title="Rectangle thickness"><span>▮</span><input id="thickness" type="range" min="0.668" max="1.338" step="0.002" value="1"></label>
    </div>
    <div class="divider"></div>
    <button id="play" title="Run" aria-label="Run">
      <svg id="play-icon" viewBox="0 0 24 24"><path d="M9 6.5 17 12l-8 5.5Z" fill="currentColor" stroke="none"/></svg>
    </button>
    <button id="reset" title="Reset" aria-label="Reset">
      <svg viewBox="0 0 24 24"><path d="M6.5 7.5A7 7 0 1 1 5 13M5 5v5h5"/></svg>
    </button>
  </div>
</main>
<script>
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
const dock=document.getElementById('dock'), play=document.getElementById('play');
const playIcon=document.getElementById('play-icon'), mark=document.getElementById('mark');
const buffer=document.getElementById('buffer');
const inputs={shape:document.getElementById('shape'),agents:document.getElementById('agents'),length:document.getElementById('length'),thickness:document.getElementById('thickness')};
const agentsValue=document.getElementById('agents-value');
const DEFAULTS={shape:'T',agents:12,upperWall:.72,lowerWall:.72,length:1,thickness:1,seed:20260839};
let p={...DEFAULTS}, header=null, frames=[], done=null, aborter=null, generation=0;
let playing=false, timeline=0, lastTime=0, starving=false, drag=null, background=null;
const STEP_MS=56, TAU=Math.PI*2, WORLD_Y=1.08, MAX_WALL=.72;
const RANDOM_APERTURE_MIN=.72, RANDOM_APERTURE_MAX=.80;
const T_BASE={width:.930,height:.920,stem:.160,cap:.152};
const QA_QUERY=new URLSearchParams(location.search),QA_AUTO=QA_QUERY.get('autoplay')==='1';
const QA_FRAME=Number(QA_QUERY.get('frame'));
let qaStarted=false;
const themes=[
  {paper:'#e8e8e8',ink:'#181818',loadLight:'#d6d6d6',load:'#898989',loadDark:'#4d4d4d',packet:'#f8f8f8',core:'#ffffff'},
  {paper:'#e3e3e3',ink:'#171717',loadLight:'#cdcdcd',load:'#767676',loadDark:'#454545',packet:'#f2f2f2',core:'#ffffff'},
  {paper:'#ececec',ink:'#1d1d1d',loadLight:'#dddddd',load:'#999999',loadDark:'#555555',packet:'#fafafa',core:'#ffffff'},
  {paper:'#dfdfdf',ink:'#181818',loadLight:'#c8c8c8',load:'#7e7e7e',loadDark:'#494949',packet:'#f5f5f5',core:'#ffffff'},
  {paper:'#e6e6e6',ink:'#151515',loadLight:'#cfcfcf',load:'#666666',loadDark:'#3f3f3f',packet:'#eeeeee',core:'#ffffff'}
];
let theme=themes[p.seed%themes.length];

function rng(seed){let a=seed>>>0;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function resize(){const d=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(innerWidth*d);canvas.height=Math.round(innerHeight*d);canvas.style.width=innerWidth+'px';canvas.style.height=innerHeight+'px';ctx.setTransform(d,0,0,d,0,0);background=null;draw()}
function view(){const pad=Math.max(34,innerWidth*.045), top=28, bottom=82;const w=innerWidth-pad*2,h=innerHeight-top-bottom;const s=Math.min(w/3.24,h/2.16);return{cx:innerWidth/2,cy:top+h/2,s}}
function S(x,y){const v=view();return[v.cx+x*v.s,v.cy-y*v.s]}
function W(x,y){const v=view();return[(x-v.cx)/v.s,(v.cy-y)/v.s]}
function morphDims(){return{width:T_BASE.width*p.length,height:T_BASE.height*p.length,stem:T_BASE.stem*p.thickness,cap:T_BASE.cap*p.thickness}}
function wallEdges(){return{upper:WORLD_Y-p.upperWall,lower:-WORLD_Y+p.lowerWall}}
function morphRects(){const d=morphDims(),center=-.06325,bottom=center-d.height/2,top=center+d.height/2,capBottom=top-d.cap,overlap=(.110/.2035)*d.cap,stemTop=capBottom+overlap;return[rect(0,(bottom+stemTop)/2,d.stem,stemTop-bottom),rect(0,(capBottom+top)/2,d.width,d.cap)]}
function rect(cx,cy,w,h){return[[-w/2+cx,-h/2+cy],[w/2+cx,-h/2+cy],[w/2+cx,h/2+cy],[-w/2+cx,h/2+cy]]}
function lerpAngle(a,b,t){let d=Math.atan2(Math.sin(b-a),Math.cos(b-a));return a+d*t}
function poseAt(){if(!frames.length||timeline<=0)return[-1.05,0,0,0,0,0];const i=Math.min(Math.floor(timeline),frames.length-1),f=frames[i],q=Math.min(1,timeline-i);const m=q<.22?0:(q-.22)/.78;return[f.state[0]+(f.next_state[0]-f.state[0])*m,f.state[1]+(f.next_state[1]-f.state[1])*m,lerpAngle(f.state[2],f.next_state[2],m),0,0,0]}
function transform(pt,state){const c=Math.cos(state[2]),s=Math.sin(state[2]);return[state[0]+c*pt[0]-s*pt[1],state[1]+s*pt[0]+c*pt[1]]}

function makeBackground(){const off=document.createElement('canvas');off.width=innerWidth;off.height=innerHeight;const g=off.getContext('2d'),r=rng(p.seed);g.fillStyle=theme.paper;g.fillRect(0,0,off.width,off.height);
  const wash=g.createRadialGradient(off.width*.52,off.height*.46,20,off.width*.52,off.height*.46,off.width*.72);wash.addColorStop(0,'rgba(255,255,255,.28)');wash.addColorStop(1,'rgba(68,68,68,.09)');g.fillStyle=wash;g.fillRect(0,0,off.width,off.height);
  g.save();g.filter='blur(9px)';for(let i=0;i<8;i++){g.beginPath();const y=off.height*(.08+r()*.84),x=-80+r()*off.width*.25;g.moveTo(x,y);for(let k=1;k<=5;k++){const xx=x+k*(off.width+160)/5,yy=y+(r()-.5)*off.height*.24;g.quadraticCurveTo(xx-(off.width/10),yy+(r()-.5)*55,xx,yy)}g.strokeStyle=`rgba(36,36,36,${.028+r()*.052})`;g.lineWidth=22+r()*82;g.lineCap='round';g.stroke()}g.restore();
  const img=g.getImageData(0,0,off.width,off.height);for(let i=0;i<img.data.length;i+=4){const n=(r()-.5)*8;img.data[i]+=n;img.data[i+1]+=n;img.data[i+2]+=n}g.putImageData(img,0,0);background=off}
function drawBackground(){if(!background)makeBackground();ctx.drawImage(background,0,0)}

function pathPoly(points,state){ctx.beginPath();points.forEach((q,i)=>{const z=S(...transform(q,state));i?ctx.lineTo(...z):ctx.moveTo(...z)});ctx.closePath()}
function drawGoal(rects){const st=[1.05,0,0];ctx.save();ctx.strokeStyle='rgba(36,36,36,.16)';ctx.lineWidth=1.15;ctx.setLineDash([3,7]);for(const q of rects){pathPoly(q,st);ctx.stroke()}ctx.restore()}
function drawBarrier(){const v=view(),x=S(0,0)[0],half=.06*v.s,edges=wallEdges(),yU=S(0,edges.upper)[1],yL=S(0,edges.lower)[1];ctx.save();const grad=ctx.createLinearGradient(x-half,0,x+half,0);grad.addColorStop(0,'rgba(24,24,24,.80)');grad.addColorStop(.48,'rgba(42,42,42,.95)');grad.addColorStop(1,'rgba(14,14,14,.78)');ctx.fillStyle=grad;ctx.strokeStyle='rgba(244,244,244,.17)';ctx.lineWidth=1;
  const slab=(from,to,visible)=>{if(!visible)return;ctx.fillStyle=grad;ctx.fillRect(x-half,from,half*2,to-from);for(let i=0;i<5;i++){const dx=x-half+(i+.7)*half*2/5;ctx.beginPath();ctx.moveTo(dx,from);ctx.lineTo(dx+(i%2?2:-2),to);ctx.stroke()}};
  slab(0,yU,p.upperWall>1e-4);slab(yL,innerHeight,p.lowerWall>1e-4);
  if(!playing){for(const y of [yU,yL]){ctx.beginPath();ctx.arc(x,y,8,0,TAU);ctx.fillStyle='rgba(238,238,238,.92)';ctx.fill();ctx.strokeStyle='rgba(28,28,28,.62)';ctx.lineWidth=1.2;ctx.stroke();ctx.beginPath();ctx.moveTo(x-3,y);ctx.lineTo(x+3,y);ctx.stroke()}}ctx.restore()}
function drawTrail(){if(!frames.length||timeline<1)return;ctx.save();ctx.beginPath();const end=Math.min(Math.floor(timeline),frames.length-1);for(let i=0;i<=end;i++){const q=S(frames[i].state[0],frames[i].state[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)}ctx.strokeStyle='rgba(46,46,46,.18)';ctx.lineWidth=1.25;ctx.stroke();ctx.restore()}
function drawBody(state,rects){ctx.save();const top=S(...transform([0,.46],state)),bottom=S(...transform([0,-.55],state));const grad=ctx.createLinearGradient(top[0],top[1],bottom[0],bottom[1]);grad.addColorStop(0,theme.loadLight);grad.addColorStop(.28,theme.load);grad.addColorStop(.74,theme.loadDark);grad.addColorStop(1,'#393939');for(let k=0;k<rects.length;k++){pathPoly(rects[k],state);ctx.fillStyle=grad;ctx.fill();ctx.strokeStyle='rgba(28,28,28,.82)';ctx.lineWidth=1.45;ctx.stroke()}ctx.globalAlpha=.17;ctx.strokeStyle='#f4f4f4';ctx.lineWidth=.8;for(let i=-4;i<=4;i++){const a=transform([i*.09,-.48],state),b=transform([i*.05,.40],state);ctx.beginPath();ctx.moveTo(...S(...a));ctx.lineTo(...S(...b));ctx.stroke()}ctx.restore()}
function drawAnt(local,state,index){const w=transform(local,state),z=S(...w),v=view();const center=[0,-.06325],a=state[2]+Math.atan2(local[1]-center[1],local[0]-center[0]);const scale=Math.max(7.2,v.s*.034);ctx.save();ctx.translate(...z);ctx.rotate(a);ctx.strokeStyle='rgba(16,16,16,.96)';ctx.lineWidth=1.2;ctx.lineCap='round';for(const sy of [-1,1])for(let k=-1;k<=1;k++){ctx.beginPath();ctx.moveTo(k*scale*.10,sy*scale*.13);ctx.quadraticCurveTo((k-.05)*scale,sy*scale*.52,(k-.34)*scale,sy*scale*.70);ctx.stroke()}ctx.fillStyle='#161616';ctx.beginPath();ctx.ellipse(-scale*.33,0,scale*.29,scale*.21,0,0,TAU);ctx.fill();ctx.beginPath();ctx.ellipse(-scale*.02,0,scale*.21,scale*.17,0,0,TAU);ctx.fill();ctx.fillStyle='rgba(232,232,232,.96)';ctx.beginPath();ctx.ellipse(scale*.27,0,scale*.18,scale*.15,0,0,TAU);ctx.fill();ctx.strokeStyle='#161616';ctx.stroke();for(const sy of [-1,1]){ctx.beginPath();ctx.moveTo(scale*.39,sy*scale*.05);ctx.quadraticCurveTo(scale*.57,sy*scale*.18,scale*.63,sy*scale*.34);ctx.stroke()}ctx.restore()}
function quadPoint(a,c,b,t){const u=1-t;return[u*u*a[0]+2*u*t*c[0]+t*t*b[0],u*u*a[1]+2*u*t*c[1]+t*t*b[1]]}
function drawPacket(a,b,t,strength,lane){const dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy)||1,nx=-dy/len,ny=dx/len,curve=(lane?1:-1)*(5+Math.min(15,len*.07)),c=[(a[0]+b[0])/2+nx*curve,(a[1]+b[1])/2+ny*curve];ctx.save();ctx.strokeStyle=`rgba(18,18,18,${.035+.095*strength})`;ctx.lineWidth=.45+.65*strength;ctx.shadowColor=theme.packet;ctx.shadowBlur=3+6*strength;ctx.beginPath();ctx.moveTo(...a);ctx.quadraticCurveTo(...c,...b);ctx.stroke();for(let k=4;k>=0;k--){const tt=Math.max(0,t-k*.052),q=quadPoint(a,c,b,tt),alpha=(1-k/5)*(.22+.72*strength);ctx.beginPath();ctx.arc(q[0],q[1],1.15+(4-k)*.22,0,TAU);ctx.fillStyle=k===0?theme.core:`rgba(250,250,250,${alpha})`;ctx.shadowColor=theme.packet;ctx.shadowBlur=k===0?15:8;ctx.fill()}const q=quadPoint(a,c,b,t),q2=quadPoint(a,c,b,Math.min(1,t+.012)),ang=Math.atan2(q2[1]-q[1],q2[0]-q[0]);ctx.translate(q[0],q[1]);ctx.rotate(ang);ctx.shadowColor='rgba(0,0,0,.52)';ctx.shadowBlur=4;ctx.beginPath();ctx.moveTo(7,0);ctx.lineTo(-3.8,-3.4);ctx.lineTo(-1.2,0);ctx.lineTo(-3.8,3.4);ctx.closePath();ctx.fillStyle='rgba(18,18,18,.62)';ctx.fill();ctx.scale(.72,.72);ctx.shadowColor=theme.packet;ctx.shadowBlur=10;ctx.beginPath();ctx.moveTo(7,0);ctx.lineTo(-3.8,-3.4);ctx.lineTo(-1.2,0);ctx.lineTo(-3.8,3.4);ctx.closePath();ctx.fillStyle=theme.core;ctx.fill();ctx.restore()}
function drawMessages(state){if((!playing&&timeline<=0)||!header||!frames.length)return;const i=Math.min(Math.floor(timeline),frames.length-1),f=frames[i],phase=timeline-i,sweeps=f.intensity.length;if(!sweeps)return;const sweep=Math.min(sweeps-1,Math.floor(phase*sweeps)),t=(phase*sweeps)%1;const vals=f.intensity[sweep],sorted=vals.filter(v=>v>0).sort((a,b)=>a-b),scale=sorted.length?sorted[Math.floor(sorted.length*.86)]:1;const contacts=header.handles.map(q=>S(...transform(q,state)));for(let m=0;m<header.sender.length;m++){const raw=vals[m]||0,str=Math.max(.10,Math.min(1,Math.sqrt(raw/(scale||1))));drawPacket(contacts[header.sender[m]],contacts[header.receiver[m]],t,str,m%2)}}
function draw(){drawBackground();if(!header&&p.shape!=='T'){drawBarrier();return}const rects=header?header.body_rects:morphRects(),state=poseAt();drawGoal(rects);drawBarrier();drawTrail();drawBody(state,rects);const handles=header?header.handles:previewHandles();handles.forEach((q,i)=>drawAnt(q,state,i));drawMessages(state)}
function previewHandles(){const d=morphDims(),center=-.06325,bottom=center-d.height/2,top=center+d.height/2,capBottom=top-d.cap,capCenter=(capBottom+top)/2,topCount=Math.floor((p.agents+1)/2),pts=[[-d.width/2,capCenter]];if(topCount>2){const extent=(topCount===4?.45:.68)*d.width/2;for(let i=0;i<topCount-2;i++){const f=i/Math.max(1,topCount-3);pts.push([-extent+2*extent*f,top])}}pts.push([d.width/2,capCenter]);const remaining=p.agents-topCount,pairs=Math.floor(remaining/2),hasCenter=remaining%2===1;for(let level=0;level<pairs;level++){let y=top+(level+1)*(bottom-top)/(pairs+1);y=Math.min(capBottom,y);pts.push([-d.stem/2,y],[d.stem/2,y])}if(hasCenter)pts.push([0,bottom]);return pts}

function paramsUrl(){return`/api/stream?shape=${encodeURIComponent(p.shape)}&agents=${p.agents}&upperWall=${p.upperWall}&lowerWall=${p.lowerWall}&length=${p.length}&thickness=${p.thickness}&seed=${p.seed}`}
async function prepare(){generation++;const mine=generation;if(aborter)aborter.abort();aborter=new AbortController();header=null;frames=[];done=null;timeline=0;playing=false;starving=false;qaStarted=false;mark.classList.remove('show');dock.classList.remove('running');buffer.classList.add('show');play.classList.add('waiting');draw();try{const res=await fetch(paramsUrl(),{signal:aborter.signal});if(!res.ok)throw Error(await res.text());const reader=res.body.getReader(),decoder=new TextDecoder();let pending='';while(true){const q=await reader.read();if(q.done)break;pending+=decoder.decode(q.value,{stream:true});let cut;while((cut=pending.indexOf('\n'))>=0){const line=pending.slice(0,cut).trim();pending=pending.slice(cut+1);if(!line)continue;const ev=JSON.parse(line);if(mine!==generation)return;if(ev.type==='header'){header=ev;draw()}else if(ev.type==='frame'){frames.push(ev);if(frames.length>=8){buffer.classList.remove('show');play.classList.remove('waiting');if(QA_AUTO&&!qaStarted){qaStarted=true;toggle()}}}else if(ev.type==='done'){done=ev;buffer.classList.remove('show');play.classList.remove('waiting');if(Number.isFinite(QA_FRAME)&&QA_FRAME>0){playing=false;timeline=Math.min(frames.length-.01,QA_FRAME);dock.classList.remove('running');setPlayIcon();draw()}}}}}catch(e){if(e.name!=='AbortError'){console.error(e);buffer.classList.remove('show');play.classList.remove('waiting')}}}
function setPlayIcon(){playIcon.innerHTML=playing?'<path d="M8 7v10M16 7v10"/>':'<path d="M9 6.5 17 12l-8 5.5Z" fill="currentColor" stroke="none"/>'}
function toggle(){if(!frames.length){play.classList.add('waiting');return}playing=!playing;dock.classList.toggle('running',playing);setPlayIcon();lastTime=performance.now();if(playing)requestAnimationFrame(tick)}
function tick(now){if(!playing)return;const dt=Math.min(50,now-lastTime);lastTime=now;const maxPlayable=done?frames.length:Math.max(0,frames.length-3);if(timeline<maxPlayable){timeline=Math.min(maxPlayable,timeline+dt/STEP_MS);starving=false}else if(!done){starving=true;buffer.classList.add('show')}if(done&&timeline>=frames.length-1e-3){playing=false;timeline=Math.max(0,frames.length-1e-3);dock.classList.remove('running');setPlayIcon();mark.classList.toggle('show',!!done.success);buffer.classList.remove('show')}draw();if(playing)requestAnimationFrame(tick)}
function reset(){playing=false;timeline=0;dock.classList.remove('running');mark.classList.remove('show');setPlayIcon();draw()}
function sync(){p.shape=inputs.shape.value;p.agents=+inputs.agents.value;p.length=+inputs.length.value;p.thickness=+inputs.thickness.value;agentsValue.value=String(p.agents);const fixed=p.shape!=='T';inputs.length.disabled=fixed;inputs.thickness.disabled=fixed;document.querySelectorAll('.shape-size').forEach(el=>el.style.opacity=fixed?'.22':'1');header=null;frames=[];done=null;timeline=0;draw()}
Object.values(inputs).forEach(el=>{el.addEventListener('input',sync);el.addEventListener('change',prepare)});
document.getElementById('random').addEventListener('click',()=>{const r=rng((Date.now()^p.seed)>>>0);p.seed=Math.floor(r()*2147483647);inputs.agents.value=p.agents=8+Math.floor(r()*13);inputs.length.value=p.length=.9683+r()*(1.0318-.9683);inputs.thickness.value=p.thickness=.668+r()*(1.338-.668);const aperture=RANDOM_APERTURE_MIN+r()*(RANDOM_APERTURE_MAX-RANDOM_APERTURE_MIN),centerLimit=(aperture-RANDOM_APERTURE_MIN)/2,center=(r()*2-1)*centerLimit;p.upperWall=WORLD_Y-center-aperture/2;p.lowerWall=WORLD_Y+center-aperture/2;agentsValue.value=String(p.agents);theme=themes[p.seed%themes.length];background=null;prepare()});
play.addEventListener('click',toggle);document.getElementById('reset').addEventListener('click',reset);
canvas.addEventListener('pointerdown',e=>{if(playing)return;const edges=wallEdges(),x=S(0,0)[0],yu=S(0,edges.upper)[1],yl=S(0,edges.lower)[1];if(Math.hypot(e.clientX-x,e.clientY-yu)<18)drag='upper';else if(Math.hypot(e.clientX-x,e.clientY-yl)<18)drag='lower';if(drag)canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!drag)return;const y=W(e.clientX,e.clientY)[1];if(drag==='upper')p.upperWall=Math.max(0,Math.min(MAX_WALL,WORLD_Y-y));else p.lowerWall=Math.max(0,Math.min(MAX_WALL,WORLD_Y+y));header=null;frames=[];done=null;timeline=0;draw()});
canvas.addEventListener('pointerup',e=>{if(!drag)return;drag=null;prepare()});
window.addEventListener('resize',resize);sync();resize();prepare();
</script>
</body>
</html>"""


class DemoHandler(BaseHTTPRequestHandler):
    engine: DemoEngine

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_text(self, status: HTTPStatus, value: str, content_type: str) -> None:
        data = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(HTTPStatus.OK, INDEX_HTML, "text/html; charset=utf-8")
            return
        if parsed.path != "/api/stream":
            self._send_text(HTTPStatus.NOT_FOUND, "not found", "text/plain")
            return
        try:
            params = DemoParameters.from_query(parse_qs(parsed.query))
        except (TypeError, ValueError) as exc:
            self._send_text(HTTPStatus.BAD_REQUEST, str(exc), "text/plain")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            with ROLLOUT_LOCK:
                for event in self.engine.events(params):
                    self.wfile.write(
                        json.dumps(event, separators=(",", ":"), allow_nan=False).encode("utf-8")
                        + b"\n"
                    )
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as exc:  # fail visibly without taking down the server
            try:
                self.wfile.write(
                    json.dumps({"type": "error", "message": str(exc)}).encode("utf-8")
                    + b"\n"
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    DemoHandler.engine = DemoEngine(args.checkpoint.resolve(), args.device)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}/"
    print(url, flush=True)
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
