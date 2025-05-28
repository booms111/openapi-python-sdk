import os
import sys
import torch
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
from collections import defaultdict
import time
import cv2
import scipy.sparse
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json # For intrinsics
import pandas as pd # For submission CSV and validation
import networkx as nx # For graph operations
import pymeshlab # For 3D mesh filtering
from sklearn.cluster import DBSCAN as SklearnDBSCAN # For scene clustering

# Configure logging for the pipeline module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import Configuration from Notebook's Main Scope ---
try:
    from main import HybridConfig
    logging.info("Pipeline_module.py: Successfully imported HybridConfig from main.")
except ImportError:
    logging.error("Pipeline_module.py: Could not import HybridConfig from main.")
    logging.error("Using a fallback HybridConfig definition. Ensure HybridConfig in main is correct.")
    # Fallback definition for standalone testing of pipeline_module.py (less likely in Kaggle)
    @dataclass
    class HybridConfig:
        IMAGE_DIR: str = "fallback_test_data"
        OUTPUT_DIR: str = "fallback_output"
        DEVICE: str = 'cpu'
        nfeatures: int = 0
        nOctaveLayers: int = 3
        contrastThreshold: float = .04 # MODIFIED
        edgeThreshold: float = 10.0
        sigma: float = 1.6
        min_sift_matches: int = 15
        VGGT_WEIGHTS: str = "dummy_vggt.pth"
        DUST3R_WEIGHTS: str = "dummy_dust3r.pth"
        VISUALIZE: bool = False
        ransac_threshold_px: float = .5 # MODIFIED
        ransac_confidence: float = .999 # MODIFIED
        MAX_WORKERS: int = 1
        BATCH_SIZE: int = 1
        DEFAULT_FX: float = 1000.0
        DEFAULT_FY: float = 1000.0
        DEFAULT_CX: float = 512.0
        DEFAULT_CY: float = 384.0
        DEFAULT_CAMERA_INTRINSICS: np.ndarray = field(default_factory=lambda:np.array([[1000.0,0,512.0],[0,1000.0,384.0],[0,0,1]],dtype=np.float32))
        MIN_IMAGES_PER_SCENE: int = 3
        OUTLIER_THRESHOLD: float = 10.0
        BA_LOSS: str = 'huber'
        BA_F_SCALE: float = .5 # MODIFIED
        BA_MAX_NFEV: int = 100
        BA_VERBOSE: int = 0
        BA_FIX_FIRST_N_CAMERAS: int = 1
        BA_RESIDUAL_THRESHOLD: float = 7.0
        DUST3R_CONFIDENCE_THRESHOLD: float = .6 # MODIFIED
        DUST3R_PAIR_THRESHOLD: float = 7.0
        SCENE_EPSILON: float = .30 # MODIFIED (e.g., .25 instead of 0.25)
        VERBOSE: bool = True
        VALIDATE: bool = False
        MESHLAB_FILTERS: List[str] = field(default_factory=lambda:["Simplification: Quadric Edge Collapse Decimation","Laplacian Smooth"])
        MESHLAB_TARGET_FACES: int = 15000
        GSPLAT_RENDER_SIZE: Tuple[int,int]=(800,600)
        DATASET_NAME: str = "dummy_dataset"
        INTRINSICS_FILE: Optional[str] = None


# --- Define Dataclasses for Data Structures ---
@dataclass
class FeatureData:
    image_id: str
    image_tensor: np.ndarray 
    image_shape: Tuple[int, int] 
    intrinsics: np.ndarray
    image_rgb: Optional[np.ndarray] = None

@dataclass
class MatchData:
    id1: str
    id2: str
    point_cloud: np.ndarray
    relative_R: np.ndarray
    relative_t: np.ndarray
    confidence: float

@dataclass
class CameraPose:
    id: str 
    R: np.ndarray 
    t: np.ndarray 
    is_fixed: bool = False

@dataclass
class SceneData:
    scene_id: str
    image_ids: List[str]
    matches: Dict[Tuple[str, str], MatchData] = field(default_factory=dict)
    poses: Dict[str, CameraPose] = field(default_factory=dict)
    points_3d: Optional[np.ndarray] = None

@dataclass
class BundleAdjustmentInput:
    camera_params: np.ndarray 
    points_3d: np.ndarray     
    point_indices: np.ndarray 
    camera_indices: np.ndarray
    observations: np.ndarray  
    camera_intrinsics_dict: Dict[str, np.ndarray] 
    fixed_camera_indices: Set[int] 
    cam_id_map: Dict[int, str] 

VGGT_DATASET_ROOT_MOUNT = "/kaggle/input/vggt"
DUST3R_DATASET_ROOT_MOUNT = "/kaggle/input/dust3r"
VGGT_PACKAGE_PARENT_DIR = os.path.join(VGGT_DATASET_ROOT_MOUNT, "transformers", "default", "1", "vggt-main", "vggt-main")
DUST3R_PACKAGE_PARENT_DIR = os.path.join(DUST3R_DATASET_ROOT_MOUNT, "transformers", "default", "1", "dust3r-main", "dust3r-main")

if os.path.isdir(os.path.join(VGGT_PACKAGE_PARENT_DIR, "vggt")):
    sys.path.append(VGGT_PACKAGE_PARENT_DIR)
else:
    logging.error(f"VGGT package 'vggt' not found at: {os.path.join(VGGT_PACKAGE_PARENT_DIR, 'vggt')}")
if os.path.isdir(os.path.join(DUST3R_PACKAGE_PARENT_DIR, "dust3r")):
    sys.path.append(DUST3R_PACKAGE_PARENT_DIR)
else:
    logging.error(f"DUSt3R package 'dust3r' not found at: {os.path.join(DUST3R_PACKAGE_PARENT_DIR, 'dust3r')}")

try:
    from vggt.models.vggt import VGGT
except ImportError:
    class VGGT_placeholder(torch.nn.Module):
        def __init__(self): super().__init__(); self.fc=torch.nn.Linear(1,1); self.device="cpu"
        def forward(self, x): return {"poses": torch.zeros(x.shape[0] if isinstance(x, torch.Tensor) else 1, 6, device=self.device), "point_cloud": torch.zeros(100, 3, device=self.device), "confidence": torch.ones(100, device=self.device)}
        def load_state_dict(self, s, strict=True): pass
        def to(self, dev, dtype=None): self.device=dev; return self
        def eval(self): return self
    VGGT = VGGT_placeholder
    logging.warning("Using DUMMY VGGT class.")
try:
    from dust3r.model import DUSt3R
except ImportError:
    class DUSt3R_placeholder(torch.nn.Module):
        def __init__(self, pretrained=None): super().__init__(); self.fc=torch.nn.Linear(1,1); self.pretrained_path=pretrained; self.device="cpu"
        def reconstruct(self, t1, t2):
            h,w = t1.shape[2], t1.shape[3]; sf = (h*w)/(512*512)
            pc = torch.randn(500,3,device=self.device)*.1*sf # MODIFIED
            poses = torch.randn(12,device=self.device)*.1 # MODIFIED
            poses[0:3]=torch.tensor([.0,.0,.0],device=self.device); poses[3:6]=torch.tensor([.0,.0,.0],device=self.device) # MODIFIED
            poses[6:9]=torch.tensor([.1,.0,.0],device=self.device); poses[9:12]=torch.tensor([.1,.0,.05],device=self.device) # MODIFIED
            return pc, poses
        def to(self, dev, dtype=None): self.device=dev; return self
        def eval(self): return self
    DUSt3R = DUSt3R_placeholder
    logging.warning("Using DUMMY DUSt3R class.")

GSPLAT_AVAILABLE = False
try:
    from gsplat import rasterization, spherical_harmonics, project_gaussians
    GSPLAT_AVAILABLE = True
except ImportError:
    def rasterization(*args,**kwargs): H,W=(kwargs.get("height",100),kwargs.get("width",100)); dev="cpu"; dev=getattr(args[0],'device',dev) if args else dev; dev=getattr(kwargs.get("means"),'device',dev); return torch.zeros((1,3,H,W),device=dev), torch.zeros((1,H,W),device=dev), {}
    def project_gaussians(*args,**kwargs): N=kwargs.get("means3D",torch.tensor([])).shape[0]; dev="cpu"; dev=getattr(args[0],'device',dev) if args else dev; dev=getattr(kwargs.get("means3D"),'device',dev); return torch.zeros((N,2),device=dev),torch.zeros((N,1),device=dev),torch.zeros((N,1),device=dev),torch.zeros((N,3),device=dev),None,torch.zeros((N,),dtype=torch.int32,device=dev),None
    def spherical_harmonics(*args,**kwargs): return args[2] if len(args)>2 and isinstance(args[2],torch.Tensor) else (torch.zeros_like(args[1]) if len(args)>1 and isinstance(args[1],torch.Tensor) else torch.tensor(.0)) # MODIFIED
    logging.warning("Using DUMMY gsplat functions.")

_dust3r_model_per_process = None
def _init_dust3r_worker_process(weights_path:str,device_str:str,dtype_str:str):
    global _dust3r_model_per_process
    if _dust3r_model_per_process is None:
        import multiprocessing
        worker_id=multiprocessing.current_process()._identity[0] if multiprocessing.current_process()._identity else 0
        num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0
        actual_device=f"cuda:{worker_id%num_gpus}" if num_gpus>0 else device_str
        dtype=getattr(torch,dtype_str)
        try: _dust3r_model_per_process=DUSt3R(pretrained=weights_path).to(actual_device,dtype=dtype).eval()
        except Exception: _dust3r_model_per_process=DUSt3R_placeholder(pretrained=weights_path).to(actual_device,dtype=dtype).eval(); logging.warning("Worker using DUMMY DUSt3R.")
def _dust3r_refine_pair_worker_func(feat1_data:dict,feat2_data:dict,config_data:dict) -> Optional[dict]:
    global _dust3r_model_per_process
    if _dust3r_model_per_process is None: return None
    img1=torch.from_numpy(np.array(feat1_data['image_tensor_np'])).permute(2,0,1).unsqueeze(0)
    img2=torch.from_numpy(np.array(feat2_data['image_tensor_np'])).permute(2,0,1).unsqueeze(0)
    id1,id2=feat1_data['image_id'],feat2_data['image_id']
    try:
        dev=next(_dust3r_model_per_process.parameters()).device
        with torch.no_grad(): pc_raw,poses_raw=_dust3r_model_per_process.reconstruct(img1.to(dev),img2.to(dev))
        pc_np,poses_np=pc_raw.cpu().numpy(),poses_raw.cpu().numpy()
        n_pts=min(pc_np.shape[0],500); pc_ltd=pc_np[:n_pts]
        conf=n_pts/500.0; conf=.0 if n_pts<10 else conf # MODIFIED .0
        R1r,t1v=poses_np[0:3],poses_np[3:6].reshape(3,1); R2r,t2v=poses_np[6:9],poses_np[9:12].reshape(3,1)
        R1m,_=cv2.Rodrigues(R1r); R2m,_=cv2.Rodrigues(R2r)
        relR=R2m@R1m.T; relt=t2v-(relR@t1v)
        if conf < config_data['DUST3R_CONFIDENCE_THRESHOLD']: return None
        return {'id1':id1,'id2':id2,'point_cloud':pc_ltd.tolist(),'relative_R':relR.tolist(),'relative_t':relt.tolist(),'confidence':conf}
    except Exception as e: logging.error(f"DUSt3R worker {id1}-{id2}: {e}"); return None

def project_point(pt3d:np.ndarray,R:np.ndarray,t:np.ndarray,K:np.ndarray)->np.ndarray:
    ptc=R@pt3d.reshape(3,1)+t.reshape(3,1)
    if ptc[2,0]<1e-7: return np.array([np.nan,np.nan])
    ptp=K@ptc; return ptp[:2,0]/ptp[2,0]
def build_sparsity_matrix(n_cam:int,n_pt:int,cam_idx:np.ndarray,pt_idx:np.ndarray,fixed_cams:Set[int])->scipy.sparse.lil_matrix:
    n_obs=cam_idx.size; n_params=n_cam*6+n_pt*3
    A=scipy.sparse.lil_matrix((n_obs*2,n_params),dtype=int)
    obs_i=np.arange(n_obs)*2
    for i in range(n_obs):
        ci,pi=cam_idx[i],pt_idx[i]; rx,ry=obs_i[i],obs_i[i]+1
        pt_p_start=n_cam*6+pi*3
        for k_ in range(3): A[rx,pt_p_start+k_]=1; A[ry,pt_p_start+k_]=1
        if ci not in fixed_cams:
            cam_p_start=ci*6
            for k_ in range(6): A[rx,cam_p_start+k_]=1; A[ry,cam_p_start+k_]=1
    return A
def bundle_adjustment_residuals(params:np.ndarray,n_cam:int,n_pt:int,cam_idx:np.ndarray,pt_idx:np.ndarray,obs:np.ndarray,K_dict:Dict[str,np.ndarray],cam_map:Dict[int,str])->np.ndarray:
    cam_p=params[:n_cam*6].reshape(n_cam,6); pts3d_ba=params[n_cam*6:].reshape(n_pt,3)
    res_arr=np.zeros(obs.shape[0]*2,dtype=float)
    for i in range(len(cam_idx)):
        ci,pi=cam_idx[i],pt_idx[i]; img_id=cam_map[ci]
        rvec,tvec=cam_p[ci,:3],cam_p[ci,3:6].reshape(3,1); Rmat,_=cv2.Rodrigues(rvec)
        pt3w=pts3d_ba[pi]; K=K_dict.get(img_id)
        if K is None: K=np.array([[1000.0,0,500.0],[0,1000.0,500.0],[0,0,1]],dtype=np.float32) # MODIFIED .0s
        proj2d=project_point(pt3w,Rmat,tvec,K); obs_xy=obs[i]
        res_arr[i*2:i*2+2]=obs_xy-proj2d if not np.isnan(proj2d).any() else 1e3
    return res_arr

class Hybrid_VGGT_DUSt3R_Pipeline:
    def __init__(self,config:HybridConfig):
        self.cfg=config; self.device=torch.device(config.DEVICE); self.intrinsics_dict=self._load_intrinsics(config.INTRINSICS_FILE)
        os.makedirs(self.cfg.OUTPUT_DIR,exist_ok=True)
        if self.cfg.VISUALIZE: os.makedirs(os.path.join(self.cfg.OUTPUT_DIR,'plots'),exist_ok=True)
        self.model_dtype=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0]>=8 else (torch.float16 if self.device.type=='cuda' and torch.cuda.is_available() else torch.float32)
        try:
            self.vggt=VGGT().to(self.device)
            if os.path.exists(self.cfg.VGGT_WEIGHTS): self.vggt.load_state_dict(torch.load(self.cfg.VGGT_WEIGHTS,map_location=self.device))
            else: logging.error(f"VGGT weights NOT FOUND: {self.cfg.VGGT_WEIGHTS}")
            self.vggt=self.vggt.to(dtype=self.model_dtype).eval()
            if not os.path.exists(self.cfg.DUST3R_WEIGHTS): logging.error(f"DUSt3R weights NOT FOUND: {self.cfg.DUST3R_WEIGHTS}")
        except Exception as e: logging.error(f"Model init failed: {e}"); raise
        self.features:Dict[str,FeatureData]={}; self.scenes:Dict[str,SceneData]={}
    def _load_intrinsics(self,intr_file:Optional[str])->Dict[str,np.ndarray]:
        intr={}; 
        if intr_file and os.path.exists(intr_file):
            try:
                with open(intr_file,'r') as f: data=json.load(f)
                for pth,Kd in data.items():
                    img_id=os.path.basename(pth)
                    if isinstance(Kd,list) and len(Kd)==9: intr[img_id]=np.array(Kd,dtype=np.float32).reshape(3,3)
                    elif isinstance(Kd,list) and len(Kd)==3 and isinstance(Kd[0],list): intr[img_id]=np.array(Kd,dtype=np.float32)
            except Exception as e: logging.warning(f"Intrinsics load failed {intr_file}: {e}")
        return intr
    def _get_intrinsics(self,img_id:str,shape:Tuple[int,int])->np.ndarray:
        if img_id in self.intrinsics_dict: return self.intrinsics_dict[img_id]
        H,W=shape; K=self.cfg.DEFAULT_CAMERA_INTRINSICS.copy(); K[0,2]=W/2.0; K[1,2]=H/2.0; return K # MODIFIED .0
    def _extract_features(self,img_path:str)->Optional[FeatureData]:
        img_id=os.path.basename(img_path)
        try:
            img=cv2.imread(img_path); assert img is not None, f"imread failed {img_path}"
            Ho,Wo=img.shape[:2]; sc=1.0; img_t=img.copy() # MODIFIED .0
            if Ho>self.cfg.MAX_IMAGE_SIZE[1] or Wo>self.cfg.MAX_IMAGE_SIZE[0]:
                sc=min(self.cfg.MAX_IMAGE_SIZE[1]/Ho,self.cfg.MAX_IMAGE_SIZE[0]/Wo)
                img_t=cv2.resize(img,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
            Hp,Wp=img_t.shape[:2]; img_rgb_p=cv2.cvtColor(img_t,cv2.COLOR_BGR2RGB)
            Ko=self._get_intrinsics(img_id,(Ho,Wo)); Ks=Ko.copy()
            if sc!=1.0: Ks[0,0]*=sc; Ks[1,1]*=sc; Ks[0,2]*=sc; Ks[1,2]*=sc # MODIFIED .0
            img_store=(img_rgb_p/255.0).astype(np.float32); img_rgb_samp=img_rgb_p if self.cfg.VISUALIZE else None # MODIFIED .0
            return FeatureData(img_id,img_store,(Hp,Wp),Ks,img_rgb_samp)
        except Exception as e: logging.error(f"Feature extraction {img_path}: {e}"); return None
    def _vggt_initial_pass(self,img_tensors:torch.Tensor,img_ids:List[str])->Tuple[Dict[str,CameraPose],np.ndarray]:
        with torch.no_grad():
            preds=self.vggt(img_tensors.to(self.device,dtype=self.model_dtype))
            poses_r,pc_r,confs_r=preds['poses'].cpu().numpy(),preds['point_cloud'].cpu().numpy(),preds.get('confidence',torch.ones(preds['point_cloud'].shape[0] if preds['point_cloud'].ndim>1 and preds['point_cloud'].shape[0]>0 else 1,device=self.device)).cpu().numpy()
        pose_d={};
        for i,id_ in enumerate(img_ids): rvc,tvc=poses_r[i,:3],poses_r[i,3:].reshape(3,1); Rmat,_=cv2.Rodrigues(rvc); pose_d[id_]=CameraPose(id_,Rmat,tvc,i==0)
        pts_f=np.array([])
        if confs_r.size>0 and pc_r.size>0 and pc_r.ndim==2:
            mask=confs_r > .5; fil_pts = pc_r[mask] if mask.sum()>0 and pc_r.shape[0]==mask.shape[0] else pc_r # MODIFIED .5
            pts_f=fil_pts[:500]
        elif pc_r.size>0 and pc_r.ndim==2: pts_f=pc_r[:500]
        return pose_d,pts_f
    def _build_match_graph(self,pose_d:Dict[str,CameraPose])->nx.Graph:
        G=nx.Graph(); G.add_nodes_from(pose_d.keys()); img_ids=list(pose_d.keys()); pair_args=[]
        for i in range(len(img_ids)):
            for j in range(i+1,len(img_ids)):
                id1,id2=img_ids[i],img_ids[j]
                if id1 in pose_d and id2 in pose_d and np.linalg.norm(pose_d[id1].t-pose_d[id2].t)<self.cfg.DUST3R_PAIR_THRESHOLD and id1 in self.features and id2 in self.features:
                    f1,f2=self.features[id1],self.features[id2]
                    pair_args.append(({'image_id':f1.image_id,'image_tensor_np':f1.image_tensor.tolist(),'image_shape':f1.image_shape,'intrinsics':f1.intrinsics.tolist(),'image_rgb':f1.image_rgb.tolist() if f1.image_rgb is not None else None},
                                      {'image_id':f2.image_id,'image_tensor_np':f2.image_tensor.tolist(),'image_shape':f2.image_shape,'intrinsics':f2.intrinsics.tolist(),'image_rgb':f2.image_rgb.tolist() if f2.image_rgb is not None else None}))
        if not pair_args: return G
        worker_cfg={'DUST3R_CONFIDENCE_THRESHOLD':self.cfg.DUST3R_CONFIDENCE_THRESHOLD}; dt_str=str(self.model_dtype).split('.')[-1]
        with ProcessPoolExecutor(max_workers=self.cfg.MAX_WORKERS,initializer=_init_dust3r_worker_process,initargs=(self.cfg.DUST3R_WEIGHTS,self.cfg.DEVICE,dt_str)) as exe:
            futs={exe.submit(_dust3r_refine_pair_worker_func,f1d,f2d,worker_cfg):(f1d['image_id'],f2d['image_id']) for f1d,f2d in pair_args}
            for fut in as_completed(futs):
                pids=futs[fut]
                try:
                    res_d=fut.result()
                    if res_d: 
                        pc,rR,rt,conf=np.array(res_d['point_cloud']),np.array(res_d['relative_R']),np.array(res_d['relative_t']),res_d['confidence']
                        match_res=MatchData(res_d['id1'],res_d['id2'],pc,rR,rt,conf)
                        if match_res.confidence>=self.cfg.DUST3R_CONFIDENCE_THRESHOLD: G.add_edge(match_res.id1,match_res.id2,weight=match_res.confidence,matches=match_res,count=len(match_res.point_cloud))
                except Exception as e: logging.error(f"DUSt3R result error {pids}: {e}")
        return G
    def _cluster_component(self,comp:set,match_g:nx.Graph)->List[set]:
        if len(comp)<=30: return [comp]
        sub_g=match_g.subgraph(list(comp)); ids_comp=sorted(list(comp)); id_map={id_:i for i,id_ in enumerate(ids_comp)}; n=len(ids_comp)
        adj=np.zeros((n,n),dtype=np.float32)
        for i,id1 in enumerate(ids_comp):
            for id2 in sub_g.neighbors(id1):
                if id2 in id_map: j=id_map[id2]; edge_d=sub_g.get_edge_data(id1,id2); adj[i,j]=edge_d["weight"] if edge_d and "weight" in edge_d else .0; adj[j,i]=adj[i,j] # MODIFIED .0
        dist_m=1.0-adj; np.fill_diagonal(dist_m,.0) # MODIFIED .0
        cl=SklearnDBSCAN(eps=self.cfg.SCENE_EPSILON,min_samples=self.cfg.MIN_IMAGES_PER_SCENE,metric="precomputed",algorithm='brute').fit(dist_m)
        cl_map={lbl:set() for lbl in set(cl.labels_) if lbl!=-1}
        for i,lbl in enumerate(cl.labels_):
            if lbl!=-1: cl_map[lbl].add(ids_comp[i])
        return [c_set for c_set in cl_map.values() if len(c_set)>=self.cfg.MIN_IMAGES_PER_SCENE]
    def _cluster_scenes(self,pose_d:Dict[str,CameraPose])->Dict[str,SceneData]:
        match_g=self._build_match_graph(pose_d); comps=list(nx.connected_components(match_g)); cl_scenes={}; sc_id_ctr=0
        for comp_s in comps:
            if len(comp_s)>=self.cfg.MIN_IMAGES_PER_SCENE:
                sub_sc_sets=self._cluster_component(comp_s,match_g)
                for sub_sc_l in sub_sc_sets:
                    sc_id=f"scene_{sc_id_ctr}"; s_ids=sorted(list(sub_sc_l)); cur_sc_d=SceneData(sc_id,s_ids)
                    for img_id in s_ids:
                        if img_id in pose_d: cur_sc_d.poses[img_id]=pose_d[img_id]
                        for id1_ in s_ids:
                            for id2_ in s_ids:
                                if id1_<id2_ and match_g.has_edge(id1_,id2_): cur_sc_d.matches[(id1_,id2_)]=match_g.get_edge_data(id1_,id2_)['matches']
                    cl_scenes[sc_id]=cur_sc_d; sc_id_ctr+=1
            else:
                for img_id in comp_s: sc_id=f"outlier_isolated_{img_id}"; pose=pose_d.get(img_id,CameraPose(img_id,np.eye(3),np.zeros((3,1)))); cl_scenes[sc_id]=SceneData(sc_id,[img_id],poses={img_id:pose}); sc_id_ctr+=1
        return cl_scenes
    def _select_dust3r_pairs(self,cur_poses:Dict[str,CameraPose])->List[Tuple[str,str]]:
        pairs=set(); ids=list(cur_poses.keys()); 
        if len(ids)<2:return []
        s_ids=sorted(ids)
        for i in range(len(s_ids)-1): id1,id2=s_ids[i],s_ids[i+1]; pairs.add(tuple(sorted((id1,id2)))) # Simplified logic, consider threshold
        # Add random pairs (original logic was more complex, this is simpler for now)
        # rng=np.random.default_rng(42); n_rand=min(10,len(s_ids)//4 if len(s_ids)>=4 else 0)
        # for _ in range(n_rand * 5): # attempts
        #     if len(s_ids)<2 or len(pairs) >= n_rand + (len(s_ids)-1): break
        #     idx1,idx2=rng.choice(len(s_ids),2,replace=False); id1r,id2r=s_ids[idx1],s_ids[idx2]
        #     if np.linalg.norm(cur_poses[id1r].t-cur_poses[id2r].t)<self.cfg.DUST3R_PAIR_THRESHOLD: pairs.add(tuple(sorted((id1r,id2r))))
        return list(pairs)
    def _refine_with_dust3r(self,scenes_d_ref:Dict[str,SceneData],all_init_poses:Dict[str,CameraPose]):
        all_feat_ids=set(self.features.keys()); all_pairs_ref=self._select_dust3r_pairs(all_init_poses)
        if not all_pairs_ref: return
        args_ref=[]
        for p1,p2 in all_pairs_ref:
            if p1 in self.features and p2 in self.features:
                f1,f2=self.features[p1],self.features[p2]
                args_ref.append(({'image_id':f1.image_id,'image_tensor_np':f1.image_tensor.tolist(),'image_shape':f1.image_shape,'intrinsics':f1.intrinsics.tolist(),'image_rgb':f1.image_rgb.tolist() if f1.image_rgb is not None else None},
                                 {'image_id':f2.image_id,'image_tensor_np':f2.image_tensor.tolist(),'image_shape':f2.image_shape,'intrinsics':f2.intrinsics.tolist(),'image_rgb':f2.image_rgb.tolist() if f2.image_rgb is not None else None}))
        if args_ref:
            worker_cfg={'DUST3R_CONFIDENCE_THRESHOLD':self.cfg.DUST3R_CONFIDENCE_THRESHOLD}; dt_str=str(self.model_dtype).split('.')[-1]
            with ProcessPoolExecutor(max_workers=self.cfg.MAX_WORKERS,initializer=_init_dust3r_worker_process,initargs=(self.cfg.DUST3R_WEIGHTS,self.cfg.DEVICE,dt_str)) as exe:
                futs={exe.submit(_dust3r_refine_pair_worker_func,a[0],a[1],worker_cfg):a for a in args_ref}
                for fut_obj in as_completed(futs):
                    p_args=futs[fut_obj]
                    try:
                        res_d=fut_obj.result(); 
                        if not res_d: continue
                        pc,rR,rt,conf=np.array(res_d['point_cloud']),np.array(res_d['relative_R']),np.array(res_d['relative_t']),res_d['confidence']
                        match_obj=MatchData(res_d['id1'],res_d['id2'],pc,rR,rt,conf); id1m,id2m=match_obj.id1,match_obj.id2
                        placed=False
                        for sc_data_it in scenes_d_ref.values():
                            if sc_data_it.scene_id.startswith("outlier_"): continue
                            if id1m in sc_data_it.image_ids and id2m in sc_data_it.image_ids: sc_data_it.matches[(id1m,id2m)]=match_obj; placed=True; break
                        if placed: continue
                        valid_sc=[s for s in scenes_d_ref.values() if not s.scene_id.startswith("outlier_") and len(s.image_ids)>0]
                        if not valid_sc: continue
                        lg_sc_obj=max(valid_sc,key=lambda s:len(s.image_ids)); cur_paired_ids={id_ for sc_lp in scenes_d_ref.values() for id_ in sc_lp.image_ids}
                        new_add_id=None; absR_new=None; abst_new=None
                        if id1m in lg_sc_obj.image_ids and id2m not in cur_paired_ids:
                            ref_id,new_id=id1m,id2m; ref_pose=lg_sc_obj.poses.get(ref_id)
                            if ref_pose: absR_new=ref_pose.R@match_obj.relative_R; abst_new=ref_pose.R@match_obj.relative_t+ref_pose.t; new_add_id=new_id
                        elif id2m in lg_sc_obj.image_ids and id1m not in cur_paired_ids:
                            ref_id,new_id=id2m,id1m; ref_pose=lg_sc_obj.poses.get(ref_id)
                            if ref_pose: Rr_inv=match_obj.relative_R.T; tr_inv=-Rr_inv@match_obj.relative_t; absR_new=ref_pose.R@Rr_inv; abst_new=ref_pose.R@tr_inv+ref_pose.t; new_add_id=new_id
                        if new_add_id:
                            lg_sc_obj.matches[(id1m,id2m)]=match_obj; new_cam_pose=CameraPose(new_add_id,absR_new,abst_new); lg_sc_obj.poses[new_add_id]=new_cam_pose
                            if new_add_id not in lg_sc_obj.image_ids: lg_sc_obj.image_ids.append(new_add_id)
                            all_init_poses[new_add_id]=new_cam_pose
                    except Exception as e: logging.error(f"DUSt3R refine error {p_args[0]['image_id']}-{p_args[1]['image_id']}: {e}")
        cur_all_p_ids={id_ for sc_obj in scenes_d_ref.values() for id_ in sc_obj.image_ids}; still_unp=all_feat_ids-cur_all_p_ids
        for id_unp_f in still_unp:
            out_sc_id=f"outlier_final_{id_unp_f}"; pose_unp=all_init_poses.get(id_unp_f,CameraPose(id_unp_f,np.eye(3),np.zeros((3,1))))
            if not any(s_id.startswith("outlier_") and id_unp_f in s_data.image_ids for s_id,s_data in scenes_d_ref.items()): scenes_d_ref[out_sc_id]=SceneData(out_sc_id,[id_unp_f],poses={id_unp_f:pose_unp})
    def _extract_sift_features_for_scene(self,scene:SceneData)->Dict[str,Dict]: # Simplified
        sift_data = {}
        sift = cv2.SIFT_create(nfeatures=self.cfg.nfeatures,nOctaveLayers=self.cfg.nOctaveLayers,contrastThreshold=self.cfg.contrastThreshold,edgeThreshold=self.cfg.edgeThreshold,sigma=self.cfg.sigma)
        for img_id in scene.image_ids:
            if img_id in self.features and self.features[img_id].image_rgb is not None:
                gray = cv2.cvtColor(self.features[img_id].image_rgb, cv2.COLOR_RGB2GRAY)
                kp, des = sift.detectAndCompute(gray, None)
                if des is not None and len(des) > 0: sift_data[img_id] = {'kp': kp, 'des': des, 'img_shape': gray.shape}
        return sift_data
    def _prepare_ba_input_robust(self,scene:SceneData)->Optional[BundleAdjustmentInput]: # Simplified
        if len(scene.image_ids)<2: return None
        sift_data=self._extract_sift_features_for_scene(scene)
        if len(sift_data)<2: return None
        img_ids=sorted([id_ for id_ in scene.image_ids if id_ in sift_data and id_ in scene.poses])
        if len(img_ids)<2: return None
        cam_map={id_:i for i,id_ in enumerate(img_ids)}; K_dict={id_:self._get_intrinsics(id_,self.features[id_].image_shape) for id_ in img_ids}
        pts3d_list,obs2d_list,cam_indices,pt_indices=[],[],[],[]
        feat_to_3d_map:Dict[Tuple[str,int],int]={}; next_3d_id=0; matcher=cv2.BFMatcher(cv2.NORM_L2)
        
        for i in range(len(img_ids)):
            for j in range(i+1,len(img_ids)):
                id1,id2=img_ids[i],img_ids[j]
                if not (sift_data.get(id1) and sift_data.get(id2)): continue
                kp1,des1=sift_data[id1]['kp'],sift_data[id1]['des']; kp2,des2=sift_data[id2]['kp'],sift_data[id2]['des']
                if des1 is None or des2 is None or len(des1)<2 or len(des2)<2: continue
                
                raw_matches = matcher.knnMatch(des1, des2, k=2)
                good_matches = [m for m, n_ in raw_matches if m.distance < .75 * n_.distance and hasattr(n_, 'distance')] # MODIFIED .75
                if len(good_matches) < self.cfg.min_sift_matches: continue

                pts1=[kp1[m.queryIdx].pt for m in good_matches]; pts2=[kp2[m.trainIdx].pt for m in good_matches]
                kp1_idx=[m.queryIdx for m in good_matches]; kp2_idx=[m.trainIdx for m in good_matches]
                K1,K2=K_dict[id1],K_dict[id2]; R1,t1=scene.poses[id1].R,scene.poses[id1].t; R2,t2=scene.poses[id2].R,scene.poses[id2].t
                
                E,E_mask=cv2.findEssentialMat(np.float32(pts1),np.float32(pts2),K1,method=cv2.RANSAC,prob=self.cfg.ransac_confidence,threshold=self.cfg.ransac_threshold_px)
                if E is None or E_mask is None or E_mask.sum()<8: continue
                pts1_in,pts2_in,kp1_idx_in,kp2_idx_in = np.float32(pts1)[E_mask.ravel()==1],np.float32(pts2)[E_mask.ravel()==1],np.array(kp1_idx)[E_mask.ravel()==1],np.array(kp2_idx)[E_mask.ravel()==1]

                _,R_rel,t_rel,rp_mask=cv2.recoverPose(E,pts1_in,pts2_in,K1)
                if rp_mask is None or rp_mask.sum()<5: continue
                pts1_fin,pts2_fin,kp1_idx_fin,kp2_idx_fin = pts1_in[rp_mask.ravel()>0],pts2_in[rp_mask.ravel()>0],kp1_idx_in[rp_mask.ravel()>0],kp2_idx_in[rp_mask.ravel()>0]

                if len(pts1_fin)<5: continue
                P1=K1@np.hstack((R1,t1)); P2=K2@np.hstack((R2,t2))
                pts4D_h=cv2.triangulatePoints(P1,P2,pts1_fin.T,pts2_fin.T); pts3D_cand=(pts4D_h[:3,:]/(pts4D_h[3,:]+1e-8)).T
                
                for k,pt3w in enumerate(pts3D_cand):
                    pt_c1=R1@pt3w.reshape(3,1)+t1; pt_c2=R2@pt3w.reshape(3,1)+t2
                    if pt_c1[2,0]<1e-2 or pt_c2[2,0]<1e-2: continue
                    rp1,rp2=project_point(pt3w,R1,t1,K1),project_point(pt3w,R2,t2,K2)
                    if np.isnan(rp1).any() or np.isnan(rp2).any() or np.linalg.norm(rp1-pts1_fin[k])>self.cfg.OUTLIER_THRESHOLD or np.linalg.norm(rp2-pts2_fin[k])>self.cfg.OUTLIER_THRESHOLD: continue
                    
                    fkey1,fkey2=(id1,kp1_idx_fin[k]),(id2,kp2_idx_fin[k])
                    gid=feat_to_3d_map.get(fkey1,feat_to_3d_map.get(fkey2))
                    if gid is None: gid=next_3d_id; pts3d_list.append(pt3w); next_3d_id+=1
                    feat_to_3d_map[fkey1]=gid; feat_to_3d_map[fkey2]=gid
                    obs2d_list.extend([pts1_fin[k],pts2_fin[k]]); cam_indices.extend([cam_map[id1],cam_map[id2]]); pt_indices.extend([gid,gid])
        
        if not pts3d_list or not obs2d_list: return None
        cam_p_ba,fix_idx_ba,cam_idx_map_ba=[],set(),{}
        for i,id_ in enumerate(img_ids):
            if id_ in scene.poses: pose=scene.poses[id_]; rv,_=cv2.Rodrigues(pose.R); cam_p_ba.append(np.concatenate((rv.flatten(),pose.t.flatten()))); cam_idx_map_ba[i]=id_; 
            if i<self.cfg.BA_FIX_FIRST_N_CAMERAS: fix_idx_ba.add(i)
        
        final_obs,final_cam_idx,final_pt_idx=[],[],[]
        id_to_ba_idx={id_:ba_i for ba_i,id_ in cam_idx_map_ba.items()}
        for i_o in range(len(obs2d_list)):
            orig_cam_id=img_ids[cam_indices[i_o]] # This was the error source: cam_indices contains cam_map values
            new_ba_idx=id_to_ba_idx.get(orig_cam_id)
            if new_ba_idx is not None and pt_indices[i_o]<len(pts3d_list):
                final_obs.append(obs2d_list[i_o]); final_cam_idx.append(new_ba_idx); final_pt_idx.append(pt_indices[i_o])
        if not final_obs: return None
        return BundleAdjustmentInput(np.array(cam_p_ba),np.array(pts3d_list),np.array(final_pt_idx),np.array(final_cam_idx),np.array(final_obs),K_dict,fix_idx_ba,cam_idx_map_ba)

    def _run_bundle_adjustment(self,scene:SceneData):
        ba_in=self._prepare_ba_input_robust(scene)
        if not ba_in: scene.points_3d=np.array([]) if scene.points_3d is None else scene.points_3d; return
        n_c,n_p=len(ba_in.camera_params),len(ba_in.points_3d)
        if n_p==0 or ba_in.observations.shape[0]==0: scene.points_3d=np.array([]) if scene.points_3d is None else scene.points_3d; return
        x0=np.concatenate([ba_in.camera_params.ravel(),ba_in.points_3d.ravel()])
        sparsity=build_sparsity_matrix(n_c,n_p,ba_in.camera_indices,ba_in.point_indices,ba_in.fixed_camera_indices)
        args=(n_c,n_p,ba_in.camera_indices,ba_in.point_indices,ba_in.observations,ba_in.camera_intrinsics_dict,ba_in.cam_id_map)
        try: res=least_squares(fun=bundle_adjustment_residuals,x0=x0,jac_sparsity=sparsity,method='trf',loss=self.cfg.BA_LOSS,f_scale=self.cfg.BA_F_SCALE,max_nfev=self.cfg.BA_MAX_NFEV,args=args,verbose=self.cfg.BA_VERBOSE)
        except Exception as e: logging.error(f"BA {scene.scene_id} fail: {e}"); scene.points_3d=np.array([]) if scene.points_3d is None else scene.points_3d; return
        mean_r=np.mean(np.abs(res.fun)) if hasattr(res,'fun') and res.fun is not None and res.fun.size>0 else float('inf')
        if res.success and mean_r<self.cfg.BA_RESIDUAL_THRESHOLD: self._update_scene_from_ba(scene,res.x,n_c,n_p,ba_in.cam_id_map); self._filter_outliers(scene,res.x,n_c,n_p,ba_in)
        else: scene.points_3d=ba_in.points_3d.copy() if ba_in.points_3d.size>0 else np.array([])
    def _update_scene_from_ba(self,scene:SceneData,opt_p:np.ndarray,n_c:int,n_p:int,cam_map:Dict[int,str]):
        cam_opt=opt_p[:n_c*6].reshape((n_c,6))
        for c_idx,id_ in cam_map.items():
            if id_ in scene.poses: rvo,tvo=cam_opt[c_idx,:3],cam_opt[c_idx,3:6].reshape(3,1); scene.poses[id_].R,_=cv2.Rodrigues(rvo); scene.poses[id_].t=tvo
        scene.points_3d=opt_p[n_c*6:].reshape((n_p,3)) if n_p>0 else np.array([])
    def _filter_outliers(self,scene:SceneData,opt_p:np.ndarray,n_c:int,n_p:int,ba_in:BundleAdjustmentInput):
        if n_p==0 or ba_in.observations.shape[0]==0: return
        res_filt=bundle_adjustment_residuals(opt_p,n_c,n_p,ba_in.camera_indices,ba_in.point_indices,ba_in.observations,ba_in.camera_intrinsics_dict,ba_in.cam_id_map)
        errs=np.sqrt(res_filt[::2]**2+res_filt[1::2]**2); out_mask=errs>self.cfg.OUTLIER_THRESHOLD
        if not np.any(out_mask): return
        cam_err_counts=defaultdict(int); cam_max_errs=defaultdict(float)
        for i,is_out in enumerate(out_mask):
            if is_out: c_idx=ba_in.camera_indices[i]; cam_err_counts[c_idx]+=1; cam_max_errs[c_idx]=max(cam_max_errs[c_idx],errs[i])
        to_rem={ba_in.cam_id_map.get(c_idx) for c_idx,cnt in cam_err_counts.items() if ba_in.cam_id_map.get(c_idx) and cam_max_errs[c_idx]>self.cfg.OUTLIER_THRESHOLD*2}
        if to_rem:
            scene.image_ids=[id_ for id_ in scene.image_ids if id_ not in to_rem]
            for id_rem in to_rem: 
                if id_rem in scene.poses: del scene.poses[id_rem]
            scene.matches={(id1,id2):m for (id1,id2),m in scene.matches.items() if id1 not in to_rem and id2 not in to_rem}
    def _meshlab_filter(self,pts3d:np.ndarray,sc_id:str)->np.ndarray: # MODIFIED .0
        if pts3d is None or pts3d.shape[0]==0: return np.array([])
        ms=pymeshlab.MeshSet()
        try:
            mesh=pymeshlab.Mesh(vertex_matrix=pts3d); 
            if mesh.vertex_number()==0: return np.array([])
            ms.add_mesh(mesh)
            if ms.current_mesh().face_number()==0 and ms.current_mesh().vertex_number()>3:
                try: ms.apply_filter('generate_surface_reconstruction_ball_pivoting',ballradius=.0) # MODIFIED .0
                except pymeshlab.PyMeshLabException as e: logging.warning(f"MeshLab BP {sc_id}: {e}")
            for filt_n in self.cfg.MESHLAB_FILTERS:
                if ms.current_mesh().vertex_number()==0: break
                try:
                    if "Quadric Edge Collapse Decimation" in filt_n and ms.current_mesh().face_number()>0:
                        tf=min(self.cfg.MESHLAB_TARGET_FACES,ms.current_mesh().face_number()//2)
                        if tf>0: ms.apply_filter(filt_n,targetfacenum=tf,preserveboundary=True)
                    elif "Laplacian Smooth" in filt_n: ms.apply_filter(filt_n,stepsmoothnum=3)
                    else: ms.apply_filter(filt_n)
                except pymeshlab.PyMeshLabException as e: logging.warning(f"MeshLab filt {filt_n} {sc_id}: {e}")
            f_pts=ms.current_mesh().vertex_matrix() if ms.current_mesh() else np.array([])
            if f_pts.size>0: ply_path=os.path.join(self.cfg.OUTPUT_DIR,f"{sc_id}_filtered.ply"); os.makedirs(os.path.dirname(ply_path),exist_ok=True); ms.save_current_mesh(ply_path)
            return f_pts
        except Exception as e: logging.error(f"MeshLab {sc_id}: {e}"); return pts3d
    def _gaussian_splatting(self,scene:SceneData): # MODIFIED .0s
        if not (self.cfg.VISUALIZE and GSPLAT_AVAILABLE and scene.points_3d is not None and scene.points_3d.size>0): return
        pts3d,n_pts=scene.points_3d,len(scene.points_3d); means=torch.tensor(pts3d,dtype=torch.float32,device=self.device)
        colors=[]
        for p3dw in pts3d:
            val_cols=[]
            for id_s in scene.image_ids:
                if id_s not in self.features or id_s not in scene.poses: continue
                feat_d,pose_d=self.features[id_s],scene.poses[id_s]; K,R,t=feat_d.intrinsics,pose_d.R,pose_d.t
                p2d=project_point(p3dw,R,t,K)
                if np.isnan(p2d).any(): continue
                x,y=int(round(p2d[0])),int(round(p2d[1])); H,W=feat_d.image_shape
                if 0<=x<W and 0<=y<H and feat_d.image_rgb is not None: val_cols.append(feat_d.image_rgb[y,x])
            colors.append(np.mean(val_cols,axis=0) if val_cols else np.array([128,128,128],dtype=np.float32))
        cols_np=np.array(colors,dtype=np.float32) if colors else np.random.rand(n_pts,3).astype(np.float32)*255.0
        cols_t=torch.tensor(cols_np/255.0,dtype=torch.float32,device=self.device) # MODIFIED .0
        quats=torch.zeros((n_pts,4),dtype=torch.float32,device=self.device); quats[:,0]=1.0 # MODIFIED .0
        scales=torch.ones((n_pts,3),dtype=torch.float32,device=self.device)*.01; opacities=torch.ones((n_pts,),dtype=torch.float32,device=self.device)*.7 # MODIFIED .01, .7
        if not scene.poses: return
        ex_id=list(scene.poses.keys())[0]; ex_pose=scene.poses[ex_id]
        Rwc,twc=ex_pose.R.T,-ex_pose.R.T@ex_pose.t
        new_twc=twc+Rwc@np.array([.0,.0,1.0]).reshape(3,1)+np.array([.0,.5,.0]).reshape(3,1) # MODIFIED .0s
        w2cR,w2ct=Rwc.T,-Rwc.T@new_twc
        vm=torch.eye(4,dtype=torch.float32,device=self.device); vm[:3,:3]=torch.tensor(w2cR,dtype=torch.float32); vm[:3,3]=torch.tensor(w2ct.flatten(),dtype=torch.float32)
        vms=vm.unsqueeze(0)
        Knp=self._get_intrinsics(ex_id,self.features[ex_id].image_shape) if ex_id in self.features else self.cfg.DEFAULT_CAMERA_INTRINSICS
        rH,rW=self.cfg.GSPLAT_RENDER_SIZE; oW,oH=Knp[0,2]*2,Knp[1,2]*2
        scX,scY=rW/(oW if oW>1e-5 else 1.0),rH/(oH if oH>1e-5 else 1.0) # MODIFIED .0
        Ksc=Knp.copy(); Ksc[0,0]*=scX; Ksc[1,1]*=scY; Ksc[0,2]=rW/2.0; Ksc[1,2]=rH/2.0 # MODIFIED .0
        Ks=torch.tensor(Ksc,dtype=torch.float32,device=self.device).unsqueeze(0)
        bg=torch.tensor([.0,.0,.0],dtype=torch.float32,device=self.device).unsqueeze(0) # MODIFIED .0
        try:
            rend,_,_=rasterization(means=means.unsqueeze(0),quats=quats.unsqueeze(0),scales=scales.unsqueeze(0),opacities=opacities.unsqueeze(0),colors=cols_t.unsqueeze(0),viewmats=vms,Ks=Ks,width=rW,height=rH,sh_degree=0,render_mode="RGB",backgrounds=bg)
            img_out=(np.clip(rend.squeeze(0).permute(1,2,0).cpu().numpy(),0,1)*255).astype(np.uint8)
            out_p=os.path.join(self.cfg.OUTPUT_DIR,'plots',f"{scene.scene_id}_gs_novel_view.png"); os.makedirs(os.path.dirname(out_p),exist_ok=True)
            cv2.imwrite(out_p,cv2.cvtColor(img_out,cv2.COLOR_RGB2BGR))
        except Exception as e: logging.error(f"GS {scene.scene_id} error: {e}")
    def run_pipeline(self)->Optional[pd.DataFrame]:
        start_t=time.time(); all_rows=[]
        root_dir=self.cfg.IMAGE_DIR; 
        if not os.path.isdir(root_dir): logging.error(f"Root dir not found: {root_dir}"); return None
        sc_dirs=sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        if not sc_dirs: logging.error(f"No scenes in {root_dir}"); return None
        for sc_name in sc_dirs:
            img_dir=os.path.join(root_dir,sc_name,'images'); self.features={}; self.scenes={}
            if not os.path.isdir(img_dir): logging.warning(f"No images dir: {img_dir}"); continue
            paths=[os.path.join(img_dir,f) for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            if not paths: logging.warning(f"No images in {img_dir}"); continue
            with ProcessPoolExecutor(max_workers=self.cfg.MAX_WORKERS) as exe: self.features={r.image_id:r for r in exe.map(self._extract_features,paths) if r}
            if not self.features: logging.warning(f"No features for {sc_name}"); continue
            
            all_ids=list(self.features.keys()); init_poses={}; init_pts=np.array([])
            if all_ids:
                for i in range(0,len(all_ids),self.cfg.BATCH_SIZE):
                    b_ids=all_ids[i:i+self.cfg.BATCH_SIZE]
                    b_tens=torch.cat([torch.from_numpy(self.features[id_].image_tensor).permute(2,0,1).unsqueeze(0) for id_ in b_ids],dim=0).float()
                    p_batch,pts_batch=self._vggt_initial_pass(b_tens,b_ids)
                    init_poses.update(p_batch)
                    if init_pts.size==0 and pts_batch.size>0: init_pts=pts_batch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            else: continue # pragma: no cover
            self.scenes=self._cluster_scenes(init_poses)
            for sc_obj in self.scenes.values(): sc_obj.points_3d=init_pts.copy() if init_pts.size>0 else np.array([])
            self._refine_with_dust3r(self.scenes,init_poses)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            for sc_id_ba,sc_data_ba in self.scenes.items():
                if sc_id_ba.startswith("outlier_") or len(sc_data_ba.image_ids)<2: sc_data_ba.points_3d=np.array([]) if sc_data_ba.points_3d is None else sc_data_ba.points_3d; continue
                self._run_bundle_adjustment(sc_data_ba)
            for sc_id_post,sc_data_post in self.scenes.items():
                if sc_data_post.points_3d is not None and sc_data_post.points_3d.size>0 and not sc_id_post.startswith("outlier_"):
                    sc_data_post.points_3d=self._meshlab_filter(sc_data_post.points_3d,f"{sc_name}_{sc_id_post}")
                    if self.cfg.VISUALIZE and GSPLAT_AVAILABLE: self._gaussian_splatting(sc_data_post)
            
            for sc_id_sub,sc_data_sub in self.scenes.items():
                for img_fn_sub in sc_data_sub.image_ids:
                    pose_s=sc_data_sub.poses.get(img_fn_sub); r_str,t_str="nan","nan"
                    if pose_s and not sc_id_sub.startswith("outlier_"): r_str=";".join(map(str,pose_s.R.flatten())); t_str=";".join(map(str,pose_s.t.flatten()))
                    all_rows.append({'dataset':self.cfg.DATASET_NAME,'scene':sc_name,'image':img_fn_sub,'rotation_matrix':r_str,'translation_vector':t_str})
            gc.collect(); 
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        sub_path=os.path.join(self.cfg.OUTPUT_DIR,"submission.csv")
        try: pd.DataFrame(all_rows).to_csv(sub_path,index=False)
        except Exception as e: logging.error(f"Write CSV failed: {e}"); return None # pragma: no cover
        logging.info(f"Pipeline done: {time.time()-start_t:.2f}s. Output: {sub_path}")
        if self.cfg.VALIDATE: logging.info("VALIDATE=True, but no GT mAA logic here.") # pragma: no cover
        return pd.DataFrame(all_rows)
