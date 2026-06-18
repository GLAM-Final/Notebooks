"""
COMPREHENSIVE EVALUATION: All 8+ Models + KD + Multi-Patient + Ablations
"""
import os, sys, pathlib, warnings, time, pickle, hashlib, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE = pathlib.Path(__file__).resolve().parent.parent
RESULTS = BASE / "testing_models" / "results" / "comprehensive"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE = BASE / "cache" / "wav2vec2_embeddings"
CKPT_BGAT = BASE / "testing_models" / "respiratory_gnn_model_cpu_full.pth"
CKPT_GSAGE = BASE / "testing_models" / "best_respiratory_graphsage_temp_scaled.pth"
CKPT_IGAT = BASE / "testing_models" / "best_improved_30epochs_no_earlystop.pt"
AUDIO = BASE / "ICBHI_final_database"
DIAG = BASE / "ICBHI_final_database" / "important" / "ICBHI_Challenge_diagnosis.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from testing_models.all_models import (
    GINGNN, GCNGNN, GraphTransformerNN, TeacherGATv2, StudentGNN,
    kd_loss, focal_loss, train_gnn_model, evaluate_gnn,
    extract_mfcc_features, train_classical_baselines,
    create_multi_patient_mixtures, evaluate_patient_diagnosis
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, SAGEConv, BatchNorm, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score

FRAME_S = 0.5; FRAME_LEN = int(16000 * FRAME_S)

# ── Models ──
class BaselineGAT(nn.Module):
    def __init__(s, id=768, hd=256, h=4, nl=3, d=0.3):
        super().__init__()
        s.proj=nn.Linear(id,hd); s.conv=nn.ModuleList([GATConv(hd,hd//h,heads=h,dropout=d) for _ in range(nl)])
        s.norm=nn.ModuleList([nn.LayerNorm(hd) for _ in range(nl)])
        s.w=nn.Sequential(nn.Linear(hd,hd//2),nn.ReLU(),nn.Dropout(d),nn.Linear(hd//2,1))
        s.c=nn.Sequential(nn.Linear(hd,hd//2),nn.ReLU(),nn.Dropout(d),nn.Linear(hd//2,1))
    def forward(s,d):
        x,dx=d.x,d.edge_index; x=s.proj(x)
        for c,n in zip(s.conv,s.norm): x=n(F.elu(c(x,dx)))+x; x=F.dropout(x,0.1,s.training)
        return s.w(x).squeeze(-1),s.c(x).squeeze(-1)

class GSage(nn.Module):
    def __init__(s, id=768, hd=256, nl=3, d=0.5):
        super().__init__()
        s.proj=nn.Linear(id,hd); s.conv=nn.ModuleList([SAGEConv(hd,hd) for _ in range(nl)])
        s.norm=nn.ModuleList([nn.BatchNorm1d(hd) for _ in range(nl)])
        s.w=nn.Sequential(nn.Linear(hd,hd//2),nn.ReLU(),nn.Dropout(d),nn.Linear(hd//2,1))
        s.c=nn.Sequential(nn.Linear(hd,hd//2),nn.ReLU(),nn.Dropout(d),nn.Linear(hd//2,1)); s.d=d
    def forward(s,d):
        x,dx=d.x,d.edge_index; x=s.proj(x)
        for c,n in zip(s.conv,s.norm): x=n(F.relu(c(x,dx)))+x; x=F.dropout(x,s.d,s.training)
        return s.w(x).squeeze(-1),s.c(x).squeeze(-1)

class ImpGAT(nn.Module):
    def __init__(s, id=768, hd=256, nl=4, h=4, d=0.4):
        super().__init__()
        s.proj=nn.Sequential(nn.Linear(id,hd),nn.LayerNorm(hd),nn.GELU(),nn.Dropout(d))
        s.gl=nn.ModuleList(); s.ns=nn.ModuleList()
        for i in range(nl):
            if i%2==0: s.gl.append(GATConv(hd,hd//h,heads=h,concat=True,dropout=d))
            else: s.gl.append(SAGEConv(hd,hd))
            s.ns.append(BatchNorm(hd))
        s.ap=nn.Sequential(nn.Linear(hd,hd//4),nn.Tanh(),nn.Linear(hd//4,1))
        s.w=nn.Sequential(nn.Linear(hd,hd//2),nn.LayerNorm(hd//2),nn.GELU(),nn.Dropout(d),nn.Linear(hd//2,1))
        s.c=nn.Sequential(nn.Linear(hd,hd//2),nn.LayerNorm(hd//2),nn.GELU(),nn.Dropout(d),nn.Linear(hd//2,1)); s.d=d
    def forward(s,d):
        x,dx=d.x,d.edge_index; b=d.batch if hasattr(d,"batch") else None; x=s.proj(x)
        res=[]
        for i,(cv,nm) in enumerate(zip(s.gl,s.ns)):
            xn=nm(F.elu(cv(x,dx)))
            if i>0 and i%2==0: xn=xn+res[-1]
            x=F.dropout(xn,s.d,s.training); res.append(x)
        if b is not None:
            at=torch.softmax(s.ap(x).squeeze(-1),dim=0); xg=[]
            for bb in torch.unique(b):
                m=b==bb; xg.append((x[m]*at[m].unsqueeze(-1)).sum(dim=0))
            x=torch.stack(xg,dim=0)
        else:
            at=torch.softmax(s.ap(x).squeeze(-1),dim=0); x=(x*at.unsqueeze(-1)).sum(dim=0,keepdim=True)
        return s.w(x).squeeze(-1),s.c(x).squeeze(-1)

# ── Edge construction ──
def temporal_edges(n):
    if n<2: return torch.empty((2,0),dtype=torch.long)
    e=[[i,i+1] for i in range(n-1)]+[[i+1,i] for i in range(n-1)]
    return torch.tensor(e,dtype=torch.long).t().contiguous()

class FeatureCache:
    """Reuse the same cache key logic as thorough_evaluation.py"""
    def __init__(s):
        s.cache_dir = CACHE
        s._model = None; s._processor = None
    def _lazy_init(s):
        if s._model is None:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            s._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
            s._model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", local_files_only=True)
            s._model.eval()
    def _cache_key(s, audio_path):
        mtime = os.path.getmtime(audio_path)
        return hashlib.md5(f"{audio_path}_{mtime}_0.5".encode()).hexdigest()
    def get_embeddings(s, audio_path):
        key = s._cache_key(audio_path)
        cpath = s.cache_dir / f"{key}.pkl"
        if cpath.exists():
            return pickle.load(open(cpath, "rb"))["embeddings"]
        s._lazy_init()
        import librosa
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        fl = int(0.5 * 16000)
        nf = len(y) // fl
        frames = [y[i*fl:(i+1)*fl] for i in range(nf)]
        embs = []
        with torch.no_grad():
            for i in range(0, len(frames), 24):
                batch = frames[i:i+24]
                inputs = s._processor(batch, sampling_rate=16000, return_tensors="pt", padding=True)
                out = s._model(inputs.input_values)
                embs.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
        emb = np.concatenate(embs, axis=0) if embs else np.zeros((0, 768), dtype=np.float32)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with open(cpath, "wb") as f:
            pickle.dump({"embeddings": emb.astype(np.float32), "audio_duration": len(y)/16000, "num_frames": nf}, f)
        return emb

FEAT_CACHE = FeatureCache()

class DS(torch.utils.data.Dataset):
    def __init__(s, meta, anns):
        s.meta=meta.reset_index(drop=True); s.anns=anns
    def __len__(s): return len(s.meta)
    def __getitem__(s, idx):
        row=s.meta.iloc[idx]
        try:
            emb = FEAT_CACHE.get_embeddings(row["wav_path"])
        except: return None
        if emb.shape[0] == 0: return None
        frames=min(20,emb.shape[0])
        x=emb[:frames]
        if x.shape[0]<20: x=np.concatenate([x,np.zeros((20-x.shape[0],768),dtype=np.float32)])
        a=s.anns.get(row["wav_path"])
        if a is None: return None
        w=np.zeros(20,dtype=np.float32); c=np.zeros(20,dtype=np.float32)
        for i in range(20):
            fs=i*0.5; fe=fs+0.5
            for _,r in a[(a["start"]<fe)&(a["end"]>fs)].iterrows():
                ov=max(0.0,min(fe,r["end"])-max(fs,r["start"]))
                if ov/0.5>=0.3:
                    if int(r["crackle"])==1: c[i]=1.0
                    if int(r["wheeze"])==1: w[i]=1.0
        return Data(x=torch.tensor(x,dtype=torch.float32),
                    edge_index=temporal_edges(x.shape[0]),
                    y_wheeze=torch.tensor(w,dtype=torch.float32),
                    y_crackle=torch.tensor(c,dtype=torch.float32))

def collate(b):
    b=[x for x in b if x is not None]
    return Batch.from_data_list(b) if b else None

# ── Load data ──
print("Loading metadata...")
diag=pd.read_csv(DIAG,sep="\t",header=None,names=["pid","diagnosis"])
diag["pid"]=diag["pid"].astype(str)
dm=dict(zip(diag["pid"],diag["diagnosis"]))
wav_map,ann_map={},{}
for f in os.listdir(AUDIO):
    fp=os.path.join(AUDIO,f)
    if not os.path.isfile(fp): continue
    b,e=os.path.splitext(f)
    if e.lower()==".wav": wav_map[b]=fp
    elif e.lower()==".txt": ann_map[b]=fp
rows=[]
for b in sorted(wav_map):
    if b not in ann_map: continue
    pid=b.split("_")[0]
    rows.append({"file_id":b,"patient_id":pid,"diagnosis":dm.get(pid,"Unknown"),
                  "wav_path":wav_map[b],"ann_path":ann_map[b]})
meta=pd.DataFrame(rows)
pats=meta["patient_id"].unique()
tp,tmp=train_test_split(pats,test_size=0.3,random_state=42)
vp,tp2=train_test_split(tmp,test_size=0.5,random_state=42)
train_m=meta[meta["patient_id"].isin(tp)].reset_index(drop=True)
val_m=meta[meta["patient_id"].isin(vp)].reset_index(drop=True)
test_m=meta[meta["patient_id"].isin(tp2)].reset_index(drop=True)

# Load annotations
anns={}
for _,row in meta.iterrows():
    anns[row["wav_path"]]=pd.read_csv(row["ann_path"],sep="\t",header=None,names=["start","end","crackle","wheeze"])

print(f"Train: {len(train_m)}, Val: {len(val_m)}, Test: {len(test_m)}")

# Filter datasets to only include files with cached embeddings
def filter_ds(ds):
    valid=[i for i in range(len(ds)) if ds[i] is not None]
    return torch.utils.data.Subset(ds, valid)

ds_t_raw=DS(train_m,anns); ds_v_raw=DS(val_m,anns); ds_test_raw=DS(test_m,anns)
ds_t=filter_ds(ds_t_raw); ds_v=filter_ds(ds_v_raw); ds_test=filter_ds(ds_test_raw)
print(f"Valid samples: train={len(ds_t)}, val={len(ds_v)}, test={len(ds_test)}")
lt=DataLoader(ds_t,batch_size=1,shuffle=True)
lv=DataLoader(ds_v,batch_size=1,shuffle=False)
ltest=DataLoader(ds_test,batch_size=1,shuffle=False)

# ── All 8 models ──
models={
    "Baseline_GAT": {"cls":lambda:BaselineGAT(),"ckpt":CKPT_BGAT},
    "GraphSAGE": {"cls":lambda:GSage(),"ckpt":CKPT_GSAGE},
    "Improved_GAT": {"cls":lambda:ImpGAT(),"ckpt":CKPT_IGAT},
    "GIN": {"cls":lambda:GINGNN(),"ckpt":None},
    "GCN": {"cls":lambda:GCNGNN(),"ckpt":None},
    "GraphTransformer": {"cls":lambda:GraphTransformerNN(max_nodes=20),"ckpt":None},
    "Teacher_GATv2": {"cls":lambda:TeacherGATv2(),"ckpt":None},
    "Student_KD": {"cls":lambda:StudentGNN(),"ckpt":None},
}

all_results=[]

# Phase 1: Evaluate pretrained
for name,cfg in models.items():
    if cfg["ckpt"] and cfg["ckpt"].exists():
        print(f"Evaluating {name} (pretrained)...")
        model=cfg["cls"]()
        sd=torch.load(str(cfg["ckpt"]),map_location="cpu")
        if isinstance(sd,dict) and "model_state_dict" in sd: sd=sd["model_state_dict"]
        mk=set(model.state_dict().keys())
        sd={k:v for k,v in sd.items() if k in mk}
        model.load_state_dict(sd,strict=False)
        res=evaluate_gnn(model,ltest,DEVICE)
        all_results.append({"model":name,"status":"pretrained",**res})
        print(f"  Wheeze F1={res['wheeze']['f1']:.4f}, Crackle F1={res['crackle']['f1']:.4f}")

# Phase 2: Train new models
for name in ["GIN","GCN","GraphTransformer","Teacher_GATv2","Student_KD"]:
    print(f"Training {name}...")
    model=models[name]["cls"]()
    t0=time.time()
    model,_=train_gnn_model(model,lt,lv,DEVICE,epochs=30)
    tt=time.time()-t0
    res=evaluate_gnn(model,ltest,DEVICE)
    all_results.append({"model":name,"status":"trained","train_time":tt,**res})
    print(f"  Wheeze F1={res['wheeze']['f1']:.4f}, Crackle F1={res['crackle']['f1']:.4f}, Time={tt:.0f}s")

# Phase 3: KD Student from Teacher
print("Knowledge Distillation: Student from Teacher...")
teacher=models["Teacher_GATv2"]["cls"]()
teacher,_=train_gnn_model(teacher,lt,lv,DEVICE,epochs=30)
student=models["Student_KD"]["cls"]()
t0=time.time()
student,_=train_gnn_model(student,lt,lv,DEVICE,epochs=30,teacher=teacher,kd_alpha=0.7)
tt=time.time()-t0
res=evaluate_gnn(student,ltest,DEVICE)
all_results.append({"model":"Student_KD","status":"trained_kd","train_time":tt,"kd_alpha":0.7,**res})
print(f"  Wheeze F1={res['wheeze']['f1']:.4f}, Crackle F1={res['crackle']['f1']:.4f}")

# Phase 4: Classical baselines (MFCC)
print("\nClassical baselines on MFCC...")
tr_feat,tr_w,tr_c=[],[],[]
for _,row in train_m.iterrows():
    try:
        feat=extract_mfcc_features(row["wav_path"])
        a=anns.get(row["wav_path"])
        tr_feat.append(feat)
        tr_w.append(1.0 if a["wheeze"].sum()>0 else 0.0)
        tr_c.append(1.0 if a["crackle"].sum()>0 else 0.0)
    except: pass
va_feat,va_w,va_c=[],[],[]
for _,row in val_m.iterrows():
    try:
        feat=extract_mfcc_features(row["wav_path"])
        a=anns.get(row["wav_path"])
        va_feat.append(feat)
        va_w.append(1.0 if a["wheeze"].sum()>0 else 0.0)
        va_c.append(1.0 if a["crackle"].sum()>0 else 0.0)
    except: pass
if tr_feat and va_feat:
    for task,tl,vl in [("Wheeze",tr_w,va_w),("Crackle",tr_c,va_c)]:
        cls_res=train_classical_baselines(np.array(tr_feat),np.array(tl),np.array(va_feat),np.array(vl))
        for cn,cr in cls_res.items():
            all_results.append({"model":f"{cn}_{task}","status":"classical",**cr})
            print(f"  {cn} ({task}): F1={cr['f1']:.4f}")

# Phase 5: Temporal ablation
print("\nTemporal ablation on Improved GAT...")
model=models["Improved_GAT"]["cls"]()
sd=torch.load(str(CKPT_IGAT),map_location="cpu")
mk=set(model.state_dict().keys())
sd={k:v for k,v in sd.items() if k in mk}
model.load_state_dict(sd,strict=False)
model.eval()
wp,cl,wt,ct=[],[],[],[]
for b in ltest:
    if b is None: continue
    b=b.to(DEVICE)
    w,c=model(b)
    wp.extend(torch.sigmoid(w).cpu().numpy().ravel())
    cl.extend(torch.sigmoid(c).cpu().numpy().ravel())
    wt.extend(b.y_wheeze.cpu().numpy().ravel())
    ct.extend(b.y_crackle.cpu().numpy().ravel())
wp=np.array(wp); cl=np.array(cl); wt=np.array(wt); ct=np.array(ct)
abl=[]
for ws in [1,2,3,5,7,10]:
    for th in [0.3,0.4,0.5,0.6]:
        pw=(median_filter(wp,size=ws)>=th).astype(int)
        pc=(median_filter(cl,size=ws)>=th).astype(int)
        wf1=f1_score(wt,pw,zero_division=0)
        cf1=f1_score(ct,pc,zero_division=0)
        abl.append({"window":ws,"threshold":th,"wheeze_f1":wf1,"crackle_f1":cf1})
pd.DataFrame(abl).to_csv(RESULTS/"temporal_ablation.csv",index=False)
print(f"  Best: window={max(abl,key=lambda x:x['wheeze_f1']+x['crackle_f1'])['window']}")

# Phase 6: Multi-patient
print("\nMulti-patient synthesis...")
mp_res=[]
for snr in [0,5,10,15]:
    mix=create_multi_patient_mixtures(test_m.head(20),AUDIO,max_patients=3,snr_levels=[snr])
    mp_res.append({"snr_db":snr,"n_mixtures":len(mix),
                   "n_patients":np.mean([m["n_patients"] for m in mix])})
    print(f"  SNR={snr}dB: {len(mix)} mixtures")
pd.DataFrame(mp_res).to_csv(RESULTS/"multi_patient.csv",index=False)

# Save summary
rows=[]
for r in all_results:
    wm=r.get("wheeze",{})
    cm=r.get("crackle",{})
    rows.append({"Model":r.get("model",""),"Status":r.get("status",""),
                 "Wheeze F1":wm.get("f1",0) if isinstance(wm,dict) else 0,
                 "Wheeze AUC":wm.get("roc_auc",0) if isinstance(wm,dict) else 0,
                 "Crackle F1":cm.get("f1",0) if isinstance(cm,dict) else 0,
                 "Crackle AUC":cm.get("roc_auc",0) if isinstance(cm,dict) else 0,
                 "Train Time":r.get("train_time",0)})
df=pd.DataFrame(rows)
df.to_csv(RESULTS/"all_models_summary.csv",index=False)

# Summary figure
fig,ax=plt.subplots(1,2,figsize=(14,5))
mn=[r["model"] for r in all_results if r.get("wheeze",{}).get("f1",0)>0 or r.get("crackle",{}).get("f1",0)>0]
wf=[r["wheeze"]["f1"] for r in all_results if r.get("wheeze",{}).get("f1",0)>0]
cf=[r["crackle"]["f1"] for r in all_results if r.get("crackle",{}).get("f1",0)>0]
x=np.arange(len(mn))
ax[0].bar(x-0.2,wf,0.4,label="Wheeze F1"); ax[0].bar(x+0.2,cf,0.4,label="Crackle F1")
ax[0].set_xticks(x); ax[0].set_xticklabels(mn,rotation=45,ha="right",fontsize=8)
ax[0].set_ylabel("F1"); ax[0].set_title("All Models Comparison"); ax[0].legend(); ax[0].grid(alpha=0.3)
if abl:
    dfa=pd.DataFrame(abl)
    for th in dfa["threshold"].unique():
        sub=dfa[dfa["threshold"]==th]
        ax[1].plot(sub["window"],sub["wheeze_f1"],marker="o",label=f"Wheeze t={th}")
        ax[1].plot(sub["window"],sub["crackle_f1"],marker="s",label=f"Crackle t={th}",linestyle="--")
    ax[1].set_xlabel("Window Size"); ax[1].set_ylabel("F1"); ax[1].set_title("Temporal Ablation"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(RESULTS/"summary.png",dpi=150); plt.close()

print(f"\nResults saved to {RESULTS}")
print("\nGenerated files:")
for f in sorted(RESULTS.rglob("*")):
    if f.is_file(): print(f"  {f.name}")