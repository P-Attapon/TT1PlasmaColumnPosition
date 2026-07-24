import sys, types
try: import tqdm
except:
    t=types.ModuleType('tqdm'); t.tqdm=lambda it=None,**k: it if it is not None else (lambda x:x); sys.modules['tqdm']=t
import numpy as np
from methods_script.toroidal_filament.curation import compute_weights
from methods_script.toroidal_filament.mprobe import MProbeEstimator
from methods_script.toroidal_filament.parameters import calibration_coeff as C, I as Iparam
from methods_script.toroidal_filament.process_probe_data import read_txt, discharge_duration

ALL12=list(range(1,13))
def load(n):
    d=read_txt('data/1641/%s.txt'%n,['t','v']); return d['t'].to_numpy(), d['v'].to_numpy()
t,Ip=load('IP1'); _,It=load('IT1'); _,Ioh=load('IOH1'); _,Iv=load('IV2')
raw={p:load('GBP%dT'%p)[1] for p in ALL12}
m=min(len(t),len(It),*(len(raw[p]) for p in raw))
def Bcal(p,k): return raw[p][k]-C['k%dt'%p]*It[k]-C['k%doh'%p]*Ioh[k]-C['k%dv'%p]*Iv[k]
t0,t1=discharge_duration(t[:m],Ip[:m])
flat=np.where((t[:m]>t0+2)&(t[:m]<t1-2))[0][::10]

for pw in (2.0, 2.5):
    w,sig,val=compute_weights('data/1641',ALL12,power=pw)
    tot=sum(w[p] for p in ALL12)
    full=MProbeEstimator(ALL12,weights=[w[p] for p in ALL12],fit_ip=False)
    # leave-one-out for every probe
    allmoves={}
    for DROP in ALL12:
        keep=[p for p in ALL12 if p!=DROP]
        sub=MProbeEstimator(keep,weights=[w[p] for p in keep],fit_ip=False)
        idx=[ALL12.index(p) for p in keep]
        mv=[]
        for k in flat:
            B=np.array([Bcal(p,k) for p in ALL12])
            r1,z1,_=full.shift(B,Ip[k]); r2,z2,_=sub.shift(B[idx],Ip[k])
            if all(np.isfinite(x) for x in (r1,z1,r2,z2)): mv.append(np.hypot(r2-r1,z2-z1)*1e3)
        allmoves[DROP]=np.median(mv) if mv else np.nan
    print("POWER %.1f  max single weight %.1f%%  cond %.2f"%(pw,100*max(w[p] for p in ALL12)/tot,full.cond))
    print("  GBP10 drop -> %.2f mm ; worst probe: GBP%d %.2f mm ; median-over-probes %.2f mm"%(
        allmoves[10], max(allmoves,key=allmoves.get), max(allmoves.values()), np.median(list(allmoves.values()))))
    print("  weight shares:", {p: round(100*w[p]/tot,1) for p in ALL12})
    print()
