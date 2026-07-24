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
w,sig,val=compute_weights('data/1641',ALL12)

full=MProbeEstimator(ALL12,weights=[w[p] for p in ALL12],fit_ip=False)
DROP=10
keep=[p for p in ALL12 if p!=DROP]
sub=MProbeEstimator(keep,weights=[w[p] for p in keep],fit_ip=False)
idx=[ALL12.index(p) for p in keep]

flat=np.where((t[:m]>t0+2)&(t[:m]<t1-2))[0][::10]
dfull=[]; ddrop=[]; moves=[]
for k in flat:
    B=np.array([Bcal(p,k) for p in ALL12])
    r1,z1,_=full.shift(B,Ip[k])
    r2,z2,_=sub.shift(B[idx],Ip[k])
    if all(np.isfinite(x) for x in (r1,z1,r2,z2)):
        dfull.append((r1,z1)); ddrop.append((r2,z2))
        moves.append(np.hypot(r2-r1,z2-z1)*1e3)
dfull=np.array(dfull)*1e3; ddrop=np.array(ddrop)*1e3; moves=np.array(moves)
print("Removing GBP%d from the 12-probe fit (shot 1641, %d flat-top samples)"%(DROP,len(moves)))
print("  probe weight share: %.1f%%   (rank by weight among 12)"%(100*w[DROP]/sum(w[p] for p in ALL12)))
print("  displacement move when dropped: median %.2f mm, p90 %.2f mm, max %.2f mm"%(
    np.median(moves),np.percentile(moves,90),moves.max()))
print("  mean |dR| shift %.2f mm, mean |dZ| shift %.2f mm"%(
    np.mean(np.abs(ddrop[:,0]-dfull[:,0])),np.mean(np.abs(ddrop[:,1]-dfull[:,1]))))
print("  full-12  flat-top mean R=%.1f Z=%.1f mm"%(dfull[:,0].mean(),dfull[:,1].mean()))
print("  minus-10 flat-top mean R=%.1f Z=%.1f mm"%(ddrop[:,0].mean(),ddrop[:,1].mean()))
print("  condition number: full %.2f -> minus-10 %.2f"%(full.cond,sub.cond))
