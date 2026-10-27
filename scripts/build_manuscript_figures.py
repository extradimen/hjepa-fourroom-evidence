"""Render frozen P7/P8 audit counts; no model fitting or new rollouts."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/applsci-hjepa-udse/figures"
OUT.mkdir(parents=True, exist_ok=True)
P7 = json.loads((ROOT / "outputs/p7-design/p7-stability-final-audit-20261022.json").read_text())
P8 = json.loads((ROOT / "outputs/p8-results/p8-sealed-confirmation-analysis-v1.json").read_text())
SEEDS = ["37", "53", "71", "89"]
T = ["50", "90", "210", "420"]
P7_ARMS = list(P7["arms"])
P8_ARMS = ["B_to_B", "IDM_to_IDM", "M1_to_M1", "M2_to_M2", "M3R_to_M3R", "M4_to_M4"]
LABEL = {"B_to_B":"B", "IDM_to_IDM":"IDM", "M1_to_M1":"M1", "M2_to_M2":"M2*", "M3R_to_M3R":"M3-R", "M4_to_M4":"M4"}

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":11,
                     "pdf.fonttype":42, "svg.fonttype":"none"})

def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=.12)
    fig.savefig(OUT / (name + ".png"), dpi=240, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)

# Development matrix: every arm and seed, plus early-minus-late contrast.
matrix = np.array([[P7["arms"][a]["420"]["seed_counts"][s] for s in SEEDS] for a in P7_ARMS])
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 6.4), gridspec_kw={"width_ratios":[1.65,1]})
im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=12, aspect="auto")
ax.set_yticks(range(len(P7_ARMS)), [a.replace("-to-", " $\\to$ ") for a in P7_ARMS])
ax.set_xticks(range(4), ["s" + s for s in SEEDS])
ax.set_title("A. Development: hits by step 420 (of 32)", loc="left", fontweight="bold")
for (i,j), v in np.ndenumerate(matrix):
    ax.text(j, i, str(v), ha="center", va="center", color="white" if v >= 8 else "black", fontsize=8)
fig.colorbar(im, ax=ax, fraction=.025, pad=.02, label="Tasks first reached")
methods=["M1","M2","M3R","M4","IDM"]
diff=[]
for m in methods:
    early="%s-to-%s"%(m,m); late="B-to-%s"%m
    diff.append([P7["arms"][early]["420"]["seed_counts"][s]-P7["arms"][late]["420"]["seed_counts"][s] for s in SEEDS])
diff=np.array(diff)
im2=ax2.imshow(diff,cmap="RdBu",norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=5),aspect="auto")
ax2.set_yticks(range(5),["M1","M2*","M3-R","M4","IDM"])
ax2.set_xticks(range(4),["s"+s for s in SEEDS])
ax2.set_title("B. Development: early $-$ late",loc="left",fontweight="bold")
for (i,j),v in np.ndenumerate(diff):ax2.text(j,i,f"{v:+d}",ha="center",va="center",color="white" if abs(v)>=3 else "black",fontsize=9)
fig.colorbar(im2,ax=ax2,fraction=.045,pad=.04,label="Difference in 32-task hits")
fig.text(.02,.01,"M2* changes branch-data exposure. Same 32 development tasks within each seed; four training seeds.",fontsize=8)
save(fig,"fig1_development_matrix")

# Sealed results: raw hits and paired differences against continuous IDM.
x=P8["per_seed_success_counts"]
raw=np.array([[x[s][a]["420"] for s in SEEDS] for a in P8_ARMS])
delta=raw-raw[1:2]
fig,(ax,ax2)=plt.subplots(1,2,figsize=(10.7,4.6),gridspec_kw={"width_ratios":[1,1]})
im=ax.imshow(raw,cmap="YlGnBu",vmin=0,vmax=12,aspect="auto")
ax.set_xticks(range(4),["s"+s for s in SEEDS]);ax.set_yticks(range(6),[LABEL[a] for a in P8_ARMS])
ax.set_title("A. Sealed tasks: hits by step 420",loc="left",fontweight="bold")
for (i,j),v in np.ndenumerate(raw):ax.text(j,i,str(v),ha="center",va="center",color="white" if v>=8 else "black",fontsize=9)
fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="Of 32 tasks")
im2=ax2.imshow(delta,cmap="RdBu",norm=TwoSlopeNorm(vmin=-7,vcenter=0,vmax=7),aspect="auto")
ax2.set_xticks(range(4),["s"+s for s in SEEDS]);ax2.set_yticks(range(6),[LABEL[a] for a in P8_ARMS])
ax2.set_title("B. Paired difference from IDM",loc="left",fontweight="bold")
for (i,j),v in np.ndenumerate(delta):ax2.text(j,i,f"{v:+d}",ha="center",va="center",color="white" if abs(v)>=5 else "black",fontsize=9)
fig.colorbar(im2,ax=ax2,fraction=.046,pad=.04,label="Difference in 32-task hits")
fig.text(.02,.005,"Four independent training seeds; 32 common sealed tasks per seed. M2* also changes branch-data exposure.",fontsize=8)
save(fig,"fig2_sealed_paired")

# Predeclared first-hit thresholds, pooled descriptive counts only.
fig,ax=plt.subplots(figsize=(8.8,5.1))
markers=['o','s','D','^','v','P']
for idx,a in enumerate(P8_ARMS):
    counts=[sum(x[s][a][t] for s in SEEDS) for t in T]
    ax.plot([int(t) for t in T],counts,marker=markers[idx],linewidth=1.7,markersize=6,label=LABEL[a])
ax.set_xticks([int(t) for t in T]);ax.set_xlim(35,435);ax.set_ylim(0,40)
ax.set_ylabel("First-hit tasks (of 128)");ax.set_xlabel("Executed physical-step threshold")
ax.set_title("Sealed task success at four prespecified thresholds",loc="left",fontweight="bold")
ax.grid(alpha=.25);ax.legend(ncol=3,loc="upper left")
fig.text(.03,.01,"Lines connect observed thresholds for readability; they do not imply interpolation or additional evaluations.",fontsize=8)
save(fig,"fig3_sealed_thresholds")

with (OUT/"p7-counts.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["arm","seed","threshold","first_hits","tasks"])
    for a in P7_ARMS:
        for s in SEEDS:
            for t in T:w.writerow([a,s,t,P7["arms"][a][t]["seed_counts"][s],32])
with (OUT/"p8-counts.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["arm","seed","threshold","first_hits","tasks"])
    for a in P8_ARMS:
        for s in SEEDS:
            for t in T:w.writerow([a,s,t,x[s][a][t],32])
print("Figure and CSV artifacts:", OUT)
