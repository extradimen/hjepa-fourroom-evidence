"""Generate manuscript longtables directly from audited count JSON."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/applsci-hjepa-udse/tables';OUT.mkdir(parents=True,exist_ok=True)
P7=json.loads((ROOT/'outputs/p7-design/p7-stability-final-audit-20261022.json').read_text())
P8=json.loads((ROOT/'outputs/p8-results/p8-sealed-confirmation-analysis-v1.json').read_text())
S=['37','53','71','89'];T=['50','90','210','420']
def esc(s):return s.replace('-to-',r'$\rightarrow$').replace('M3R','M3-R')

p7=[r'\begin{longtable}{llrrrrr}',
 r'\caption{Complete P7 development-set first-hit counts. Each row represents one audited arm--seed final checkpoint evaluated on the same 32 development tasks. The last column is the 420-step count minus the matched-timing IDM count for the same seed; M2 additionally changes branch-data exposure.\label{tab:p7full}}\\',
 r'\toprule',r'Arm & Seed & $H_{50}$ & $H_{90}$ & $H_{210}$ & $H_{420}$ & $\Delta_{\mathrm{IDM},420}$\\',r'\midrule',r'\endfirsthead',
 r'\multicolumn{7}{c}{\tablename\ \thetable{} -- continued}\\',r'\toprule',r'Arm & Seed & $H_{50}$ & $H_{90}$ & $H_{210}$ & $H_{420}$ & $\Delta_{\mathrm{IDM},420}$\\',r'\midrule',r'\endhead',
 r'\midrule\multicolumn{7}{r}{Continued on next page}\\',r'\endfoot',r'\bottomrule',r'\endlastfoot']
for arm, item in P7['arms'].items():
    for j,seed in enumerate(S):
        counts=[item[t]['seed_counts'][seed] for t in T]
        comp=item['420']['paired_difference_vs_matched_IDM'][seed]
        p7.append(f'{esc(arm) if j==0 else ""} & {seed} & '+ ' & '.join(map(str,counts))+f' & {comp:+d}'+r'\\')
    p7.append(r'\addlinespace[2pt]')
p7.append(r'\end{longtable}')
(OUT/'p7_longtable.tex').write_text('\n'.join(p7)+'\n')

p8=[r'\begin{longtable}{llrrrrr}',
 r'\caption{Complete P8 sealed-set first-hit counts. Every checkpoint is evaluated once on the same 32 previously unused tasks. The last column is the within-seed 420-step count difference from continuous IDM. M2 additionally changes branch-data exposure.\label{tab:p8full}}\\',
 r'\toprule',r'Arm & Seed & $H_{50}$ & $H_{90}$ & $H_{210}$ & $H_{420}$ & $\Delta_{\mathrm{IDM},420}$\\',r'\midrule',r'\endfirsthead',
 r'\multicolumn{7}{c}{\tablename\ \thetable{} -- continued}\\',r'\toprule',r'Arm & Seed & $H_{50}$ & $H_{90}$ & $H_{210}$ & $H_{420}$ & $\Delta_{\mathrm{IDM},420}$\\',r'\midrule',r'\endhead',
 r'\midrule\multicolumn{7}{r}{Continued on next page}\\',r'\endfoot',r'\bottomrule',r'\endlastfoot']
X=P8['per_seed_success_counts']; arms=['B_to_B','IDM_to_IDM','M1_to_M1','M2_to_M2','M3R_to_M3R','M4_to_M4']
for arm in arms:
    for j,seed in enumerate(S):
        counts=[X[seed][arm][t] for t in T]
        comp=counts[-1]-X[seed]['IDM_to_IDM']['420']
        p8.append(f'{esc(arm.replace("_to_","-to-")) if j==0 else ""} & {seed} & '+' & '.join(map(str,counts))+f' & {comp:+d}'+r'\\')
    p8.append(r'\addlinespace[2pt]')
p8.append(r'\end{longtable}')
(OUT/'p8_longtable.tex').write_text('\n'.join(p8)+'\n')
print('Wrote longtables to',OUT)
