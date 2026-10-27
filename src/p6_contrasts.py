"""Descriptive, within-seed contrasts on already-audited development tasks.

No statistical significance or pure-order causal interpretation is implied.
"""
from dataclasses import asdict,replace
from itertools import combinations
from src.p6_schedule import TimingCell,all_cells


def contrast_plan():
    result=[]
    def add(kind,terms,**labels):
        result.append({'kind':kind,**labels,'terms':[{'cell':asdict(c),'weight':w} for c,w in terms]})
    for seed in (11,23):
        for anchor in (c for c in all_cells() if c.seed==seed and c.aux_start is None):
            for a,b in combinations((None,1,3001),2):
                add('auxiliary_timing',[(replace(anchor,aux_start=b),1),(replace(anchor,aux_start=a),-1)],seed=seed,from_aux=a,to_aux=b)
        for aux in (None,1,3001):
            baseline=TimingCell(seed,aux,None,None)
            for mechanism in ('M1','M2','M3-R','M4'):
                cells={t:TimingCell(seed,aux,mechanism,t,t if mechanism=='M2' else None) for t in (1,3001)}
                for t,c in cells.items():
                    add('mechanism_effect',[(c,1),(baseline,-1)],seed=seed,mechanism=mechanism,control='B',timing=t)
                    if mechanism=='M2':
                        add('mechanism_effect',[(c,1),(TimingCell(seed,aux,None,None,t),-1)],seed=seed,mechanism=mechanism,control='same_branch_data',timing=t)
                add('mechanism_timing',[(cells[3001],1),(cells[1],-1)],seed=seed,mechanism=mechanism,aux_start=aux)
            add('data_timing',[(TimingCell(seed,aux,None,None,3001),1),(TimingCell(seed,aux,None,None,1),-1)],seed=seed,aux_start=aux)
        for mechanism in ('M1','M2','M3-R','M4'):
            for a,b in combinations((None,1,3001),2):
                terms=[]
                for aux,t,w in ((b,3001,1),(b,1,-1),(a,3001,-1),(a,1,1)):
                    terms.append((TimingCell(seed,aux,mechanism,t,t if mechanism=='M2' else None),w))
                add('auxiliary_by_mechanism_timing_interaction',terms,seed=seed,mechanism=mechanism,from_aux=a,to_aux=b,control='raw_mechanism_outcomes')
                if mechanism=='M2':
                    adjusted=terms+[(TimingCell(c.seed,c.aux_start,None,None,c.data_start),-w) for c,w in terms]
                    add('auxiliary_by_mechanism_timing_interaction',adjusted,seed=seed,mechanism=mechanism,from_aux=a,to_aux=b,control='same_branch_data_adjusted')
    return result


def summarize_audited(rows):
    lookup={}
    valid=set(all_cells())
    for row in rows:
        c=TimingCell(**row['cell'])
        if c not in valid or c in lookup:raise ValueError('Unknown or duplicate timing cell')
        if row.get('evidence_status')!='fully_audited':raise ValueError('Unaudited outcomes cannot enter contrasts')
        values=row['episode_successes'];resets=row['reset_seeds']
        if len(values)!=32 or any(type(v) is not bool for v in values) or len(resets)!=32 or len(set(resets))!=32:
            raise ValueError('Expected all 32 distinct paired tasks; failures stay in denominator')
        if any(type(v) is not int for v in resets) or resets!=list(range(3000000,3000032)):
            raise ValueError('Expected the frozen 32 development reset seeds in episode order')
        lookup[c]=row
    results=[];missing_contrasts=[]
    for plan in contrast_plan():
        terms=[(TimingCell(**t['cell']),t['weight']) for t in plan['terms']]
        missing=[asdict(c) for c,w in terms if c not in lookup]
        if missing:
            missing_contrasts.append({**plan,'missing_cells':missing});continue
        resets=lookup[terms[0][0]]['reset_seeds']
        if any(lookup[c]['reset_seeds']!=resets for c,w in terms):raise ValueError('Paired task reset seeds differ')
        deltas=[sum(w*int(lookup[c]['episode_successes'][i]) for c,w in terms) for i in range(32)]
        result={**plan,'difference_percentage_points':100*sum(deltas)/32,'task_differences':deltas}
        if len(terms)==2:result.update(paired_wins=sum(d>0 for d in deltas),paired_losses=sum(d<0 for d in deltas))
        results.append(result)
    missing=[asdict(c) for c in all_cells() if c not in lookup]
    return {'status':'complete_descriptive_contrasts' if not missing else 'partial_descriptive_contrasts',
            'audited_cells':len(lookup),'expected_cells':66,'missing_cells':missing,
            'contrasts':results,'unavailable_contrasts':missing_contrasts,
            'limitations':['Two training seeds; results remain per seed.',
                           'Introduction timing also changes exposure duration.',
                           'Shared development tasks; no sealed-test or significance claim.']}
