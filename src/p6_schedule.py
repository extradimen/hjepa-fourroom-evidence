"""Pure schedule compiler for the proposed timing study; never launches work."""
from dataclasses import dataclass

@dataclass(frozen=True)
class TimingCell:
    seed: int
    aux_start: int | None
    mechanism: str | None
    mechanism_start: int | None
    data_start: int | None = None

    def __post_init__(self):
        if self.seed not in (11, 23) or self.aux_start not in (None, 1, 3001):
            raise ValueError('Unsupported seed or auxiliary timing')
        if self.mechanism not in (None, 'M1', 'M2', 'M3-R', 'M4'):
            raise ValueError('Only a single registered mechanism is allowed')
        if (self.mechanism is None) != (self.mechanism_start is None):
            raise ValueError('No meaningless timing on absent mechanisms')
        if self.mechanism_start not in (None, 1, 3001) or self.data_start not in (None, 1, 3001):
            raise ValueError('Unsupported timing')
        if self.mechanism == 'M2' and self.data_start != self.mechanism_start:
            raise ValueError('M2 must use paired data from its activation step')
        if self.mechanism not in (None, 'M2') and self.data_start is not None:
            raise ValueError('Branch data only belongs to M2 or its matched control')

    def at(self, step):
        if not 1 <= step <= 4000:
            raise ValueError('Step outside frozen study scope')
        local_step = step if step <= 3000 else step - 3000
        active = self.mechanism if self.mechanism_start is not None and step >= self.mechanism_start else None
        branch = self.data_start is not None and step >= self.data_start
        return dict(step=step, phase=1 if step <= 3000 else 2, local_step=local_step,
                    idm=int(self.aux_start is not None and step >= self.aux_start),
                    mechanism=active, branch_data=branch,
                    create_head=active in ('M2','M3-R') and step == self.mechanism_start,
                    recalibrate_bn_before=step == 3001, freeze_bn=step >= 3001,
                    reset_loader=step in (1,3001), loader_seed=self.seed+4000,
                    dropout_seed=self.seed*100000+local_step,
                    perturb_seed=self.seed+5000+local_step,
                    branch_indices=[(i+(local_step-1)*8)%64 for i in range(8)] if branch else [],
                    reset_existing_optimizer=False)

    def prefix_key(self):
        s=self.at(1)
        return self.seed,s['idm'],s['mechanism'],s['branch_data']

    def p5_reuse_source(self):
        # A candidate only; full implementation and provenance checks remain required.
        if self.aux_start == 1 or self.mechanism_start == 1 or self.data_start == 1:
            return None
        arm=self.mechanism or ('B-same-branch-data' if self.data_start else 'B')
        return f'p5-seed{self.seed}-idm{int(self.aux_start==3001)}-{arm}-v1'


def all_cells():
    cells=[]
    for seed in (11,23):
        for aux in (None,1,3001):
            cells.append(TimingCell(seed,aux,None,None))
            for mechanism in ('M1','M2','M3-R','M4'):
                for timing in (1,3001):
                    cells.append(TimingCell(seed,aux,mechanism,timing,timing if mechanism=='M2' else None))
            for timing in (1,3001):
                cells.append(TimingCell(seed,aux,None,None,timing))
    return cells
