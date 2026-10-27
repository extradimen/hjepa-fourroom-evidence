# H-JEPA FourRoom: auxiliary-mechanism evidence (v1.0.0)

Jinpeng Wang, Guangzhou City University of Technology. Contact: extradimen@live.com.

Companion materials for *When Do Auxiliary Mechanisms Help Hierarchical Predictive World Models? A Frozen FourRoom Design-Space and Sealed-Task Study* (manuscript; not an accepted publication).

## Scope and results
P7 compares 11 training schedules across seeds 37, 53, 71, 89: 44 final checkpoints and 1,408 development rollouts. P8 evaluates six continuous arms across the same four seeds on 32 previously unused tasks each: 24 models, 768 audited rollouts. The endpoint is first physical entry into the target radius within 420 actions. Continuous M4 reaches 36/128 sealed task-seed cases versus IDM 25/128, with seed differences +2, +1, +1, +7. This is a descriptive, task-specific observation, not a significance or universal superiority claim. M1/M3-R do not show a consistent four-seed advantage; M2 changes branch-data exposure. P8 does not test early-versus-late training timing.

## Reproduce paper figures and tables
From this repository root, install `requirements-figures.txt`, then run:

    python scripts/build_manuscript_figures.py
    python scripts/build_manuscript_longtables.py

These scripts only read the packaged audit-derived count summaries; they do not run models. Outputs are written under `manuscript/applsci-hjepa-udse/figures/` and `tables/`. The aggregate JSON files preserve the fields needed by these scripts; operational metadata has been omitted. Original source file hashes are recorded in `data/source-provenance.json`.

## Data dictionary and independence
The CSV columns are arm, seed, threshold, first_hits, tasks. `tasks=32` for each row; thresholds 50, 90, 210, 420 are nested summaries of the same trajectories, not independent replications. Seeds identify four independently trained realizations, while B-prefix schedules share ancestors within seed. The public P8 audit index retains checkpoint/task hashes, successful exit indicators, 39 native SHA-item counts and 32 audited task rows per model.

## Implementation and access limitations
`src/` and `scripts/train_p7_*.py` provide project implementation source, together with evaluation entry points and the frozen Linux dependency lock. They are reference source snapshots from the manuscript workspace. Individual historical executions used immutable deployments; this release is not asserted to be a byte-identical copy of every historical deployment. Re-running full training/evaluation additionally requires upstream source trees, model/data inputs, experiment configurations and governed execution setup, which are not all included. Original weights, raw task payloads, native per-task trajectories, private machine paths and operational budget records are excluded. This release supports reproduction of the reported aggregate figures/tables; it is not a self-contained full experiment reproduction image. Contact the author about additional materials and applicable access conditions.

## Third-party software and license
The upstream H-JEPA project is https://github.com/kevinghst/H-JEPA (source snapshot 97054ecaff83e13839195ec4b92d74c08a0127c5). Its license notice is included under `third-party/H-JEPA-LICENSE.txt`; upstream source and dependencies retain their respective terms. No upstream model/data redistribution rights are implied. No new license is assigned here to third-party material. The author-created figure/table scripts and derived aggregate data are provided for research review; a general reuse license for this initial release has not yet been assigned. See the repository for updates.

## Integrity
`SHA256SUMS.json` records all release files except itself. Frozen outcomes were not recomputed or selected for publication. All 24 P8 payload audits passed before release; the initial zero-task failed deployment and authorized versioned recovery remain part of the governed source record.
