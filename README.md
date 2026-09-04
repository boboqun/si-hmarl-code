# SI-HMARL — Anonymized Code Release

Code for the journal submission "SI-HMARL: Size-Invariant
Hierarchical Multi-Agent Reinforcement Learning for UAV--UGV Collaborative
Coverage in Large-Scale Agriculture". Checkpoints and result CSVs are not
included; every number in the paper is reproducible from the commands below.

## Install

Python 3.12. `pip install -r requirements.txt` (CPU-only is sufficient; all
paper experiments were trained on CPU).

## Layout

- `experiments/my_method/` — SI-HMARL: environment (`HierarchicalEnvV2.py`,
  `env_defs.py`), policy networks, training driver
  (`hierarchical_train_v2.py`).
- `experiments/baselines/` — Standard H-MARL, MAPPO-Flat, SDA-MAPPO,
  scale-coupled interface variant, and the four planning baselines
  (Porcelli CACPP, AG-CVG, Eker DP, Heuristic MACPP) plus Greedy-Predictive.
- `experiments/scripts/` — evaluation and analysis scripts.

## Training

Headline SI-HMARL (2 km, seed 42, 6,000 iterations, ~124 h on a laptop-class
CPU):

    BATCH_MODE=complete_episodes KL_COEFF=0.2 GRAD_CLIP=none \
    NUM_WORKERS=12 ENVS_PER_RUNNER=4 TRAIN_SEED=42 \
    python experiments/my_method/hierarchical_train_v2.py

The ten-seed robustness farm uses the same command with `TRAIN_SEED` in
{42, 2^13 ... 2^21}. Resuming from a milestone:
`RESUME_FROM=<milestone_dir> START_ITER=<n> NUM_ITER=<6000-n> ...`.

## Reproducing the paper tables

| Paper item | Script |
|---|---|
| Main comparison (Table 3 of the manuscript) | `experiments/scripts/evaluate_performance.py` |
| Equal-coverage @97% (+RS) | `experiments/scripts/eval_equal_coverage.py` |
| Standard H-MARL @97% row | `experiments/scripts/eval_standard_equalcov.py` |
| Training-seed robustness | `experiments/scripts/eval_native_makespan.py` (once per seed checkpoint) |
| Zero-shot scale envelope | `experiments/scripts/eval_envelope_constnodes.py --scale_mult <m>` |
| Scale-coupled baseline | `experiments/baselines/scale_coupled/{train,eval}_scale_coupled.py` |
| SDA-MAPPO row | `experiments/scripts/eval_sda_mappo.py` |
| Energy-model sensitivity (Supp. S2) | `experiments/scripts/eval_energy_sensitivity.py` |
| Fine-grid coverage audit | `experiments/scripts/fine_grid_audit.py` |

Evaluation map seeds are 1001--1010 throughout (1001--1030 for the 30-seed
ablation re-check).

## License

MIT (see LICENSE).
