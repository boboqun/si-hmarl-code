# C1 Validation Experiment — Scale-Coupled Baseline

**What this answers.** Both reviewers' top concern: contribution **C1 ("a fixed,
size-invariant policy interface")** is not *directly* tested. MAPPO Flat fails for
hierarchy/sparse-reward reasons (not action scaling), and Standard H-MARL *shares*
SI-HMARL's fixed interface — so neither isolates the interface itself. This
experiment adds the missing direct control: a baseline **identical to SI-HMARL
except its interface is coupled to map size.**

---

## 1. The two interfaces being compared

| | coverage cell | coverage grid | macro-block | macro_k | action dim |
|---|---|---|---|---|---|
| **SI-HMARL (size-invariant)** | `L/20` (scales) | **20×20 fixed** | `L/5` (scales) | **5 fixed** | **27 fixed** |
| **Scale-coupled (this baseline)** | **100 m fixed** | `(L/100)²` (grows) | **~400 m fixed** | `round(L/400)` (grows) | `K²+2` (grows) |

At the 2 km training scale the two are **identical** (K=5, 20×20, 27 actions). They
diverge at every larger scale: SI-HMARL keeps the same shapes (enabling zero-shot
transfer); the scale-coupled baseline's action space and input grid both grow, so it
**cannot transfer** and must be retrained at each scale with a larger network.

Scales used: **2 km (K=5), 3 km (K=8), 4 km (K=10)** (extend if cheap).

---

## 2. Files

- `scalable_models.py` — K-adaptive UAV/UGV networks (the only new code). The UAV
  cross-attention emits **K² spatial tokens** (`AdaptiveAvgPool2d((K,K))` + K²
  position embeddings); both action heads emit **K²+2** logits. `macro_k` is read
  from `custom_model_config`. Run its `__main__` shape smoke-test first.
- `train_scale_coupled.py` — trains the baseline at one scale. `configure_scale()`
  patches the scale constants in **both** `env_defs` and `HierarchicalEnvV2` (the
  latter defines its own `MAP_SIZE`/`MAX_EPISODE_STEPS`), holds `GRID_RES=100`, sets
  `macro_k=round(L/400)`, and passes `macro_k` to the env + models. PPO hyper-params
  mirror the SI-HMARL run.
- `eval_scale_coupled.py` — evaluates a trained checkpoint on the held-out seeds →
  CSV (success / makespan / coarse-cov / deadhead).

---

## 3. How to run (cluster)

```bash
cd experiments/baselines/scale_coupled
python scalable_models.py                              # 1) shape smoke-test (needs torch)

# 2) train the scale-coupled baseline at each scale (separate runs; shapes differ)
python train_scale_coupled.py --scale_km 2.0 --seed 1   # K=5  (architecture control; see note below)
python train_scale_coupled.py --scale_km 3.0 --seed 1   # K=8
python train_scale_coupled.py --scale_km 4.0 --seed 1   # K=10

# 3) evaluate each on seeds 1001–1010
python eval_scale_coupled.py --scale_km 2.0 --ckpt runs/scale_coupled_2000m_seed1
python eval_scale_coupled.py --scale_km 3.0 --ckpt runs/scale_coupled_3000m_seed1
python eval_scale_coupled.py --scale_km 4.0 --ckpt runs/scale_coupled_4000m_seed1
```

**Cost control (the reviewer explicitly allows this).** If 6000-iter runs are too
expensive, reduce `--iters` (e.g. 2000–3000), but apply the **same budget to every
run** *and* to any SI-HMARL reference you compare against. The existing SI-HMARL
checkpoint is 6000 iters, so a reduced-budget baseline must be paired with a
reduced-budget SI-HMARL control — never pit a 3000-iter baseline against the full
6000-iter checkpoint. Convergence plateaus early (see the learning curves), so
2000–3000 iters already capture the trend. **4 km is by far the costliest run**
(the area-scaled episode cap is ~4×, so each iteration collects far more env-steps);
if you can afford only one large scale, **3 km (K=8)** already shows the
K-degradation trend.

> **The 2 km (K=5) run is NOT the SI-HMARL checkpoint you already trained.** At 2 km
> the *interface* matches SI-HMARL (20×20 grid, macro_k=5, 27 actions), but this
> baseline runs the **scalable networks** (`AdaptiveAvgPool2d((K,K))`, freshly
> initialized weights), whereas the 6000-iter SI-HMARL checkpoint used the original
> hard-coded 20→10→5 CNN. Identical I/O shapes, **different architecture and
> weights** — so the 2 km run is an **architecture control**, not a duplicate. Its
> job is to confirm that the scalable net, reduced to K=5, reproduces
> SI-HMARL-level performance; that is what licenses attributing the K=8/10
> degradation to **scale coupling** rather than to the scalable architecture itself.
> If scalable@K=5 came out much worse than SI-HMARL, the larger-K comparison would
> be confounded and the network would need fixing first.
>
> - **Transfer results table (claim b, the K=5 row):** you may reuse SI-HMARL's
>   existing 2 km number and skip retraining at 2 km.
> - **Convergence-vs-K curves (claim a):** train scalable@K=5 too, so all three
>   curves (K=5/8/10) share one architecture and the only variable is K; do **not**
>   substitute SI-HMARL's own curve there (it would inject an architecture difference).
>
> Recommended: run scalable@K=5 at least once — it is the cheapest run — and verify
> it lands near SI-HMARL before trusting the larger-K results.

---

## 4. What to report (two C1 claims)

**(a) Training cost/stability grows with the action space.** Plot or tabulate, vs
macro_k (5/8/10): final converged makespan, success rate, training-time variance,
and iterations-to-threshold. Expected: the scale-coupled interface converges slower
/ to a worse plateau / less stably as K grows — directly supporting the paper's
motivation that scale-coupled action spaces impede convergence.

**(b) Fixed interface transfers; scale-coupled does not.** Put side by side, per
scale:
- **SI-HMARL** — *one* policy trained at 2 km, evaluated **zero-shot** (already in
  Table `tab:scalability`: 100 % to 6 km, etc.);
- **Scale-coupled** — a *separate* policy **retrained at each scale** (this CSV).

Expected: SI-HMARL matches or beats the per-scale-retrained scale-coupled baseline
**without any retraining**, showing the fixed interface is not a handicap — it buys
free transfer that the scale-coupled design can only approach by paying a retraining
(and convergence) cost.

Suggested results table for the paper:

| Scale | SI-HMARL (zero-shot, K=5) | Scale-coupled (retrained, K) |
| | success / makespan | K / success / makespan / train-iters |
| 2 km | … (Table II) | 5 / … / … |
| 3 km | … (tab:scalability) | 8 / … / … |
| 4 km | … (tab:scalability) | 10 / … / … |

---

## 5. Reviewer-response framing

> "To isolate the size-invariant interface (C1) from hierarchical abstraction and
> from reward design, we add a **scale-coupled** control identical to SI-HMARL
> except that the coverage cell and macro-block are held at fixed *physical* size,
> so the observation grid and action space (K²+2) grow with the map. Trained
> per-scale, this baseline's convergence degrades as the action dimension grows
> (Fig./Table X), and a single zero-shot SI-HMARL policy matches or exceeds it
> without retraining — evidence that the fixed interface, not the hierarchy or the
> reward, is what enables stable scaling."

A second, complementary control (the reviewer's *coordinate-option* suggestion —
hierarchy retained but the high level outputs a normalized coordinate instead of a
discrete block) can be added later to separate "hierarchy" from "discrete fixed
interface"; the scale-coupled control above is the most direct test and is the
priority.

---

## 6. Integration caveats (please verify on the cluster)

- Authored without a live torch/RLlib environment — **run the shape smoke-test and a
  1-iteration training smoke test first.** The scripts print the resolved
  `MAP_SIZE / cell / grid / macro_k / action_dim` at startup; confirm they match the
  table in §1 for each scale.
- `configure_scale()` patches module globals in both `env_defs` and
  `HierarchicalEnvV2`. **Verified:** the env derives `self.macro_block_size = MAP_SIZE/macro_k`
  and the reward `MAX_DIST = MAP_SIZE*1.5` (computed inside the reward fn) from the patched
  module-level `MAP_SIZE`; the module-level `MACRO_BLOCK_SIZE` constant is unused by the env,
  so no extra patch is needed.
- The PPO config now mirrors `build_ppo_config` in `hierarchical_train_v2.py`
  **verbatim** (verified against the source): `.api_stack(...False...)` hybrid stack,
  `kl_coeff=0.0` (KL penalty disabled — critical, avoids a confound), `grad_clip=10.0`,
  `batch_mode="complete_episodes"`, `rollout_fragment_length="auto"`, `minibatch_size=512`,
  `num_epochs=5`, `vf_clip_param=500`, `entropy_coeff=0.05`+schedule. Only the env name,
  the two `*_scalable` model names, and `custom_model_config={"macro_k": K}` differ. If your
  installed Ray version renames any argument, reconcile against that file.
- Keep all other knobs (reward weights, $E_{safe}$, swap duration, seeds) identical
  to SI-HMARL so the only variable is the interface.
