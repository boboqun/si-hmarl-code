"""
train_scale_coupled.py — Train the SCALE-COUPLED baseline for the C1 ablation
=============================================================================
Trains a policy that is identical to SI-HMARL EXCEPT its interface is coupled to
map scale:
    * coverage cell size held FIXED at 100 m  -> coverage grid is (L/100)^2 (grows),
    * macro-block size held FIXED at ~400 m   -> macro_k = round(L/400)  (grows),
    * action space = macro_k^2 + 2            (grows).

Contrast: SI-HMARL keeps GRID_RES = L/20 (20x20 fixed) and macro_k = 5 (27 actions)
at every scale, so its interface is size-INVARIANT.

Run one training per scale (the action/obs shapes differ, so weights are NOT shared
across scales):
    python train_scale_coupled.py --scale_km 2.0   # macro_k=5  (== SI-HMARL at train scale)
    python train_scale_coupled.py --scale_km 3.0   # macro_k=8
    python train_scale_coupled.py --scale_km 4.0   # macro_k=10

PPO hyper-parameters mirror the SI-HMARL run (paper Table III / Supplementary).

NOTE: authored without a live RLlib/torch env. Before a full run, do a 1-iteration
smoke test and confirm the scale overrides took effect (the script prints the
resolved MAP_SIZE / GRID / macro_k / action-dim at startup).
"""
import os, sys, argparse, csv, datetime
# Keep Ray rollout workers single-threaded: many workers x torch's default
# thread count (= machine cores, or the host's cores in a container) would spawn
# hundreds of threads thrashing the CPU and crawl the rollout. The driver still
# calls torch.set_num_threads(16) below, which overrides this for SGD.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MY_METHOD = os.path.abspath(os.path.join(HERE, "..", "..", "my_method"))
sys.path.insert(0, MY_METHOD)      # so `import env_defs`, `HierarchicalEnvV2`, models resolve
sys.path.insert(0, HERE)           # so `import scalable_models` resolves

# ----------------------------------------------------------------------
#  (1) SCALE CONFIG — must run BEFORE constructing any env
# ----------------------------------------------------------------------
def configure_scale(scale_km: float, cell_phys_m: float = 400.0):
    """Patch the module-level constants that define the scale-coupled interface.

    HierarchicalEnvV2 defines its OWN MAP_SIZE / MAX_EPISODE_STEPS (it does not
    import them from env_defs), so we patch BOTH modules.
    """
    import env_defs
    import HierarchicalEnvV2 as H

    L = float(scale_km * 1000.0)
    GRID_RES_FIXED = 100.0                       # <-- FIXED cell size => grid GROWS with L
    macro_k = max(1, round(L / cell_phys_m))     # <-- FIXED block size => K GROWS with L

    # env_defs governs RandomMapEnv (physical bounds + coverage-grid shape)
    env_defs.MAP_SIZE  = L
    env_defs.GRID_RES  = GRID_RES_FIXED
    env_defs.GRID_ROWS = int(L / GRID_RES_FIXED)
    env_defs.GRID_COLS = int(L / GRID_RES_FIXED)

    # HierarchicalEnvV2 governs macro_block_size, coordinate normalization, episode cap
    H.MAP_SIZE = L
    if hasattr(H, "GRID_RES"):
        H.GRID_RES = GRID_RES_FIXED
    # Episode horizon. An episode ends on full coverage (cov>=1.0) or this cap.
    # At <=2km the early policy completes the 20x20 grid in ~7.3k steps, so the
    # 30000*s^2 area-scaled cap never bites (kept unchanged for the validated runs).
    # At larger scales the bigger grid is NOT covered by the early policy, so cov<1
    # and episodes run to the cap. With s^2 scaling that cap is 270k@3km / 480k@4km
    # -> NO episode finishes during training -> episode_reward_mean is nan and the
    # policy never sees a terminal signal. Bound it to ~2x the area-scaled natural
    # completion time (~7.3k@2km) so episodes terminate, log, and train; a good
    # policy still has room to reach full coverage well inside the cap.
    if scale_km <= 2.0:
        H.MAX_EPISODE_STEPS = int(30000 * (scale_km ** 2))   # unchanged: 120k @2km
    else:
        # Empirically 4km completes coverage in ~38.8k steps; the earlier 3650*s^2
        # cap (3km=32.8k) was just BELOW 3km's completion time -> 3km pinned at the
        # cap, truncating every episode (neg reward, never learns completion).
        # Linear 15000*s gives room above the ~39k ceiling: 3km->45k, 4km->60k.
        H.MAX_EPISODE_STEPS = int(15000 * scale_km)           # 3km->45k, 4km->60k

    return L, GRID_RES_FIXED, env_defs.GRID_ROWS, macro_k


# ----------------------------------------------------------------------
#  (2) Robust metric extraction (compatible with multiple RLlib versions)
# ----------------------------------------------------------------------
def _extract(result, key):
    """Extract a metric from RLlib result dict, trying multiple paths."""
    for sub in ["env_runners", "sampler_results"]:
        v = result.get(sub, {}).get(key)
        if v is not None:
            return v
    v = result.get(key)
    if v is not None:
        return v
    return float("nan")


def _extract_learner(result, policy_key, metric_key):
    """Extract a per-policy learner metric (loss, entropy, etc.)."""
    learner_info = result.get("info", {}).get("learner", {}).get(policy_key, {})
    # Try different paths for Ray 2.x compatibility
    for path in [
        lambda: learner_info.get("learner_stats", {}).get(metric_key),
        lambda: learner_info.get(metric_key),
    ]:
        v = path()
        if v is not None:
            return v
    # Recursive search as fallback
    def _find(d):
        if not isinstance(d, dict):
            return None
        if metric_key in d:
            return d[metric_key]
        for k, v in d.items():
            r = _find(v)
            if r is not None:
                return r
        return None
    v = _find(learner_info)
    return v if v is not None else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale_km", type=float, required=True)
    ap.add_argument("--cell_phys_m", type=float, default=400.0)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    # Default output under experiments/results/scale_coupled/ (project convention)
    _DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "..", "results", "scale_coupled")
    ap.add_argument("--out", type=str, default=os.path.abspath(_DEFAULT_OUT))
    ap.add_argument("--ckpt_freq", type=int, default=10,
                    help="'latest' checkpoint overwrite frequency (iterations)")
    ap.add_argument("--milestone_freq", type=int, default=100,
                    help="'milestones' permanent checkpoint frequency (iterations)")
    ap.add_argument("--log_freq", type=int, default=1,
                    help="Console log frequency (iterations); CSV is always every iter")
    # ── Checkpoint resume ──
    ap.add_argument("--resume", action="store_true",
                    help="Auto-resume from latest checkpoint in --out directory")
    ap.add_argument("--resume_from", type=str, default="",
                    help="Explicit checkpoint path to resume from")
    ap.add_argument("--start_iter", type=int, default=0,
                    help="Starting iteration number (for correct numbering on resume)")
    # ── Infrastructure (does NOT affect PPO algorithm, only data collection speed) ──
    ap.add_argument("--num_workers", type=int, default=12,
                    help="Number of parallel env runners (Ray workers)")
    ap.add_argument("--envs_per_worker", type=int, default=None,
                    help="Parallel envs per worker. Auto: 4 at <=2km, 1 at >2km "
                         "(long episodes x4 envs buffer too much -> OOM). Override if you wish.")
    ap.add_argument("--batch_mode", type=str, default="complete_episodes",
                    choices=["complete_episodes", "truncate_episodes"],
                    help="complete_episodes matches SI-HMARL (fine at 2km). At large "
                         "scales (3km+) episodes balloon toward the area-scaled cap and "
                         "complete_episodes buffers a whole giant episode per worker -> "
                         "memory blows up / OOM. Use truncate_episodes there: it returns "
                         "bounded fragments, so memory stays flat and iter 1 appears fast.")
    args = ap.parse_args()
    # kl/grad_clip default to the SI-HMARL HEADLINE config (kl=0.2, grad_clip=none).
    # The earlier kl=0.0/grad_clip=10 mirrored a STALE belief; headline = kl=0.2 (verified 2026-06-27).
    KL_COEFF = float(os.environ.get("KL_COEFF", "0.2"))
    _GC = os.environ.get("GRAD_CLIP", "none")
    GRAD_CLIP = None if _GC.lower() in ("none", "null", "") else float(_GC)
    print(f"  kl_coeff={KL_COEFF}  grad_clip={GRAD_CLIP}  (headline: kl=0.2, grad_clip=None)")
    # Long episodes at >2km make 4 envs/worker buffer too much under complete_episodes
    # (each of 48 envs holds a multi-10k-step episode -> OOM). Default to 1 env/worker
    # at large scales unless the user explicitly set it.
    if args.envs_per_worker is None:
        args.envs_per_worker = 1 if args.scale_km > 2.0 else 4

    L, grid_res, grid_rows, macro_k = configure_scale(args.scale_km, args.cell_phys_m)
    n_actions = macro_k * macro_k + 2

    print("=" * 72)
    print(f"  SCALE-COUPLED baseline training")
    print(f"  L={L:.0f} m  cell={grid_res:.0f} m  grid={grid_rows}x{grid_rows}  "
          f"macro_k={macro_k}  action_dim={n_actions}")
    print(f"  (SI-HMARL: cell=L/20, grid=20x20, macro_k=5, action_dim=27)")
    print(f"  iters={args.iters}  seed={args.seed}  ckpt_freq={args.ckpt_freq}")
    print(f"  workers={args.num_workers}  envs/worker={args.envs_per_worker}  "
          f"total_envs={args.num_workers * args.envs_per_worker}")
    if args.resume or args.resume_from:
        print(f"  RESUME MODE: start_iter={args.start_iter}")
    print("=" * 72)

    # imports AFTER configure_scale so the patched constants are picked up
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from env_defs import RandomMapEnv, GRID_RES
    from HierarchicalEnvV2 import HierarchicalEnv
    import scalable_models   # registers RLlibUAVModel_scalable / RLlibUGVModel_scalable

    # ----- env factory: scale-coupled macro_k -----
    # Ray rollout workers are SEPARATE processes: they re-import env_defs /
    # HierarchicalEnvV2 fresh, so the driver's module-level configure_scale()
    # monkey-patch does NOT reach them (only `macro_k`, passed as an argument,
    # crosses the process boundary). That makes the driver declare a 30x30 obs
    # space at 3km while workers still build 20x20 -> "observation outside space".
    # Fix: re-apply the scale patch INSIDE env_creator so every worker builds the
    # env at the correct MAP_SIZE / GRID_RES / grid shape.
    def env_creator(cfg):
        configure_scale(args.scale_km, args.cell_phys_m)
        import env_defs as _ed
        random_map = _ed.RandomMapEnv(grid_resolution=_ed.GRID_RES)
        hier = HierarchicalEnv(map_env=random_map, macro_k=macro_k)
        return ParallelPettingZooEnv(hier)
    tune.register_env("scale_coupled_env", env_creator)

    def policy_mapping_fn(agent_id, *a, **k):
        return "ugv_policy" if agent_id == "ugv_0" else "uav_policy"

    probe = env_creator({})
    uav_obs = probe.observation_space["uav_0"]; uav_act = probe.action_space["uav_0"]
    ugv_obs = probe.observation_space["ugv_0"]; ugv_act = probe.action_space["ugv_0"]

    MCFG = {"macro_k": macro_k, "cnn_channels": 32, "token_dim": 64,
            "state_dim": 64, "nhead": 4, "dropout": 0.05, "hidden_dim": 128}

    # ----- PPO config: EXACTLY mirrors SI-HMARL build_ppo_config -----
    # Only the env name, the two custom_model names, and macro_k differ; every
    # optimization knob is copied from experiments/my_method/hierarchical_train_v2.py
    # so the ONLY experimental variable is the (scale-coupled) interface.
    config = (
        PPOConfig()
        .environment("scale_coupled_env")
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .debugging(seed=args.seed)
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=args.envs_per_worker,
            rollout_fragment_length="auto",
            batch_mode=args.batch_mode,
            sample_timeout_s=1800,   # slower cloud CPU: default 60s times out before a
                                     # full ~7000-step episode completes -> 0 samples
        )
        .training(
            gamma=0.99, lr=3e-4,
            train_batch_size=8000, minibatch_size=512, num_epochs=5,
            grad_clip=GRAD_CLIP, clip_param=0.2, vf_clip_param=500.0, vf_loss_coeff=1.0,
            entropy_coeff=0.05,
            kl_coeff=KL_COEFF,   # match SI-HMARL HEADLINE (kl=0.2, grad_clip=none); env-configurable
            entropy_coeff_schedule=[[0, 0.10], [2_000_000, 0.05],
                                    [8_000_000, 0.01], [20_000_000, 0.001]],
            lambda_=0.95,
        )
        .multi_agent(
            policies={
                "uav_policy": (None, uav_obs, uav_act,
                               {"model": {"custom_model": "RLlibUAVModel_scalable",
                                          "custom_model_config": MCFG}}),
                "ugv_policy": (None, ugv_obs, ugv_act,
                               {"model": {"custom_model": "RLlibUGVModel_scalable",
                                          "custom_model_config": MCFG}}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["ugv_policy", "uav_policy"],
        )
    )

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {
            "PYTHONPATH": f"{MY_METHOD}:{HERE}:" + os.environ.get("PYTHONPATH", ""),
        }},
    )

    # ── Performance: PyTorch threading + Apple Silicon MPS ─────────────
    import torch
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.set_num_threads(16)           # use all cores for SGD (learner)
    torch.set_num_interop_threads(4)    # inter-op parallelism
    print(f"[*] PyTorch threads: intra={torch.get_num_threads()} inter=4")
    if torch.backends.mps.is_available():
        print("[*] MPS (Apple Silicon GPU) available — learner SGD may benefit")

    print("[*] Building PPO algorithm...")
    algo = config.build()

    run_dir = os.path.join(args.out, f"scale_coupled_{int(args.scale_km*1000)}m_seed{args.seed}")
    latest_dir    = os.path.join(run_dir, "latest")
    milestone_dir = os.path.join(run_dir, "milestones")
    os.makedirs(run_dir,        exist_ok=True)
    os.makedirs(latest_dir,     exist_ok=True)
    os.makedirs(milestone_dir,  exist_ok=True)

    # ── Checkpoint resume ──────────────────────────────────────────────
    resume_path = args.resume_from
    if args.resume and not resume_path:
        # Auto-detect latest checkpoint. Two on-disk layouts are possible:
        #  (a) checkpoint files saved DIRECTLY in the dir (algo.save(dir) ->
        #      dir/rllib_checkpoint.json) -- this is what THIS script produces;
        #  (b) older Ray nesting them under dir/checkpoint_*/.
        import glob
        def _ckpt_in(d):
            if os.path.exists(os.path.join(d, "rllib_checkpoint.json")):
                return d                                              # layout (a)
            subs = sorted(glob.glob(os.path.join(d, "checkpoint_*")))  # layout (b)
            return subs[-1] if subs else None
        resume_path = _ckpt_in(latest_dir)
        if not resume_path:                                          # fall back to newest milestone
            for d in sorted(glob.glob(os.path.join(milestone_dir, "iter_*")), reverse=True):
                resume_path = _ckpt_in(d)
                if resume_path:
                    break
        if resume_path:
            print(f"[*] Auto-detected checkpoint: {resume_path}")
        else:
            print("[*] --resume specified but no checkpoint found, training from scratch")

    start_iter = args.start_iter
    if resume_path:
        print(f"[*] Restoring from checkpoint: {resume_path}")
        algo.restore(resume_path)
        # Reset KL coefficient (may have exploded due to masked actions)
        for pid in ["ugv_policy", "uav_policy"]:
            policy = algo.get_policy(pid)
            if hasattr(policy, 'kl_coeff') and policy.kl_coeff > 1.0:
                print(f"  [!] {pid} kl_coeff={policy.kl_coeff:.2e} → reset to 0.0")
                policy.kl_coeff = 0.0
        # Auto-detect start_iter from CSV if not explicitly set
        if start_iter == 0:
            csv_path_check = os.path.join(run_dir, "training_log.csv")
            if os.path.exists(csv_path_check):
                with open(csv_path_check, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        start_iter = int(rows[-1]["iter"])
                        print(f"  [*] Auto-detected start_iter={start_iter} from CSV")
        print(f"[✓] Restored! Continuing from iter {start_iter + 1}")

    # ── Auto-detect start_iter from CSV even without checkpoint ────────
    # (handles case where training was interrupted before first ckpt_freq)
    csv_path = os.path.join(run_dir, "training_log.csv")
    if start_iter == 0 and os.path.exists(csv_path) and (args.resume or args.resume_from):
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                csv_rows = [r for r in reader if r.get("iter", "").strip()]
                if csv_rows:
                    start_iter = int(csv_rows[-1]["iter"])
                    print(f"[*] Auto-detected start_iter={start_iter} from CSV ({len(csv_rows)} rows)")
        except Exception as e:
            print(f"[!] Failed to read CSV for start_iter: {e}")

    # ── CSV log setup ──────────────────────────────────────────────────
    CSV_FIELDS = [
        "iter", "timestamp", "timesteps_total",
        "episode_reward_mean", "episode_reward_min", "episode_reward_max",
        "episode_len_mean",
        "uav_policy_loss", "uav_vf_loss", "uav_entropy", "uav_kl",
        "ugv_policy_loss", "ugv_vf_loss", "ugv_entropy", "ugv_kl",
        "num_episodes", "best_reward",
    ]
    # Append if CSV exists and has content; otherwise write fresh
    csv_has_data = os.path.exists(csv_path) and os.path.getsize(csv_path) > 100
    csv_file = open(csv_path, "a" if csv_has_data else "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not csv_has_data:
        csv_writer.writeheader()
    csv_file.flush()

    print(f"[*] Training log: {csv_path}" + (" (append)" if csv_has_data else ""))
    print(f"[*] Checkpoints:  {run_dir}/")
    remaining = args.iters - start_iter
    print(f"[*] Training {remaining} iterations (iter {start_iter+1} → {args.iters})...\n")

    best_reward = float("-inf")

    for it in range(start_iter + 1, args.iters + 1):
        result = algo.train()

        # ── First-iteration debug: dump result keys to help locate metrics ──
        if it == start_iter + 1:
            print(f"  [DEBUG] result top keys: {list(result.keys())}")
            for _dbg_key in ["sampler_results", "env_runners", "info"]:
                if _dbg_key in result:
                    _val = result[_dbg_key]
                    if isinstance(_val, dict):
                        print(f"  [DEBUG] result['{_dbg_key}'] keys: "
                              f"{list(_val.keys())[:20]}")
            # Dump learner info structure
            learner = result.get("info", {}).get("learner", {})
            for pol_key in ["uav_policy", "ugv_policy"]:
                pol_info = learner.get(pol_key, {})
                if pol_info:
                    print(f"  [DEBUG] learner['{pol_key}'] keys: "
                          f"{list(pol_info.keys())[:15]}")
            print()

        # ── Extract episode-level metrics ──────────────────────────────
        rew_mean = _extract(result, "episode_reward_mean")
        rew_min  = _extract(result, "episode_reward_min")
        rew_max  = _extract(result, "episode_reward_max")
        ep_len   = _extract(result, "episode_len_mean")
        n_eps    = _extract(result, "episodes_this_iter")
        steps    = result.get("timesteps_total", 0)

        # ── Extract per-policy learner metrics ─────────────────────────
        uav_ploss   = _extract_learner(result, "uav_policy", "policy_loss")
        uav_vfloss  = _extract_learner(result, "uav_policy", "vf_loss")
        uav_entropy = _extract_learner(result, "uav_policy", "entropy")
        uav_kl      = _extract_learner(result, "uav_policy", "kl")

        ugv_ploss   = _extract_learner(result, "ugv_policy", "policy_loss")
        ugv_vfloss  = _extract_learner(result, "ugv_policy", "vf_loss")
        ugv_entropy = _extract_learner(result, "ugv_policy", "entropy")
        ugv_kl      = _extract_learner(result, "ugv_policy", "kl")

        trend = "📈" if rew_mean > best_reward else "  "
        if rew_mean > best_reward:
            best_reward = rew_mean

        # ── Write CSV (every iteration) ────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "iter": it, "timestamp": ts, "timesteps_total": steps,
            "episode_reward_mean": rew_mean,
            "episode_reward_min": rew_min, "episode_reward_max": rew_max,
            "episode_len_mean": ep_len,
            "uav_policy_loss": uav_ploss, "uav_vf_loss": uav_vfloss,
            "uav_entropy": uav_entropy, "uav_kl": uav_kl,
            "ugv_policy_loss": ugv_ploss, "ugv_vf_loss": ugv_vfloss,
            "ugv_entropy": ugv_entropy, "ugv_kl": ugv_kl,
            "num_episodes": n_eps, "best_reward": best_reward,
        }
        csv_writer.writerow(row)
        csv_file.flush()

        # ── Console output (every log_freq iterations) ─────────────────
        if it % args.log_freq == 0 or it == 1 or it == args.iters:
            print(
                f"[{ts}] iter {it:>5}/{args.iters} | "
                f"rew: {rew_mean:+10.2f} {trend} | "
                f"len: {ep_len:>8.1f} | "
                f"ploss_uav: {uav_ploss:>8.4f}  ugv: {ugv_ploss:>8.4f} | "
                f"ent_uav: {uav_entropy:>6.4f}  ugv: {ugv_entropy:>6.4f} | "
                f"steps: {steps:>12,}",
                flush=True,
            )

        # ── Dual-track checkpoint save ──────────────────────────────────
        # Track 1: latest/ — overwrite every ckpt_freq, for quick resume
        if it % args.ckpt_freq == 0 or it == args.iters:
            algo.save(latest_dir)
            print(f"  💾 latest/ updated (iter={it})", flush=True)

        # Track 2: milestones/ — permanent save every milestone_freq
        if it % args.milestone_freq == 0 or it == args.iters:
            iter_dir = os.path.join(milestone_dir, f"iter_{it:04d}")
            os.makedirs(iter_dir, exist_ok=True)
            algo.save(iter_dir)
            print(f"  📌 milestones/iter_{it:04d}/ saved", flush=True)

    csv_file.close()

    print("\n" + "=" * 72)
    print(f"  Training complete!")
    print(f"  Best episode_reward_mean = {best_reward:.4f}")
    print(f"  Checkpoints: {run_dir}/")
    print(f"    latest/      ← every {args.ckpt_freq} iters (overwrite)")
    print(f"    milestones/  ← every {args.milestone_freq} iters (permanent)")
    print(f"  Training log: {csv_path}")
    print("=" * 72)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()

