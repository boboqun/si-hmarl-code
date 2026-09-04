"""
smoke_test_local.py — Local shape + gradient smoke test (no Ray/RLlib needed)
=============================================================================
Tests the core UAV/UGV scalable networks at multiple K values.
Verifies: forward shapes, gradient flow, and action masking.

Run:  python3 smoke_test_local.py
"""
import sys, os

# Add my_method to path for hierarchical_models imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "my_method")))

import torch
import torch.nn as nn

# Import ONLY the core networks (not the RLlib wrappers which need ray)
from hierarchical_models import (
    MLPBlock, masked_logits,
    UAV_SELF_DIM, UAV_ALLIES_DIM, UGV_SELF_DIM,
)

# ---------- inline copy of core network classes (avoid importing ray) ----------
# We re-import from the module but skip the ray-dependent parts.
# The safest way: mock ray before importing scalable_models.

import types
ray_mock = types.ModuleType("ray")
ray_mock.rllib = types.ModuleType("ray.rllib")
ray_mock.rllib.models = types.ModuleType("ray.rllib.models")
ray_mock.rllib.models.torch = types.ModuleType("ray.rllib.models.torch")
ray_mock.rllib.models.torch.torch_modelv2 = types.ModuleType("ray.rllib.models.torch.torch_modelv2")
ray_mock.rllib.models.ModelCatalog = type("MockCatalog", (), {"register_custom_model": staticmethod(lambda *a, **k: None)})()

# Create a mock TorchModelV2 class
class MockTorchModelV2:
    def __init__(self, *a, **k): pass
ray_mock.rllib.models.torch.torch_modelv2.TorchModelV2 = MockTorchModelV2

sys.modules["ray"] = ray_mock
sys.modules["ray.rllib"] = ray_mock.rllib
sys.modules["ray.rllib.models"] = ray_mock.rllib.models
sys.modules["ray.rllib.models.torch"] = ray_mock.rllib.models.torch
sys.modules["ray.rllib.models.torch.torch_modelv2"] = ray_mock.rllib.models.torch.torch_modelv2

# Now we can also mock ModelCatalog
mc_mod = types.ModuleType("ray.rllib.models")
mc_mod.ModelCatalog = type("MockCatalog", (), {"register_custom_model": staticmethod(lambda *a, **k: None)})()
sys.modules["ray.rllib.models"] = mc_mod

from scalable_models import UAVSpatialCommanderScalable, UGVNodeDispatcherScalable

print("=" * 70)
print("  Scale-coupled local smoke test (torch only, no Ray)")
print("=" * 70)

all_pass = True

for K, H in [(5, 20), (8, 30), (10, 40)]:
    print(f"\n--- K={K}, coverage_grid={H}x{H}, action_dim={K*K+2} ---")
    B = 4

    uav = UAVSpatialCommanderScalable(macro_k=K)
    ugv = UGVNodeDispatcherScalable(macro_k=K)

    uav_obs = {"coverage_grid": torch.randn(B, H, H),
               "self_state":    torch.randn(B, UAV_SELF_DIM),
               "allies_state":  torch.randn(B, UAV_ALLIES_DIM)}
    ugv_obs = {"coverage_grid": torch.randn(B, H, H),
               "self_state":    torch.randn(B, UGV_SELF_DIM),
               "allies_state":  torch.randn(B, 16)}

    n_act = K * K + 2
    mask = torch.ones(B, n_act)

    # ---- Forward ----
    lu, vu, aw = uav(uav_obs, mask)
    lg, vg     = ugv(ugv_obs, mask)

    ok_shape = (lu.shape == (B, n_act) and lg.shape == (B, n_act)
                and vu.shape == (B, 1) and vg.shape == (B, 1))
    print(f"  [shape]  UAV logits={tuple(lu.shape)}  UGV logits={tuple(lg.shape)}  "
          f"UAV value={tuple(vu.shape)}  UGV value={tuple(vg.shape)}  "
          f"{'OK' if ok_shape else 'FAIL'}")
    if not ok_shape:
        all_pass = False

    # ---- Gradient flow ----
    loss = lu.sum() + lg.sum() + vu.sum() + vg.sum()
    loss.backward()
    uav_grad_ok = all(p.grad is not None and not torch.isnan(p.grad).any()
                      for p in uav.parameters() if p.requires_grad)
    ugv_grad_ok = all(p.grad is not None and not torch.isnan(p.grad).any()
                      for p in ugv.parameters() if p.requires_grad)
    print(f"  [grad]   UAV grads OK={uav_grad_ok}  UGV grads OK={ugv_grad_ok}")
    if not uav_grad_ok or not ugv_grad_ok:
        all_pass = False

    # ---- Action masking ----
    uav2 = UAVSpatialCommanderScalable(macro_k=K)
    mask_strict = torch.zeros(B, n_act)
    mask_strict[:, 0] = 1       # only noop
    mask_strict[:, n_act-1] = 1  # and recharge
    with torch.no_grad():
        lu2, _, _ = uav2(uav_obs, mask_strict)
    probs = torch.softmax(lu2, dim=-1)
    illegal_mass = probs[:, 1:n_act-1].sum(dim=-1).max().item()
    mask_ok = illegal_mass < 1e-4
    print(f"  [mask]   illegal action prob mass = {illegal_mass:.2e}  "
          f"{'OK' if mask_ok else 'FAIL'}")
    if not mask_ok:
        all_pass = False

    # ---- Parameter count ----
    uav_params = sum(p.numel() for p in uav.parameters())
    ugv_params = sum(p.numel() for p in ugv.parameters())
    print(f"  [params] UAV={uav_params:,}  UGV={ugv_params:,}  total={uav_params+ugv_params:,}")

print("\n" + "=" * 70)
if all_pass:
    print("  ALL TESTS PASSED ✅")
else:
    print("  SOME TESTS FAILED ❌")
print("=" * 70)
