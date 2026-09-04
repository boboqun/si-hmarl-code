import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)

class FlatModelBase(TorchModelV2, nn.Module):
    """Base class for flat continuous control without any semantic structural priors."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, self_dim, allies_dim):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        cfg = model_config.get("custom_model_config", {})
        cnn_channels = cfg.get("cnn_channels", 32)
        hidden_dim   = cfg.get("hidden_dim", 128)
        dropout      = cfg.get("dropout", 0.05)
        
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels),   # LayerNorm equivalent, safe for multi-worker
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, cnn_channels),   # LayerNorm equivalent, safe for multi-worker
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        in_dim = cnn_channels + self_dim + allies_dim
        self.fuse_mlp = MLPBlock(in_dim, hidden_dim, hidden_dim, dropout)
        
        self.actor_head = nn.Linear(hidden_dim, num_outputs)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observation"]
        cov = obs["coverage_grid"].float().unsqueeze(1)
        st  = obs["self_state"].float()
        al  = obs["allies_state"].float()
        
        grid_feat = self.grid_cnn(cov).flatten(1)
        fused = self.fuse_mlp(torch.cat([grid_feat, st, al], dim=-1))
        
        logits = self.actor_head(fused)
        self._cur_value = self.critic_head(fused)
        
        return logits, state

    def value_function(self):
        return self._cur_value.squeeze(-1)

class RLlibFlatUAVModel(FlatModelBase):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, self_dim=7, allies_dim=8)

class RLlibFlatUGVModel(FlatModelBase):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, self_dim=5, allies_dim=10)
