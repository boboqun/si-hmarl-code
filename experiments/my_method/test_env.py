import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from HierarchicalEnv import HierarchicalEnv
from env_defs import RandomMapEnv, GRID_RES

print("Initializing purely from isolated directory...")
map_env = RandomMapEnv(grid_resolution=GRID_RES)
env = HierarchicalEnv(map_env=map_env)
obs, info = env.reset()
print("Isolated Environment Reset OK.")
print("Obs keys:", obs.keys())
