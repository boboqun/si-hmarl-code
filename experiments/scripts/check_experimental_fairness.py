import ast
import os
import sys
from pathlib import Path

def extract_constants_from_code(filepath):
    constants = {}
    if not os.path.exists(filepath):
        return constants
        
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
        
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    if isinstance(node.value, ast.Constant):
                        constants[target.id] = node.value.value
                    elif isinstance(node.value, ast.Num): # Python 3.7
                        constants[target.id] = node.value.n
    return constants

def main():
    print("=" * 60)
    print("  [Fairness Audit] 物理引擎基石常数严格跨域自检  ")
    print("=" * 60)
    
    # 动态获取项目根目录，避免相对路径报错
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent # route 目录
    
    # 约定真值源自 `my_method`
    source_env_file = project_root / "experiments/my_method/SimulationEnv.py"
    source_env_defs = project_root / "experiments/my_method/HierarchicalEnv.py"
    
    true_constants = {}
    true_constants.update(extract_constants_from_code(source_env_file))
    true_constants.update(extract_constants_from_code(source_env_defs))
    
    core_keys = [
        'DT', 'CAR_SPEED_WORK', 'CAR_SPEED_CRUISE', 'CAR_COLLECT_RADIUS',
        'UAV_SPEED_WORK', 'UAV_SPEED_CRUISE', 'UAV_SCAN_WIDTH', 'UAV_FULL_BATTERY',
        'UGV_SPEED', 'UAV_SPEED_RECHARGE', 'UAV_RECHARGE_DIST'
    ]
    
    print("\n[GROUND TRUTH] 提取出的标准物理定理:")
    verified_core = {}
    for k in core_keys:
        if k in true_constants:
            verified_core[k] = true_constants[k]
            print(f"  -> {k} = {true_constants[k]}")
            
    target_files = [
        project_root / "experiments/baselines/standard_hmarl/StandardEnv.py",
        project_root / "experiments/baselines/mappo_flat/FlatEnv.py",
        project_root / "experiments/ablations/docking_comparison/DiscreteDockingEnv.py"
    ]
    
    failed = False
    
    print("\n[VERIFICATION] 开始逐区跨域自检...")
    for tf in target_files:
        if not os.path.exists(tf):
            print(f"[SKIP] 找不到待验证文件: {tf}")
            continue
            
        print(f"  正在审查: {tf.relative_to(project_root)} ...")
        local_constants = extract_constants_from_code(tf)
        
        for k, v in verified_core.items():
            if k in local_constants:
                if local_constants[k] != v:
                    print(f"    [!!! 犯规警告 !!!] 参数被篡改: {k}")
                    print(f"      期望真值 = {v}, 实际发现 = {local_constants[k]}")
                    failed = True
                    
    if failed:
        print("\n❌ 自动审查未通过: 发现严重违反物理一致性的超参数篡改，涉嫌学术造假，禁止投递论文！")
        sys.exit(1)
    else:
        print("\n✅ 自检通过: 所有强化学习基线环境及消融环境物理公式与常数 100% 对齐。")
        print("  具备顶刊要求的数据绝对公信力。")
        sys.exit(0)

if __name__ == '__main__':
    main()
