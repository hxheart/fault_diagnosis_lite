import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))

import numpy as np

from semantics import FactBaseSemanticsDataset
from bgp_semantics import BgpSemantics
from nutils import choose_random


class ConfiguredBgpSemantics:
    """原始配置：中等规模，合成拓扑"""

    def __init__(self):
        self.s = BgpSemantics(labeled_networks=False)

    def sample(self, seed):
        s = np.random.RandomState(seed=seed)

        # 原始网络参数
        real_world_topology = False
        num_nodes = choose_random(range(16, 24), s)
        num_networks = choose_random(list(range(4, 8)), s)
        num_gateway_nodes = 3

        sample_config_overrides = {
            "fwd": {"n": choose_random([8, 10, 12], s)},
            "reachable": {"n": choose_random([4, 5, 6, 7], s)},
            "trafficIsolation": {"n": choose_random(range(10, 30), s)},
        }

        seed = s.randint(0, 1024 * 1024 * 1024)

        return self.s.sample(
            num_nodes=num_nodes,
            real_world_topology=real_world_topology,
            num_networks=num_networks,
            predicate_semantics_sample_config_overrides=sample_config_overrides,
            seed=seed,
            NUM_GATEWAY_NODES=num_gateway_nodes
        )

class LargerScaleBgpSemantics:
    """专门设计来体现edge-type重要性的配置"""
    
    def __init__(self):
        self.s = BgpSemantics(labeled_networks=False)
    
    def sample(self, seed):
        s = np.random.RandomState(seed=seed)
        
        # larger-scale networks
        real_world_topology = False
        num_nodes = choose_random(range(24, 32), s)
        num_networks = choose_random(list(range(10, 16)), s)
        num_gateway_nodes = choose_random([7, 8, 9], s)
        
        # 关键2：更多destinations
        
        # 关键3：约束分布偏向"跨边类型"的约束
        sample_config_overrides = {
            # 大量forwarding约束（跨OSPF和BGP边）
            "fwd": {"n": choose_random([25, 30, 35], s)},
            
            # 较多reachability约束
            "reachable": {"n": choose_random([15, 18, 20], s)},
            
            # 较少isolation约束（这个太简单）
            "trafficIsolation": {"n": choose_random(range(10, 30), s)},
        }
        
        seed = s.randint(0, 1024 * 1024 * 1024)
        return self.s.sample(
            num_nodes=num_nodes,
            real_world_topology=real_world_topology,
            num_networks=num_networks,
            predicate_semantics_sample_config_overrides=sample_config_overrides,
            seed=seed,
            NUM_GATEWAY_NODES=num_gateway_nodes
        )

class RealWorldTopologyBgpSemantics:
    """真实世界拓扑配置"""

    def __init__(self):
        self.s = BgpSemantics(labeled_networks=False)

    def sample(self, seed):
        s = np.random.RandomState(seed=seed)

        # 使用真实世界拓扑
        real_world_topology = True  # 关键差异
        num_networks = choose_random(list(range(4, 8)), s)
        num_gateway_nodes = 3
        num_nodes = choose_random(range(16, 24), s)

        sample_config_overrides = {
            "fwd": {"n": choose_random([8, 10, 12], s)},
            "reachable": {"n": choose_random([4, 5, 6, 7], s)},
            "trafficIsolation": {"n": choose_random(range(10, 30), s)},
        }

        seed = s.randint(0, 1024 * 1024 * 1024)

        return self.s.sample(
            num_nodes=num_nodes,
            real_world_topology=real_world_topology,
            num_networks=num_networks,
            predicate_semantics_sample_config_overrides=sample_config_overrides,
            seed=seed,
            NUM_GATEWAY_NODES=num_gateway_nodes
        )


if __name__ == "__main__":
    # 生成测试数据集的样本数量
    NUM_TEST_SAMPLES = 100  # 可以根据需要调整

    print("=" * 60) # 打印 60 个等号，要不下面看着不方便
    print("开始生成测试数据集...")
    print("=" * 60)

    # 1. 原始配置数据集（已存在，这里仅作示例）
    print("\n[1/3] 生成原始配置数据集 (bgp-ospf-dataset-test-original)...")
    dataset_original = FactBaseSemanticsDataset(
        ConfiguredBgpSemantics(),
        "bgp-ospf-dataset-test-origin",
        num_samples=NUM_TEST_SAMPLES,
        tmp_directory="tmp-bgp-dataset-test-original"
    )
    print(f"✓ 完成！数据集大小: {len(dataset_original)}")

    # 2. 更大规模拓扑数据集
    print("\n[2/3] 生成大规模拓扑数据集 (bgp-ospf-dataset-test-larger)...")
    dataset_larger = FactBaseSemanticsDataset(
        LargerScaleBgpSemantics(),
        "bgp-ospf-dataset-test-larger",
        num_samples=NUM_TEST_SAMPLES,
        tmp_directory="tmp-bgp-dataset-test-larger"
    )
    print(f"✓ 完成！数据集大小: {len(dataset_larger)}")

    # 3. 真实世界拓扑数据集
    print("\n[3/3] 生成真实世界拓扑数据集 (bgp-ospf-dataset-test-real)...")
    dataset_real = FactBaseSemanticsDataset(
        RealWorldTopologyBgpSemantics(),
        "bgp-ospf-dataset-test-real",
        num_samples=NUM_TEST_SAMPLES,
        tmp_directory="tmp-bgp-dataset-test-real"
    )
    print(f"✓ 完成！数据集大小: {len(dataset_real)}")

    print("\n" + "=" * 60)
    print("所有测试数据集生成完成！")
    print("=" * 60)
    print("\n数据集摘要:")
    print(f"  - 原始配置:       {len(dataset_original)} 个样本")
    print(f"  - 大规模拓扑:     {len(dataset_larger)} 个样本")
    print(f"  - 真实世界拓扑:   {len(dataset_real)} 个样本")
    print("\n可以使用 test_models.py 脚本进行测试\n")