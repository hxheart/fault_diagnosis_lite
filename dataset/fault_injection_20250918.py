import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))

import numpy as np
import torch
from torch_geometric.data import Data
from semantics import FactBaseSemanticsDataset, Semantics
from bgp_semantics import BgpSemantics, compute_forwarding_state
from nutils import choose_random
from factbase import Constant
import copy


class SimpleFaultyBgpSemantics(Semantics):
    """
    简化版故障语义生成器
    避免复杂的配对结构，先把基础功能跑通
    """

    def __init__(self, fault_probability=0.6):
        super().__init__()
        self.normal_semantics = BgpSemantics()
        self.fault_probability = fault_probability

    def sample(self, seed=None):
        """
        生成故障样本
        每个样本要么是正常的，要么包含一个简单的OSPF权重故障
        """
        s = np.random.RandomState(seed=seed)

        # 生成基础配置参数
        num_networks = choose_random(list(range(2, 4)), s)
        num_gateway_nodes = 2
        num_nodes = choose_random(range(6, 10), s)  # 进一步简化

        sample_config_overrides = {
            "fwd": {"n": choose_random([3, 4], s)},  # 减少规范数量
            "reachable": {"n": choose_random([2, 3], s)},
            "trafficIsolation": {"n": choose_random(range(2, 5), s)},
        }

        base_seed = s.randint(0, 1024 * 1024 * 1024)

        # 生成正常配置
        normal_config = self.normal_semantics.sample(
            num_nodes=num_nodes,
            real_world_topology=False,
            num_networks=num_networks,
            predicate_semantics_sample_config_overrides=sample_config_overrides,
            seed=base_seed,
            NUM_GATEWAY_NODES=num_gateway_nodes
        )

        # 决定是否注入故障
        if s.random() < self.fault_probability:
            return self._inject_fault(normal_config, s)
        else:
            # 标记为正常样本
            return self._mark_as_normal(normal_config)

    def _mark_as_normal(self, config):
        """标记为正常样本"""
        # 在to_data()时添加故障标签
        original_to_data = config.to_data

        def enhanced_to_data():
            data = original_to_data()
            # 添加故障标签
            data.is_faulty = torch.tensor([0], dtype=torch.long)  # 0 = normal
            data.fault_type = "normal"
            data.fault_location = "none"
            return data

        config.to_data = enhanced_to_data
        return config

    def _inject_fault(self, normal_config, s):
        """注入OSPF权重故障"""
        try:
            # 重建NetworkX图
            graph, _ = self.normal_semantics.network(normal_config)

            # 找到OSPF边
            ospf_edges = [(src, dst) for src, dst in graph.edges()
                          if graph[src][dst].get('type') == 'ospf']

            fault_info = {
                'is_faulty': True,
                'fault_type': 'ospf_weight',
                'fault_location': 'none'
            }

            if ospf_edges:
                # 随机选择一条边
                faulty_edge_idx = s.randint(0, len(ospf_edges))
                src, dst = ospf_edges[faulty_edge_idx]

                # 修改权重
                original_weight = graph[src][dst]['weight']
                fault_weight = 999

                graph[src][dst]['weight'] = fault_weight
                graph[dst][src]['weight'] = fault_weight

                fault_info['fault_location'] = f"edge_{src}_{dst}"

                # 重新计算转发状态
                compute_forwarding_state(graph)

                # 重建FactBase
                faulty_config = self._rebuild_factbase(graph, normal_config, s)

            else:
                # 没有OSPF边，返回正常配置
                faulty_config = normal_config
                fault_info['is_faulty'] = False

            # 添加故障标签
            return self._add_fault_labels(faulty_config, fault_info)

        except Exception as e:
            print(f"Error injecting fault: {e}")
            # 出错时返回正常配置
            return self._mark_as_normal(normal_config)

    def _rebuild_factbase(self, graph, original_config, s):
        """从修改后的图重建FactBase - 简化版本"""
        from factbase import FactBase

        # 使用原始的predicate declarations
        new_config = FactBase(original_config.predicate_declarations)

        # 复制非规范的配置事实
        config_predicates = {'router', 'network', 'external', 'route_reflector',
                             'ibgp', 'ebgp', 'bgp_route'}

        for fact in original_config.get_all_facts():
            if fact.name in config_predicates:
                new_config.add_fact(fact.name, fact.is_negated, *fact.args)

        # 重新生成connected事实
        ospf_edges = [(src, dst) for src, dst in graph.edges()
                      if graph[src][dst].get('type') == 'ospf']
        processed_edges = set()

        for src, dst in ospf_edges:
            edge_key = tuple(sorted([src, dst]))
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)

            weight = graph[src][dst]['weight']
            new_config.add_fact("connected", False,
                                Constant(f"c{src}"), Constant(f"c{dst}"), weight)

        # 重新生成规范事实
        try:
            for pred_s in self.normal_semantics.predicate_semantics:
                config = self.normal_semantics.sampling_config(pred_s)
                derived = pred_s.sample(graph, random=s, **config)
                for f in derived:
                    new_config.add_fact(f.name, f.is_negated, *f.args)
        except Exception as e:
            print(f"Warning: Error generating specifications: {e}")
            # 如果规范生成失败，至少保留配置事实

        return new_config

    def _add_fault_labels(self, config, fault_info):
        """给FactBase添加故障标签"""
        original_to_data = config.to_data

        def enhanced_to_data():
            data = original_to_data()
            # 添加故障标签
            data.is_faulty = torch.tensor([1 if fault_info['is_faulty'] else 0], dtype=torch.long)
            data.fault_type = fault_info['fault_type']
            data.fault_location = fault_info['fault_location']
            return data

        config.to_data = enhanced_to_data
        return config

    def check(self, p):
        """检查一致性"""
        return self.normal_semantics.check(p)


if __name__ == "__main__":
    print("Testing SimpleFaultyBgpSemantics...")

    # 先测试单个样本生成
    print("Testing single sample generation...")
    semantics = SimpleFaultyBgpSemantics(fault_probability=1.0)  # 确保生成故障样本

    try:
        sample = semantics.sample(seed=42)
        print("Single sample generation successful!")

        # 转换为PyG数据
        data = sample.to_data()
        print(f"Data shape: x={data.x.shape}, edge_index={data.edge_index.shape}")
        print(f"Is faulty: {data.is_faulty}")
        print(f"Fault type: {data.fault_type}")
        print(f"Fault location: {data.fault_location}")

    except Exception as e:
        print(f"Error in single sample: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50 + "\n")

    # 测试数据集生成
    print("Testing dataset generation...")
    try:
        dataset = FactBaseSemanticsDataset(
            SimpleFaultyBgpSemantics(fault_probability=0.5),
            "simple-fault-dataset",
            num_samples=10,  # 先用小数量测试
            tmp_directory="tmp-simple-fault"
        )

        print(f"Dataset size: {len(dataset)}")

        # 统计故障样本数量
        faulty_count = 0
        for i in range(len(dataset)):
            data = dataset[i]
            if hasattr(data, 'is_faulty') and data.is_faulty.item() == 1:
                faulty_count += 1

        print(f"Normal samples: {len(dataset) - faulty_count}")
        print(f"Faulty samples: {faulty_count}")
        print(f"Fault ratio: {faulty_count / len(dataset):.2f}")

    except Exception as e:
        print(f"Error in dataset generation: {e}")
        import traceback

        traceback.print_exc()