import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))

import numpy as np
import copy
import random

from semantics import FactBaseSemanticsDataset  # 1) FRAMEWORK:  how to create datasets
from bgp_semantics import BgpSemantics          # 3) CORE LOGIC: actual BGP/OSPF simulation
from nutils import choose_random
from factbase import FactBase, Constant


class FaultInjector:
    """
    故障注入器：注入故障后重新运行协议仿真
    """
    def __init__(self, bgp_semantics_instance):
        """
        Args:
            bgp_semantics_instance: BgpSemantics实例，用于重新运行协议仿真
        """
        self.bgp_semantics = bgp_semantics_instance
        self.fault_types = [
            'ospf_weight_corruption',
            'bgp_preference_corruption',
            'missing_ibgp_session',
            'wrong_route_reflector',
        ]

    def inject_fault_and_resimulate(self, original_factbase, fault_type=None, seed=42):
        """
        注入故障并重新运行协议仿真
        """
        random.seed(seed)
        np.random.seed(seed)

        # 1. 深拷贝原始 FactBase
        faulty_factbase = copy.deepcopy(original_factbase)

        # 2. 选择并注入故障
        if fault_type is None:
            fault_type = random.choice(self.fault_types)

        fault_info = {"fault_type": fault_type, "details": []}

        # 3. 根据故障类型修改配置事实
        if fault_type == 'ospf_weight_corruption':
            fault_info = self._inject_ospf_weight_fault(faulty_factbase, fault_info)
        elif fault_type == 'bgp_preference_corruption':
            fault_info = self._inject_bgp_preference_fault(faulty_factbase, fault_info)
        elif fault_type == 'missing_ibgp_session':
            fault_info = self._inject_missing_ibgp_fault(faulty_factbase, fault_info)
        elif fault_type == 'wrong_route_reflector':
            fault_info = self._inject_wrong_rr_fault(faulty_factbase, fault_info)

        # 4. 重新运行协议仿真
        try:
            faulty_factbase = self._resimulate_from_factbase(faulty_factbase)
        except Exception as e:
            print(f"重新仿真失败: {e}")
            # 如果重新仿真失败，返回None表示这个故障样本无效
            return None, fault_info

        return faulty_factbase, fault_info

    def _resimulate_from_factbase(self, modified_factbase):
        """
        从修改后的FactBase重新运行协议仿真
        """
        # 导入计算转发状态的函数
        from bgp_semantics import compute_forwarding_state

        # 1. 从FactBase构建NetworkX图
        network, network_nodes = self.bgp_semantics.network(modified_factbase)

        # 2. 重新运行协议仿真
        compute_forwarding_state(network)

        # 3. 从更新后的网络重新生成规范事实
        updated_factbase = self._rebuild_factbase_with_updated_specs(
            modified_factbase, network, network_nodes
        )

        return updated_factbase

    def _rebuild_factbase_with_updated_specs(self, original_factbase, updated_network, network_nodes):
        """
        基于重新仿真后的网络，重建包含更新规范的FactBase
        """
        # 创建新的FactBase，保留所有配置事实
        new_factbase = FactBase(original_factbase.predicate_declarations)

        # 1. 复制所有非规范事实（配置事实）
        config_predicates = {"router", "network", "external", "route_reflector",
                            "ibgp", "ebgp", "bgp_route", "connected"}

        for fact in original_factbase.get_all_facts():
            if fact.name in config_predicates:
                new_factbase.add_fact(fact.name, fact.is_negated, *fact.args)

        # 2. 基于重新仿真后的网络重新生成规范事实
        s = np.random.RandomState(seed=42)

        # 处理网络映射（如果使用labeled_networks）
        network_mapping = {}
        if hasattr(self.bgp_semantics, 'labeled_networks') and self.bgp_semantics.labeled_networks:
            for n in network_nodes:
                if 'net_label' in updated_network.nodes[n]:
                    network_mapping[updated_network.nodes[n]['net_label']] = n

        # 重新生成规范事实
        for pred_s in self.bgp_semantics.predicate_semantics:
            config = getattr(self.bgp_semantics, 'predicate_semantics_sample_config', {}).get(
                pred_s.predicate_name, {}
            )
            derived = pred_s.sample(updated_network, random=s, **config)

            # 处理labeled_networks
            if hasattr(self.bgp_semantics, 'labeled_networks') and self.bgp_semantics.labeled_networks:
                for f in derived:
                    def network_constants_to_label(a):
                        if type(a) is Constant and a.name in network_mapping.keys():
                            return network_mapping[a.name]
                        return a
                    f.args = [network_constants_to_label(a) for a in f.args]

            # 添加规范事实
            for f in derived:
                new_factbase.add_fact(f.name, f.is_negated, *f.args)

        return new_factbase

    def _inject_ospf_weight_fault(self, factbase, fault_info):
        """注入OSPF权重故障"""
        all_facts = factbase.get_all_facts()
        connected_facts = [fact for fact in all_facts if fact.name == "connected"]

        if not connected_facts:
            fault_info["details"].append("No connected facts found")
            return fault_info

        target_fact = random.choice(connected_facts)
        src, dst, original_weight = target_fact.args
        original_weight = int(original_weight)

        # 生成故障权重
        wrong_weight = min(original_weight * random.randint(5, 20), 999)

        # 移除原始事实，添加故障事实
        factbase.remove_fact(target_fact.name, target_fact.is_negated, *target_fact.args)
        factbase.add_fact("connected", False, src, dst, wrong_weight)

        fault_info["details"].append({
            "type": "ospf_weight_corruption",
            "src": str(src),
            "dst": str(dst),
            "original_weight": original_weight,
            "corrupted_weight": wrong_weight
        })
        return fault_info

    def _inject_bgp_preference_fault(self, factbase, fault_info):
        """注入BGP偏好值故障"""
        all_facts = factbase.get_all_facts()
        bgp_route_facts = [fact for fact in all_facts if fact.name == "bgp_route"]

        if not bgp_route_facts:
            fault_info["details"].append("No BGP route facts found")
            return fault_info

        target_fact = random.choice(bgp_route_facts)
        gateway, network, local_pref, med, origin, as_path_len, weight, as_num = target_fact.args

        original_local_pref = int(local_pref)
        corrupted_local_pref = max(1, original_local_pref // 2)

        factbase.remove_fact(target_fact.name, target_fact.is_negated, *target_fact.args)
        factbase.add_fact("bgp_route", False, gateway, network, corrupted_local_pref,
                         med, origin, as_path_len, weight, as_num)

        fault_info["details"].append({
            "type": "bgp_preference_corruption",
            "gateway": str(gateway),
            "network": str(network),
            "original_local_pref": original_local_pref,
            "corrupted_local_pref": corrupted_local_pref
        })
        return fault_info

    def _inject_missing_ibgp_fault(self, factbase, fault_info):
        """注入缺失的iBGP会话故障"""
        all_facts = factbase.get_all_facts()
        ibgp_facts = [fact for fact in all_facts if fact.name == "ibgp"]

        if not ibgp_facts:
            fault_info["details"].append("No iBGP facts found")
            return fault_info

        target_fact = random.choice(ibgp_facts)
        factbase.remove_fact(target_fact.name, target_fact.is_negated, *target_fact.args)

        fault_info["details"].append({
            "type": "missing_ibgp_session",
            "removed_session": f"ibgp({target_fact.args[0]},{target_fact.args[1]})"
        })
        return fault_info

    def _inject_wrong_rr_fault(self, factbase, fault_info):
        """注入错误的路由反射器配置故障"""
        all_facts = factbase.get_all_facts()
        rr_facts = [fact for fact in all_facts if fact.name == "route_reflector"]

        if not rr_facts:
            fault_info["details"].append("No route reflector facts found")
            return fault_info

        target_rr = random.choice(rr_facts)
        rr_node = target_rr.args[0]

        factbase.remove_fact(target_rr.name, target_rr.is_negated, *target_rr.args)
        factbase.add_fact("router", False, rr_node)

        fault_info["details"].append({
            "type": "wrong_route_reflector",
            "node": str(rr_node),
            "change": "route_reflector -> router"
        })
        return fault_info


class ConfiguredBgpSemantics:
    """配置BGP语义学，包含故障注入能力"""
    def __init__(self, enable_fault_injection=False, fault_probability=1):
        self.s = BgpSemantics(labeled_networks=False)
        self.enable_fault_injection = enable_fault_injection
        self.fault_probability = fault_probability

        if self.enable_fault_injection:
            self.fault_injector = FaultInjector(self.s)

    def sample(self, seed):
        s = np.random.RandomState(seed=seed)

        # 网络参数配置
        real_world_topology = False
        num_networks = choose_random(list(range(4,8)), s)
        num_gateway_nodes = 3
        num_nodes = choose_random(range(16,24), s)

        sample_config_overrides = {
            "fwd":              {"n": choose_random([8, 10, 12], s)},
            "reachable":        {"n": choose_random([4, 5, 6, 7], s)},
            "trafficIsolation": {"n": choose_random(range(10, 30), s)},
        }

        seed = s.randint(0, 1024*1024*1024)

        # the Neurips paper, returned this "original_factbase" in their "ConfiguredBgpSemantics";
        original_factbase = self.s.sample(
            num_nodes=num_nodes,
            real_world_topology=real_world_topology,
            num_networks=num_networks,
            predicate_semantics_sample_config_overrides=sample_config_overrides,
            seed=seed,
            NUM_GATEWAY_NODES=num_gateway_nodes
        )

        # ---- above code is the same as the Neurips paper's ----
        if self.enable_fault_injection and s.random() < self.fault_probability:
            # 注入故障
            fault_seed = s.randint(0, 1024*1024*1024)
            faulty_factbase, fault_info = self.fault_injector.inject_fault_and_resimulate(
                original_factbase,
                fault_type=None,  # 随机选择故障类型
                seed=fault_seed
            )

            if faulty_factbase is not None:
                # 将故障信息附加到FactBase
                faulty_factbase.fault_info = fault_info
                faulty_factbase.is_faulty = True
                return faulty_factbase
            else:
                # 故障注入失败，返回原始样本
                original_factbase.is_faulty = False
                return original_factbase
        else:
            # 返回原始样本（无故障）
            original_factbase.is_faulty = False
            return original_factbase


class FaultAwareFactBaseSemanticsDataset(FactBaseSemanticsDataset):
    """
    扩展的数据集类，支持故障注入
    """
    def __init__(self, semantics, name, num_samples, tmp_directory, fault_enabled=True, fault_probability=1):
        # 配置故障注入
        if hasattr(semantics, 'enable_fault_injection'):
            semantics.enable_fault_injection = fault_enabled
            semantics.fault_probability = fault_probability
            if fault_enabled and not hasattr(semantics, 'fault_injector'):
                semantics.fault_injector = FaultInjector(semantics.s)

        super().__init__(semantics, name, num_samples, tmp_directory)

        self.seeds = np.random.randint(0, 10**9, size=num_samples).tolist()
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        获取数据项，包含故障信息
        """
        data = super().__getitem__(idx)

        # 获取原始的FactBase来检查故障信息
        sample = self.semantics.sample(seed=self.seeds[idx])

        # 添加故障信息到数据对象
        if hasattr(sample, 'is_faulty'):
            data.is_faulty = sample.is_faulty
            if sample.is_faulty and hasattr(sample, 'fault_info'):
                data.fault_info = sample.fault_info
        else:
            data.is_faulty = False

        return data


if __name__ == "__main__":
    print("生成包含故障注入的BGP-OSPF数据集")
    print("============================")

    # 创建支持故障注入的语义学实例
    semantics = ConfiguredBgpSemantics(
        enable_fault_injection=True,    # 启用故障注入
        fault_probability=1             # if this is 0.4, it means that 40% 的样本包含故障, here we consider 100%
    )

    # 创建数据集
    dataset = FaultAwareFactBaseSemanticsDataset( # this is built on "FactBaseSemanticsDataset()""
        semantics,
        "bgp-ospf-fault-dataset",
        num_samples=10,  # 为了测试，使用较小的样本数
        tmp_directory="tmp-bgp-fault-dataset",
        fault_enabled=True,
        fault_probability=1
    )

    print(f"数据集大小: {len(dataset)} \n")

    # 统计故障样本
    fault_count = 0
    normal_count = 0
    fault_types = {}

    for i in range(min(50, len(dataset))):  # 检查前50个样本, or 全部
        data = dataset[i]
        if hasattr(data, 'is_faulty') and data.is_faulty:
            fault_count += 1
            if hasattr(data, 'fault_info'):
                fault_type = data.fault_info['fault_type']
                fault_types[fault_type] = fault_types.get(fault_type, 0) + 1
                print(f"样本 {i}: 故障类型 = {fault_type}")
        else:
            normal_count += 1

    print(f"\n统计结果 (前50个样本):")
    print(f"正常样本: {normal_count}")
    print(f"故障样本: {fault_count}")
    print(f"故障类型分布: {fault_types}")

    # 展示一个样本的详细信息
    if len(dataset) > 0:
        print(f"\n第一个样本详细信息:")
        data = dataset[0]
        print(f"数据类型: {type(data)}")
        print(f"数据键: {data.keys if hasattr(data, 'keys') else 'N/A'}")
        print(f"节点特征矩阵形状: {data.x.shape if hasattr(data, 'x') else 'N/A'}")
        print(f"边索引形状: {data.edge_index.shape if hasattr(data, 'edge_index') else 'N/A'}")
        print(f"是否包含故障: {data.is_faulty if hasattr(data, 'is_faulty') else 'N/A'}")

        if hasattr(data, 'fault_info') and data.is_faulty:
            print(f"故障信息: {data.fault_info}")

    print("\n✅ 故障注入数据集生成完成！")











