import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))

import numpy as np
import copy
import pickle
import glob
from collections import defaultdict

from semantics import FactBaseSemanticsDataset
from bgp_semantics import BgpSemantics
from nutils import choose_random
from factbase import FactBase, Constant


class FaultAwareFactBaseSemanticsDataset(FactBaseSemanticsDataset):
    """
    故障感知的FactBase数据集，支持同时生成正常和故障样本
    """

    def __init__(self, semantics, dataset_name, num_samples, tmp_directory="tmp",
                 fault_enabled=True, fault_probability=0.5, force_regenerate=False):
        self.fault_enabled = fault_enabled
        self.fault_probability = fault_probability
        self.dataset_name = dataset_name
        self.cache_dir = f"{dataset_name}_cache"
        self.force_regenerate = force_regenerate

        # 创建缓存目录
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # 检查是否已经存在完整的数据集
        if not force_regenerate and self._check_existing_dataset(num_samples):
            print(f"✅ 发现已存在的数据集 '{dataset_name}'，直接加载...")
            self._load_existing_dataset()
        else:
            print(f"🔄 生成新的数据集 '{dataset_name}'...")
            super().__init__(semantics, dataset_name, num_samples, tmp_directory)
            self._save_dataset_metadata(num_samples)

    def _check_existing_dataset(self, num_samples):
        """检查是否存在完整的数据集"""
        metadata_file = os.path.join(self.cache_dir, "dataset_metadata.pkl")
        if not os.path.exists(metadata_file):
            return False

        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            # 检查样本数量和配置是否匹配
            return (metadata.get('num_samples') == num_samples and
                    metadata.get('fault_enabled') == self.fault_enabled and
                    metadata.get('fault_probability') == self.fault_probability)
        except:
            return False

    def _load_existing_dataset(self):
        """加载已存在的数据集"""
        metadata_file = os.path.join(self.cache_dir, "dataset_metadata.pkl")
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        self.data = []
        self.normal_samples = []
        self.fault_samples = []

        # 加载所有样本
        for i in range(metadata['num_samples']):
            sample_file = os.path.join(self.cache_dir, f"sample_{i}.pkl")
            if os.path.exists(sample_file):
                with open(sample_file, 'rb') as f:
                    sample = pickle.load(f)
                self.data.append(sample)

                # 分类样本
                if hasattr(sample, 'is_faulty') and sample.is_faulty:
                    self.fault_samples.append(sample)
                else:
                    self.normal_samples.append(sample)

        print(f"  📊 加载完成: {len(self.data)} 个样本")
        print(f"    - 正常样本: {len(self.normal_samples)}")
        print(f"    - 故障样本: {len(self.fault_samples)}")

    def _save_dataset_metadata(self, num_samples):
        """保存数据集元数据和样本"""
        metadata = {
            'num_samples': num_samples,
            'fault_enabled': self.fault_enabled,
            'fault_probability': self.fault_probability,
            'dataset_name': self.dataset_name
        }

        # 保存元数据
        metadata_file = os.path.join(self.cache_dir, "dataset_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        # 保存每个样本
        self.normal_samples = []
        self.fault_samples = []

        for i, sample in enumerate(self.data):
            sample_file = os.path.join(self.cache_dir, f"sample_{i}.pkl")
            with open(sample_file, 'wb') as f:
                pickle.dump(sample, f)

            # 分类样本
            if hasattr(sample, 'is_faulty') and sample.is_faulty:
                self.fault_samples.append(sample)
            else:
                self.normal_samples.append(sample)

        print(f"  💾 数据集已保存到缓存: {self.cache_dir}")
        print(f"    - 正常样本: {len(self.normal_samples)}")
        print(f"    - 故障样本: {len(self.fault_samples)}")

    def get_normal_samples(self):
        """获取所有正常样本"""
        if not hasattr(self, 'normal_samples'):
            self._classify_samples()
        return self.normal_samples

    def get_fault_samples(self):
        """获取所有故障样本"""
        if not hasattr(self, 'fault_samples'):
            self._classify_samples()
        return self.fault_samples

    def _classify_samples(self):
        """分类样本"""
        self.normal_samples = []
        self.fault_samples = []

        for sample in self.data:
            if hasattr(sample, 'is_faulty') and sample.is_faulty:
                self.fault_samples.append(sample)
            else:
                self.normal_samples.append(sample)

    def get_fault_statistics(self):
        """获取故障统计信息"""
        fault_types = defaultdict(int)
        fault_impacts = defaultdict(list)

        for sample in self.get_fault_samples():
            if hasattr(sample, 'fault_info'):
                fault_type = sample.fault_info['fault_type']
                fault_types[fault_type] += 1

                # 如果有分析结果，记录影响
                if hasattr(sample, 'analysis_result'):
                    impact = sample.analysis_result.get('consistency_degradation', 0)
                    fault_impacts[fault_type].append(impact)

        return {
            'fault_type_counts': dict(fault_types),
            'fault_impacts': dict(fault_impacts),
            'total_normal': len(self.get_normal_samples()),
            'total_fault': len(self.get_fault_samples())
        }

    def clear_cache(self):
        """清理数据集缓存"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"🗑️  已清理数据集缓存: {self.cache_dir}")


class FaultDetectionAnalyzer:
    """
    故障检测分析器：比较原始和故障样本的规范违反情况
    """

    def __init__(self, bgp_semantics_instance):
        """
        Args:
            bgp_semantics_instance: BgpSemantics实例，用于检查规范一致性
        """
        self.bgp_semantics = bgp_semantics_instance

    def analyze_dataset(self, dataset, max_samples=None):
        """
        分析整个数据集中的故障影响

        Args:
            dataset: FaultAwareFactBaseSemanticsDataset实例
            max_samples: 最大分析样本数，None表示分析所有故障样本

        Returns:
            dict: 分析结果
        """
        print("🔍 开始分析故障对规范的影响...")

        fault_samples = dataset.get_fault_samples()
        normal_samples = dataset.get_normal_samples()

        if max_samples:
            fault_samples = fault_samples[:max_samples]

        print(f"  - 待分析故障样本: {len(fault_samples)}")
        print(f"  - 参考正常样本: {len(normal_samples)}")

        analysis_results = []
        fault_type_stats = defaultdict(list)

        for i, fault_sample in enumerate(fault_samples):
            if i % 10 == 0:
                print(f"  处理进度: {i + 1}/{len(fault_samples)}")

            # 找到对应的正常样本（基于相同的基础拓扑）
            original_sample = self._find_corresponding_normal_sample(
                fault_sample, normal_samples
            )

            if original_sample is None:
                # 如果找不到对应的正常样本，为故障样本生成一个
                original_sample = self._generate_original_for_fault(fault_sample)

            if original_sample is not None:
                # 分析故障影响
                analysis = self.analyze_fault_impact(
                    original_sample, fault_sample, fault_sample.fault_info
                )

                analysis_results.append(analysis)
                fault_type = fault_sample.fault_info["fault_type"]
                fault_type_stats[fault_type].append(analysis["consistency_degradation"])

                # 将分析结果保存到样本中
                fault_sample.analysis_result = analysis

        # 汇总统计
        summary = self._generate_analysis_summary(analysis_results, fault_type_stats)

        print("✅ 故障影响分析完成！")
        return {
            'detailed_results': analysis_results,
            'summary': summary,
            'fault_type_stats': dict(fault_type_stats)
        }

    def _find_corresponding_normal_sample(self, fault_sample, normal_samples):
        """寻找对应的正常样本"""
        # 简化版本：返回第一个正常样本
        # 在实际应用中，可以根据网络拓扑特征匹配更相似的样本
        return normal_samples[0] if normal_samples else None

    def _generate_original_for_fault(self, fault_sample):
        """为故障样本生成对应的原始样本"""
        # 这里可以实现根据故障样本反推原始样本的逻辑
        # 暂时返回None，表示无法生成
        return None

    def _generate_analysis_summary(self, analysis_results, fault_type_stats):
        """生成分析摘要"""
        if not analysis_results:
            return {}

        avg_degradation = np.mean([r["consistency_degradation"] for r in analysis_results])
        samples_with_violations = sum(1 for r in analysis_results if r["has_new_violations"])

        summary = {
            'total_analyzed_samples': len(analysis_results),
            'average_consistency_degradation': avg_degradation,
            'samples_with_new_violations': samples_with_violations,
            'violation_rate': samples_with_violations / len(analysis_results),
            'fault_type_analysis': {}
        }

        for fault_type, degradations in fault_type_stats.items():
            if degradations:
                violation_count = sum(1 for r in analysis_results
                                      if r["fault_info"]["fault_type"] == fault_type
                                      and r["has_new_violations"])

                summary['fault_type_analysis'][fault_type] = {
                    'sample_count': len(degradations),
                    'avg_degradation': np.mean(degradations),
                    'violation_count': violation_count,
                    'violation_rate': violation_count / len(degradations)
                }

        return summary

    def analyze_fault_impact(self, original_factbase, faulty_factbase, fault_info):
        """
        分析故障对规范的影响

        Args:
            original_factbase: 原始FactBase
            faulty_factbase: 故障FactBase
            fault_info: 故障信息

        Returns:
            dict: 包含详细分析结果的字典
        """
        # 检查原始样本的规范一致性
        original_consistency = self.bgp_semantics.check(
            original_factbase,
            return_summary=True,
            return_corrected_program=True
        )
        original_score, original_corrected, original_summary = original_consistency

        # 检查故障样本的规范一致性
        faulty_consistency = self.bgp_semantics.check(
            faulty_factbase,
            return_summary=True,
            return_corrected_program=True
        )
        faulty_score, faulty_corrected, faulty_summary = faulty_consistency

        # 计算规范违反的差异
        violation_analysis = self._analyze_violations(
            original_summary, faulty_summary,
            original_corrected, faulty_corrected
        )

        analysis_result = {
            "fault_info": fault_info,
            "original_consistency_score": original_score,
            "faulty_consistency_score": faulty_score,
            "consistency_degradation": original_score - faulty_score,
            "original_summary": original_summary,
            "faulty_summary": faulty_summary,
            "violation_analysis": violation_analysis,
            "has_new_violations": violation_analysis["newly_violated_predicates"] > 0
        }

        return analysis_result

    def _analyze_violations(self, original_summary, faulty_summary,
                            original_corrected, faulty_corrected):
        """
        分析规范违反的详细情况
        """
        # 找出新违反的谓词类型
        newly_violated = set()
        fixed_violations = set()

        for pred_name in set(list(original_summary.keys()) + list(faulty_summary.keys())):
            if pred_name == "overall":
                continue

            original_score = original_summary.get(pred_name, 1.0)
            faulty_score = faulty_summary.get(pred_name, 1.0)

            # 如果原来是对的，现在是错的
            if original_score == 1.0 and faulty_score < 1.0:
                newly_violated.add(pred_name)
            # 如果原来是错的，现在是对的
            elif original_score < 1.0 and faulty_score == 1.0:
                fixed_violations.add(pred_name)

        # 统计具体违反的事实数量
        violation_details = self._count_fact_violations(
            original_corrected, faulty_corrected
        )

        return {
            "newly_violated_predicates": len(newly_violated),
            "fixed_violations": len(fixed_violations),
            "newly_violated_predicate_types": list(newly_violated),
            "fixed_violation_types": list(fixed_violations),
            "violation_details": violation_details
        }

    def _count_fact_violations(self, original_corrected, faulty_corrected):
        """
        统计具体违反的事实数量
        """
        # 获取所有规范事实（非配置事实）
        spec_predicates = {"fwd", "reachable", "trafficIsolation"}

        violation_counts = defaultdict(int)

        original_spec_facts = set()
        faulty_spec_facts = set()

        # 收集原始样本的规范事实
        for fact in original_corrected.get_all_facts():
            if fact.name in spec_predicates:
                fact_str = f"{fact.name}({','.join(str(arg) for arg in fact.args)})"
                if not fact.is_negated:
                    original_spec_facts.add(fact_str)

        # 收集故障样本的规范事实
        for fact in faulty_corrected.get_all_facts():
            if fact.name in spec_predicates:
                fact_str = f"{fact.name}({','.join(str(arg) for arg in fact.args)})"
                if not fact.is_negated:
                    faulty_spec_facts.add(fact_str)

        # 计算差异
        lost_facts = original_spec_facts - faulty_spec_facts
        new_facts = faulty_spec_facts - original_spec_facts

        for fact_str in lost_facts:
            pred_name = fact_str.split('(')[0]
            violation_counts[f"lost_{pred_name}_facts"] += 1

        for fact_str in new_facts:
            pred_name = fact_str.split('(')[0]
            violation_counts[f"new_{pred_name}_facts"] += 1

        return dict(violation_counts)


# 继承之前的故障注入相关类
class FaultInjector:
    """故障注入器：注入故障后重新运行协议仿真"""

    def __init__(self, bgp_semantics_instance):
        self.bgp_semantics = bgp_semantics_instance
        self.fault_types = [
            'ospf_weight_corruption',
            'bgp_preference_corruption',
            'missing_ibgp_session',
            'wrong_route_reflector',
        ]

    def inject_fault_and_resimulate(self, original_factbase, fault_type=None, seed=42):
        """注入故障并重新运行协议仿真"""
        import random
        random.seed(seed)
        np.random.seed(seed)

        faulty_factbase = copy.deepcopy(original_factbase)

        if fault_type is None:
            fault_type = random.choice(self.fault_types)

        fault_info = {"fault_type": fault_type, "details": []}

        if fault_type == 'ospf_weight_corruption':
            fault_info = self._inject_ospf_weight_fault(faulty_factbase, fault_info)
        elif fault_type == 'bgp_preference_corruption':
            fault_info = self._inject_bgp_preference_fault(faulty_factbase, fault_info)
        elif fault_type == 'missing_ibgp_session':
            fault_info = self._inject_missing_ibgp_fault(faulty_factbase, fault_info)
        elif fault_type == 'wrong_route_reflector':
            fault_info = self._inject_wrong_rr_fault(faulty_factbase, fault_info)

        try:
            faulty_factbase = self._resimulate_from_factbase(faulty_factbase)
        except Exception as e:
            print(f"重新仿真失败: {e}")
            return None, fault_info

        return faulty_factbase, fault_info

    def _resimulate_from_factbase(self, modified_factbase):
        """从修改后的FactBase重新运行协议仿真"""
        from bgp_semantics import compute_forwarding_state

        network, network_nodes = self.bgp_semantics.network(modified_factbase)
        compute_forwarding_state(network)

        updated_factbase = self._rebuild_factbase_with_updated_specs(
            modified_factbase, network, network_nodes
        )
        return updated_factbase

    def _rebuild_factbase_with_updated_specs(self, original_factbase, updated_network, network_nodes):
        """基于重新仿真后的网络，重建包含更新规范的FactBase"""
        new_factbase = FactBase(original_factbase.predicate_declarations)

        config_predicates = {"router", "network", "external", "route_reflector",
                             "ibgp", "ebgp", "bgp_route", "connected"}

        for fact in original_factbase.get_all_facts():
            if fact.name in config_predicates:
                new_factbase.add_fact(fact.name, fact.is_negated, *fact.args)

        s = np.random.RandomState(seed=42)

        network_mapping = {}
        if hasattr(self.bgp_semantics, 'labeled_networks') and self.bgp_semantics.labeled_networks:
            for n in network_nodes:
                if 'net_label' in updated_network.nodes[n]:
                    network_mapping[updated_network.nodes[n]['net_label']] = n

        for pred_s in self.bgp_semantics.predicate_semantics:
            config = getattr(self.bgp_semantics, 'predicate_semantics_sample_config', {}).get(
                pred_s.predicate_name, {}
            )
            derived = pred_s.sample(updated_network, random=s, **config)

            if hasattr(self.bgp_semantics, 'labeled_networks') and self.bgp_semantics.labeled_networks:
                for f in derived:
                    def network_constants_to_label(a):
                        if type(a) is Constant and a.name in network_mapping.keys():
                            return network_mapping[a.name]
                        return a

                    f.args = [network_constants_to_label(a) for a in f.args]

            for f in derived:
                new_factbase.add_fact(f.name, f.is_negated, *f.args)

        return new_factbase

    def _inject_ospf_weight_fault(self, factbase, fault_info):
        """注入OSPF权重故障"""
        import random
        all_facts = factbase.get_all_facts()
        connected_facts = [fact for fact in all_facts if fact.name == "connected"]

        if not connected_facts:
            fault_info["details"].append("No connected facts found")
            return fault_info

        target_fact = random.choice(connected_facts)
        src, dst, original_weight = target_fact.args
        original_weight = int(original_weight)
        wrong_weight = min(original_weight * random.randint(5, 20), 999)

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
        import random
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
        import random
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
        import random
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

    def __init__(self, enable_fault_injection=False, fault_probability=0.5):
        self.s = BgpSemantics(labeled_networks=False)
        self.enable_fault_injection = enable_fault_injection
        self.fault_probability = fault_probability

        if self.enable_fault_injection:
            self.fault_injector = FaultInjector(self.s)

    def sample(self, seed):
        s = np.random.RandomState(seed=seed)

        real_world_topology = False
        num_networks = choose_random(list(range(4, 8)), s)
        num_gateway_nodes = 3
        num_nodes = choose_random(range(16, 24), s)

        sample_config_overrides = {
            "fwd": {"n": choose_random([8, 10, 12], s)},
            "reachable": {"n": choose_random([4, 5, 6, 7], s)},
            "trafficIsolation": {"n": choose_random(range(10, 30), s)},
        }

        seed = s.randint(0, 1024 * 1024 * 1024)

        original_factbase = self.s.sample(
            num_nodes=num_nodes,
            real_world_topology=real_world_topology,
            num_networks=num_networks,
            predicate_semantics_sample_config_overrides=sample_config_overrides,
            seed=seed,
            NUM_GATEWAY_NODES=num_gateway_nodes
        )

        if self.enable_fault_injection and s.random() < self.fault_probability:
            fault_seed = s.randint(0, 1024 * 1024 * 1024)
            faulty_factbase, fault_info = self.fault_injector.inject_fault_and_resimulate(
                original_factbase,
                fault_type=None,
                seed=fault_seed
            )

            if faulty_factbase is not None:
                faulty_factbase.fault_info = fault_info
                faulty_factbase.is_faulty = True
                return faulty_factbase
            else:
                original_factbase.is_faulty = False
                return original_factbase
        else:
            original_factbase.is_faulty = False
            return original_factbase


def print_dataset_summary(dataset):
    """打印数据集摘要信息"""
    print(f"\n📊 数据集 '{dataset.dataset_name}' 摘要:")
    print(f"  总样本数: {len(dataset)}")

    stats = dataset.get_fault_statistics()
    print(f"  正常样本: {stats['total_normal']}")
    print(f"  故障样本: {stats['total_fault']}")

    if stats['fault_type_counts']:
        print(f"  故障类型分布:")
        for fault_type, count in stats['fault_type_counts'].items():
            print(f"    {fault_type}: {count}")


if __name__ == "__main__":
    print("🚀 智能故障检测系统 - 带缓存优化")
    print("=======================")

    # 步骤1: 生成或加载故障数据集
    print("\n步骤1: 生成/加载故障数据集")
    print("-----------------------")

    # 创建支持故障注入的语义学实例
    semantics = ConfiguredBgpSemantics(
        enable_fault_injection=True,  # 启用故障注入
        fault_probability=1.0  # 100%的样本包含故障
    )

    # 创建故障数据集 (如果已存在会自动加载)
    fault_dataset = FaultAwareFactBaseSemanticsDataset(
        semantics,
        "bgp-ospf-fault-dataset",
        num_samples=1,  # 测试用小样本
        tmp_directory="tmp-bgp-fault-dataset",
        fault_enabled=True,
        fault_probability=1.0,
        force_regenerate=False  # 设为True可强制重新生成
    )

    print_dataset_summary(fault_dataset)

    # 步骤2: