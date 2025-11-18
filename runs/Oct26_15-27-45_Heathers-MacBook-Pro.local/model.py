import sys
import os

import glob
import re
import matplotlib.pyplot as plt

from torch.nn.modules.transformer import TransformerDecoderLayer
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.nn import global_mean_pool

sys.path.append(os.path.join(os.path.dirname(__file__), "dataset"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

import numpy as np
from multiprocessing import Pool
from functools import reduce
from tqdm import tqdm

from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GatedGraphConv  # 添加 GATv2Conv
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from feature import *
from coders import *
from sampling import sample_random_order

from factbase import FactBase, Constant
from nutils import *
from semantics import FactBaseSemanticsDataset

from snapshot import ModelSnapshot
from bgp_semantics import BgpSemantics

import argparse

NUM_EDGE_TYPES = 4
NUM_TRAINING_EPOCH = 20
NUM_LEN_DATASET = 128
NUM_LEN_DATASET_EVAL = max(100, int(NUM_LEN_DATASET * 0.25))


# determines factbase consistency
def check_factbase(semantics, data):
    program = FactBase.from_data(data)
    _, summary = semantics.check(program, ignore_missing_fwd_facts=True, return_summary=True)
    return summary


def combine_dict(dicts):
    flatten = lambda seq: reduce(lambda a, b: a.union(b), seq, set())
    keys = flatten([set(d.keys()) for d in dicts])
    values_for_key = lambda k: [d[k] for d in dicts if k in d.keys()]
    return dict([(k, torch.tensor(values_for_key(k))) for k in keys])


class AsyncFactBaseChecker:
    def __init__(self, pool):
        self.checking_tasks = []
        self.current_step = -1
        self.on_step_finish = None
        self.pool = pool

    def wait_for_step_results(self, global_step):
        if self.current_step == global_step:
            return
        else:
            if any([not task.ready() for task in self.checking_tasks]):
                print("info: waiting for program checker in evaluation (consider using more processes)")
            self._get_results_blocking()
            self.current_step = global_step
            self.checking_tasks = []

    def _get_results_blocking(self):
        summaries = [task.get() for task in self.checking_tasks]
        results = combine_dict(summaries)
        if self.current_step != -1:
            assert self.on_step_finish is not None, "must set self.on_step_finish callback function"
            self.on_step_finish(self.current_step, results)

    def check_last_step_results(self):
        if any([not task.ready() for task in self.checking_tasks]):
            return
        self._get_results_blocking()

    def queue(self, semantics, data, global_step):
        self.wait_for_step_results(global_step)
        self.checking_tasks.append(self.pool.apply_async(check_factbase, (semantics, data)))


class MaxGraphLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='max')
        self.lin = torch.nn.Linear(hidden_dim, hidden_dim)
        self.message_func = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.lin(x)
        x = torch.relu(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return torch.relu(self.message_func.forward(x_j))


class EdgeTypeTransformerLayer(torch.nn.Module):
    """
    原始的 Edge Type Aware GAT Layer (使用经典 GAT)
    """

    def __init__(self, hidden_dim, d_inner, num_edge_types, n_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = d_inner

        # 为每种边类型创建独立的 GAT 层
        self.layers = []
        for j in range(num_edge_types):
            l = GATConv(self.hidden_dim, self.hidden_dim, n_heads, False)
            self.layers.append(l)
            self.add_module("layer_edge_type_" + str(j), l)

        self.drop = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.drop1 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.drop2 = torch.nn.Dropout(dropout)

        self.linear1 = torch.nn.Linear(self.hidden_dim, self.d_inner)
        self.linear2 = torch.nn.Linear(self.d_inner, self.hidden_dim)

    def forward(self, x, edge_index, edge_type):
        def edge_index_for_type(type):
            indices = (edge_type == type)
            index = edge_index[:, indices]
            return index

        def layer(l, x, ei):
            return l(x, ei)

        # 对每种边类型分别应用注意力机制，然后求和
        x2 = torch.stack([layer(l, x.view(-1, self.hidden_dim), edge_index_for_type(t))
                          for t, l in enumerate(self.layers)], axis=0).sum(axis=0)
        x = x + self.drop1(x2)
        x = self.norm1.forward(x.view(-1, self.hidden_dim)).view(x.shape)
        x = self.linear2(self.drop(torch.relu(self.linear1(x))))
        x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)

        return x


class EdgeTypeTransformerLayerV2(torch.nn.Module):
    """
    🔥 Edge Type Aware GATv2 Layer
    关键改进：将每种边类型的 GATConv 替换为 GATv2Conv
    """

    def __init__(self, hidden_dim, d_inner, num_edge_types, n_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = d_inner

        # 🔥 为每种边类型创建独立的 GATv2 层
        self.layers = []
        for j in range(num_edge_types):
            l = GATv2Conv(
                self.hidden_dim,
                self.hidden_dim,
                heads=n_heads,
                concat=False,
                dropout=dropout
            )
            self.layers.append(l)
            self.add_module("layer_edge_type_" + str(j), l)

        self.drop = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.drop1 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.drop2 = torch.nn.Dropout(dropout)

        self.linear1 = torch.nn.Linear(self.hidden_dim, self.d_inner)
        self.linear2 = torch.nn.Linear(self.d_inner, self.hidden_dim)

    def forward(self, x, edge_index, edge_type):
        def edge_index_for_type(type):
            indices = (edge_type == type)
            index = edge_index[:, indices]
            return index

        def layer(l, x, ei):
            return l(x, ei)

        # 🔥 对每种边类型分别应用 GATv2 注意力机制，然后求和
        x2 = torch.stack([layer(l, x.view(-1, self.hidden_dim), edge_index_for_type(t))
                          for t, l in enumerate(self.layers)], axis=0).sum(axis=0)
        x = x + self.drop1(x2)
        x = self.norm1.forward(x.view(-1, self.hidden_dim)).view(x.shape)
        x = self.linear2(self.drop(torch.relu(self.linear1(x))))
        x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)

        return x


class EdgeTypeGraphTransformer(torch.nn.Module):
    """
    原始的 Edge Type Aware Graph Transformer (使用经典 GAT)
    """

    def __init__(self, hidden_dim, num_layers, num_edge_types):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.propagation_layers = [
            EdgeTypeTransformerLayer(hidden_dim, 4 * hidden_dim, num_edge_types, 8)
            for i in range(num_layers)
        ]
        for i, l in enumerate(self.propagation_layers):
            self.add_module("prop_layer_" + str(i), l)

    def forward(self, x, edge_index, edge_type):
        for l in self.propagation_layers:
            x = l.forward(x, edge_index, edge_type)
            x = torch.relu(x)
        return x


class EdgeTypeGraphTransformerV2(torch.nn.Module):
    """
    🔥 Edge Type Aware Graph Transformer V2 (使用 GATv2)
    """

    def __init__(self, hidden_dim, num_layers, num_edge_types):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.propagation_layers = [
            EdgeTypeTransformerLayerV2(hidden_dim, 4 * hidden_dim, num_edge_types, 8)
            for i in range(num_layers)
        ]
        for i, l in enumerate(self.propagation_layers):
            self.add_module("prop_layer_" + str(i), l)

    def forward(self, x, edge_index, edge_type):
        for l in self.propagation_layers:
            x = l.forward(x, edge_index, edge_type)
            x = torch.relu(x)
        return x


class PredicateGraphEmbedding(torch.nn.Module):
    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices=set(), num_layers=6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.encoder = NodeFeatureEmbedding(hidden_dim, features, excluded_feature_indices=excluded_feature_indices)
        self.decoder = NodeFeatureDecoder(hidden_dim, features, excluded_feature_indices=excluded_feature_indices)

    def forward(self, x, mask, edge_index, edge_type, reliable_masking):
        return self.encoder.forward(x, mask, reliable_masking, positional_encoding=False)


# class Model_fd(torch.nn.Module):
#     """
#     原始的 Edge Type Aware GAT 故障诊断模型
#     """

#     def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices):
#         super().__init__()

#         self.hidden_dim = hidden_dim
#         self.num_edge_types = num_edge_types

#         self.embedding = PredicateGraphEmbedding(features, hidden_dim, self.num_edge_types, excluded_feature_indices)
#         self.transformer_encoder = EdgeTypeGraphTransformer(self.hidden_dim, 2, num_edge_types)

#         self.fault_classifier = torch.nn.Sequential(
#             torch.nn.Linear(hidden_dim, hidden_dim // 2),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.2),
#             torch.nn.Linear(hidden_dim // 2, 7)
#         )

#     def forward(self, x, edge_index, edge_type, batch_idx):
#         mask = torch.zeros_like(x, dtype=torch.bool)
#         x = self.embedding.forward(x, mask, edge_index, edge_type, True)[:, 0]
#         x = self.transformer_encoder.forward(x, edge_index, edge_type)
#         graph_representation = global_mean_pool(x, batch_idx)
#         fault_logits = self.fault_classifier(graph_representation)
#         return fault_logits


class Model_fd_EtaGATv2(torch.nn.Module):
    """
    🔥 Edge Type Aware GATv2 故障诊断模型
    结合了 Edge Type Aware 机制和 GATv2 的动态注意力
    """

    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.embedding = PredicateGraphEmbedding(features, hidden_dim, self.num_edge_types, excluded_feature_indices)

        # 🔥 使用 GATv2 版本的 Edge Type Aware Transformer
        self.transformer_encoder = EdgeTypeGraphTransformerV2(self.hidden_dim, 2, num_edge_types)

        self.fault_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, 7)
        )

    def forward(self, x, edge_index, edge_type, batch_idx):
        mask = torch.zeros_like(x, dtype=torch.bool)
        x = self.embedding.forward(x, mask, edge_index, edge_type, True)[:, 0]

        # 🔥 使用 EtaGATv2 编码器进行图推理
        x = self.transformer_encoder.forward(x, edge_index, edge_type)

        graph_representation = global_mean_pool(x, batch_idx)
        fault_logits = self.fault_classifier(graph_representation)
        return fault_logits


def inject_single_graph_fault_and_label(batch, feature, num_parameters=7, device='cpu'):
    """
    batch: PyG Data batch
    feature: prog.feature_registry.feature
    num_parameters: 总共可出错参数数量
    """
    batch_size = batch.num_graphs
    labels = torch.randint(0, num_parameters, (batch_size,), device=device)

    for i in range(batch_size):
        fault_idx = labels[i].item()
        if fault_idx == 0:
            col_idx = feature("predicate_connected_arg2").idx
        elif fault_idx == 1:
            col_idx = feature("predicate_bgp_route_arg2").idx
        elif fault_idx == 2:
            col_idx = feature("predicate_bgp_route_arg3").idx
        elif fault_idx == 3:
            col_idx = feature("predicate_bgp_route_arg4").idx
        elif fault_idx == 4:
            col_idx = feature("predicate_bgp_route_arg5").idx
        elif fault_idx == 5:
            col_idx = feature("predicate_bgp_route_arg6").idx
        elif fault_idx == 6:
            col_idx = feature("predicate_bgp_route_arg7").idx
        else:
            raise ValueError("Invalid fault index")

        if i == 0:
            start_idx = 0
        else:
            start_idx = torch.sum(batch.batch < i).item()
        end_idx = torch.sum(batch.batch <= i).item()

        x_orig = batch.x[start_idx:end_idx, :, col_idx]
        batch.x[start_idx:end_idx, :, col_idx] = x_orig + torch.randint(1, 5, x_orig.shape, device=device)

    return batch.x, labels


def eval_fd(model, dataset, features, num_parameters, step_writer, prefix, device):
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = batch.x.unsqueeze(1)
            batch.edge_index = reflexive(bidirectional(batch.edge_index), num_nodes=batch.x.size(0))
            batch.edge_type = reflexive_bidirectional_edge_type(batch.edge_type, batch.x.size(0))

            batch.x, fault_labels = inject_single_graph_fault_and_label(batch, features, num_parameters=num_parameters, device=device)
            fault_predictions = model.forward(batch.x, batch.edge_index, batch.edge_type, batch.batch)

            val_loss += criterion(fault_predictions, fault_labels).item()
            _, predicted = torch.max(fault_predictions.data, 1)
            val_total += fault_labels.size(0)
            val_correct += (predicted == fault_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
    step_writer.add_scalar(f"{prefix}/Loss", val_loss / len(loader))
    step_writer.add_scalar(f"{prefix}/Accuracy", val_accuracy)

    print(f"{prefix}: Loss = {val_loss / len(loader):.4f}, Accuracy = {val_accuracy:.2f}%")


def eval_fd_detailed(model, dataset, features, num_parameters, step_writer, prefix, device):
    """
    改进版的评估函数，保存每个样本的详细信息
    """
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    all_losses = []
    all_predictions = []
    all_labels = []
    all_correct = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = batch.x.unsqueeze(1)
            batch.edge_index = reflexive(bidirectional(batch.edge_index), num_nodes=batch.x.size(0))
            batch.edge_type = reflexive_bidirectional_edge_type(batch.edge_type, batch.x.size(0))

            batch.x, fault_labels = inject_single_graph_fault_and_label(
                batch, features, num_parameters=num_parameters, device=device)

            fault_predictions = model.forward(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            losses = criterion(fault_predictions, fault_labels)
            _, predicted = torch.max(fault_predictions.data, 1)
            correct = (predicted == fault_labels)

            all_losses.extend(losses.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(fault_labels.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())

    all_losses = np.array(all_losses)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_correct = np.array(all_correct)

    mean_loss = all_losses.mean()
    std_loss = all_losses.std()
    accuracy = 100 * all_correct.sum() / len(all_correct)

    print(f"\n{'=' * 60}")
    print(f"{prefix} Results:")
    print(f"  Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  Accuracy:  {accuracy:.2f}% ({all_correct.sum()}/{len(all_correct)})")
    print(f"  Min Loss:  {all_losses.min():.4f}")
    print(f"  Max Loss:  {all_losses.max():.4f}")
    print(f"{'=' * 60}\n")

    step_writer.add_scalar(f"{prefix}/Loss_Mean", mean_loss)
    step_writer.add_scalar(f"{prefix}/Loss_Std", std_loss)
    step_writer.add_scalar(f"{prefix}/Accuracy", accuracy)

    results = {
        'losses': all_losses,
        'predictions': all_predictions,
        'labels': all_labels,
        'correct': all_correct,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'accuracy': accuracy
    }

    return results


def batch_evaluate_models():
    """批量评估所有模型检查点"""

    # 1. 查找所有模型文件
    model_pattern = "models/FD-EtaGATv2-epoch*.pt"
    model_files = sorted(glob.glob(model_pattern))

    if not model_files:
        print(f"❌ 未找到模型文件！请检查路径: {model_pattern}")
        return

    print(f"✅ 找到 {len(model_files)} 个模型文件")

    # 2. 提取 epoch 编号
    def extract_epoch(filename):
        match = re.search(r'epoch(\d+)', filename)
        return int(match.group(1)) if match else -1

    model_files = sorted(model_files, key=extract_epoch)

    # 3. 创建结果存储目录
    os.makedirs("./results", exist_ok=True)

    # 4. 存储所有结果
    all_results = {
        'epochs': [],
        'mean_losses': [],
        'std_losses': [],
        'accuracies': []
    }

    # 5. 选择验证集（可以修改）
    validation_datasets = {
        'origin': validation_dataset_origin,
        'larger': validation_dataset_larger,
        'real': validation_dataset_real
    }

    # 选择要评估的数据集
    # dataset_name = 'origin'  # 改成 'origin' 或 'real' 来切换数据集
    # validation_dataset = validation_datasets[dataset_name]
    # print(f"📊 使用数据集: {dataset_name}")

    # 6. 批量评估
    print("\n" + "=" * 70)
    print("开始批量评估 EtaGATv2 模型...")
    print("=" * 70)

    for dataset_name, validation_dataset in validation_datasets.items():
        print(f"\n{'=' * 70}")
        print(f"📊 正在评估数据集: {dataset_name.upper()}")
        print(f"{'=' * 70}")

        # 重置结果存储（每个数据集独立）
        all_results = {
            'epochs': [],
            'mean_losses': [],
            'std_losses': [],
            'accuracies': []
        }


        for model_file in model_files:
            epoch = extract_epoch(model_file)
            print(f"\n>>> 正在评估: {model_file} (Epoch {epoch})")

            try:
                # 加载模型
                checkpoint = torch.load(model_file, map_location=device)
                model.load_state_dict(checkpoint[0])

                # 创建 step_writer
                step_writer = StepWriter(writer, epoch)

                # 评估模型
                results = eval_fd_detailed(
                    model,
                    validation_dataset,
                    feature,
                    NUM_FAULT_CLASSES,
                    step_writer,
                    prefix=f"Val_EtaGATv2_{dataset_name}_Epoch{epoch}",
                    device=device
                )

                # 保存详细结果（关键：文件名格式要匹配）
                if dataset_name == 'orign':
                    filename = f"./results/val_EtaGATv2_epoch{epoch}.npz"
                else:
                    filename = f"./results/val_EtaGATv2_epoch{epoch}_{dataset_name}.npz"

                # 保存详细结果
                np.savez(
                    f"./results/val_EtaGATv2_epoch{epoch}_{dataset_name}.npz",
                    losses=results['losses'],
                    predictions=results['predictions'],
                    labels=results['labels'],
                    correct=results['correct']
                )

                # 收集汇总数据
                all_results['epochs'].append(epoch)
                all_results['mean_losses'].append(results['mean_loss'])
                all_results['std_losses'].append(results['std_loss'])
                all_results['accuracies'].append(results['accuracy'])

                print(f"✅ Epoch {epoch}: Loss={results['mean_loss']:.4f}, Acc={results['accuracy']:.2f}%")

            except Exception as e:
                print(f"❌ 评估 {model_file} 时出错: {e}")
                continue

        # 7. 保存汇总结果
        summary_file = f"./results/summary_EtaGATv2_{dataset_name}.npz"
        np.savez(
            summary_file,
            epochs=np.array(all_results['epochs']),
            mean_losses=np.array(all_results['mean_losses']),
            std_losses=np.array(all_results['std_losses']),
            accuracies=np.array(all_results['accuracies'])
        )
        print(f"\n✅ 汇总结果已保存到: {summary_file}")

    return all_results, dataset_name


# ====================================================================
# 第二部分：可视化函数（新增）
# ====================================================================

def plot_training_curves(all_results, dataset_name):
    """绘制训练曲线"""

    epochs = all_results['epochs']
    mean_losses = all_results['mean_losses']
    std_losses = all_results['std_losses']
    accuracies = all_results['accuracies']

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === 左图：Loss 曲线 ===
    ax1 = axes[0]
    ax1.plot(epochs, mean_losses, 'b-o', linewidth=2, markersize=6, label='Mean Loss')
    ax1.fill_between(
        epochs,
        np.array(mean_losses) - np.array(std_losses),
        np.array(mean_losses) + np.array(std_losses),
        alpha=0.3,
        label='±1 Std Dev'
    )
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'Validation Loss over Training\n(Dataset: {dataset_name})',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # === 右图：Accuracy 曲线 ===
    ax2 = axes[1]
    ax2.plot(epochs, accuracies, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Validation Accuracy over Training\n(Dataset: {dataset_name})',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])  # Accuracy 范围 0-100%

    # 在最高点标注
    max_acc_idx = np.argmax(accuracies)
    max_acc = accuracies[max_acc_idx]
    max_acc_epoch = epochs[max_acc_idx]
    ax2.annotate(
        f'Max: {max_acc:.2f}%\n(Epoch {max_acc_epoch})',
        xy=(max_acc_epoch, max_acc),
        xytext=(10, -20),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )

    plt.tight_layout()

    # 保存图片
    fig_path = f"./results/training_curves_{dataset_name}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {fig_path}")

    plt.show()



# mainloop of classification task
if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=128)
        parser.add_argument("--epochs",     dest="epochs",     type=int, default=NUM_TRAINING_EPOCH)
        parser.add_argument("--checkpoint", dest="checkpoint", type=str, default=None)
        return parser.parse_args()

    args = get_args()
    HIDDEN_DIM = args.hidden_dim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("====> Running on", device)
    snapshot = ModelSnapshot(__file__)

    # Data loading
    sem = BgpSemantics()
    # dataset = FactBaseSemanticsDataset(sem, "bgp-ospf-dataset-sub", num_samples=NUM_LEN_DATASET_EVAL)
    # print("====> Dataset Size", len(dataset))

    num_validation_samples = int(NUM_LEN_DATASET_EVAL)
    # training_dataset, validation_dataset = dataset[num_validation_samples:], dataset[:num_validation_samples]

    validation_dataset_origin = FactBaseSemanticsDataset(sem, "bgp-ospf-dataset-test-origin", num_samples=NUM_LEN_DATASET_EVAL)
    validation_dataset_larger = FactBaseSemanticsDataset(sem, "bgp-ospf-dataset-test-larger", num_samples=NUM_LEN_DATASET_EVAL)
    validation_dataset_real   = FactBaseSemanticsDataset(sem, "bgp-ospf-dataset-test-real",   num_samples=NUM_LEN_DATASET_EVAL)

    # print("====> Validation Dataset Size", len(validation_dataset))

    # Feature Settings
    predicate_declarations = sem.decls()
    prog = FactBase(predicate_declarations)
    feature = prog.feature_registry.feature

    excluded_feature_indices = set([1])
    features = prog.feature_registry.get_all_features()
    print(prog.predicate_declarations)
    print(features)

    # === 这里你要设定 fault classification 的类别数量（比如 7） ===
    NUM_FAULT_CLASSES = 7

    # Creating a fault diagnosis model
    model = Model_fd_EtaGATv2(features, HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices).to(device)
    model.feature = feature

    # Loss function and optimizer of Classification task
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    writer = snapshot.writer()
    print("Model parameters:", len(list(model.parameters())))

    # Training parameters
    batch_size = 4
    num_samples_per_epoch = 32
    num_batches_per_epoch = int(num_samples_per_epoch / batch_size)

    # ########################### HERE, do the evaluations ###########################
    # validation_dataset = validation_dataset_origin
    validation_dataset = validation_dataset_larger
    # validation_dataset = validation_dataset_real

    # 批量评估所有模型
    all_results, dataset_name = batch_evaluate_models()

    # 绘制训练曲线
    if all_results and len(all_results['epochs']) > 0:
        plot_training_curves(all_results, dataset_name)
        print("\n🎉 EtaGATv2 所有评估完成！")
    else:
        print("\n❌ EtaGATv2 没有成功评估任何模型")