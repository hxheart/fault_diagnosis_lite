import sys
import os

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
NUM_TRAINING_EPOCH = 400
NUM_LEN_DATASET = 1024
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
    def __init__(self, hidden_dim, d_inner, num_edge_types, n_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = d_inner

        # 使用原始的GATConv
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

        x2 = torch.stack(
            [layer(l, x.view(-1, self.hidden_dim), edge_index_for_type(t)) for t, l in enumerate(self.layers)],
            axis=0).sum(axis=0)
        x = x + self.drop1(x2)
        x = self.norm1.forward(x.view(-1, self.hidden_dim)).view(x.shape)
        x = self.linear2(self.drop(torch.relu(self.linear1(x))))
        x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)

        return x


class SimpleGATv2Layer(torch.nn.Module):
    """
    使用 GATv2 的层，不区分边类型
    GATv2 的关键改进：先concat再做非线性变换，实现真正的dynamic attention
    """

    def __init__(self, hidden_dim, d_inner, n_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = d_inner

        # 🔥 关键改动：使用 GATv2Conv 替换 GATConv
        self.gat_layer = GATv2Conv(
            self.hidden_dim,
            self.hidden_dim,
            heads=n_heads,
            concat=False,
            dropout=dropout
        )

        # Normalization 和 FFN（保持和原来一样）
        self.drop = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.drop1 = torch.nn.Dropout(dropout)
        self.norm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.drop2 = torch.nn.Dropout(dropout)

        self.linear1 = torch.nn.Linear(self.hidden_dim, self.d_inner)
        self.linear2 = torch.nn.Linear(self.d_inner, self.hidden_dim)

    def forward(self, x, edge_index):
        """
        使用 GATv2 进行注意力传播
        """
        # GATv2 注意力传播（所有边一起处理，不区分类型）
        x2 = self.gat_layer(x.view(-1, self.hidden_dim), edge_index)

        # 残差连接 + LayerNorm
        x = x + self.drop1(x2)
        x = self.norm1.forward(x.view(-1, self.hidden_dim)).view(x.shape)

        # FFN
        x = x + self.drop2(self.linear2(self.drop(torch.relu(self.linear1(x)))))
        x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)

        return x


class EdgeTypeGraphTransformer(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, num_edge_types):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.propagation_layers = [EdgeTypeTransformerLayer(hidden_dim, 4 * hidden_dim, num_edge_types, 8) for i in
                                   range(num_layers)]
        for i, l in enumerate(self.propagation_layers):
            self.add_module("prop_layer_" + str(i), l)

    def forward(self, x, edge_index, edge_type):
        for l in self.propagation_layers:
            x = l.forward(x, edge_index, edge_type)
            x = torch.relu(x)
        return x


class SimpleGATv2Encoder(torch.nn.Module):
    """
    🔥 使用 GATv2 的编码器，堆叠多层 SimpleGATv2Layer
    不使用边类型信息
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.propagation_layers = [
            SimpleGATv2Layer(hidden_dim, 4 * hidden_dim, n_heads=8)
            for i in range(num_layers)
        ]

        for i, l in enumerate(self.propagation_layers):
            self.add_module("prop_layer_" + str(i), l)

    def forward(self, x, edge_index):
        """
        使用 GATv2 进行多层传播
        """
        for l in self.propagation_layers:
            x = l.forward(x, edge_index)
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


class Model_fd(torch.nn.Module):
    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.embedding = PredicateGraphEmbedding(features, hidden_dim, self.num_edge_types, excluded_feature_indices)
        self.transformer_encoder = EdgeTypeGraphTransformer(self.hidden_dim, 2, num_edge_types)

        self.fault_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, 7)
        )

    def forward(self, x, edge_index, edge_type, batch_idx):
        mask = torch.zeros_like(x, dtype=torch.bool)
        x = self.embedding.forward(x, mask, edge_index, edge_type, True)[:, 0]
        x = self.transformer_encoder.forward(x, edge_index, edge_type)
        graph_representation = global_mean_pool(x, batch_idx)
        fault_logits = self.fault_classifier(graph_representation)
        return fault_logits


class Model_fd_GATv2(torch.nn.Module):
    """
    🔥 使用 GATv2 的故障诊断模型
    不使用边类型信息
    """

    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        # Embedding 层（保持不变）
        self.embedding = PredicateGraphEmbedding(
            features, hidden_dim, self.num_edge_types, excluded_feature_indices
        )

        # 🔥 使用 GATv2 编码器
        self.transformer_encoder = SimpleGATv2Encoder(self.hidden_dim, num_layers=2)

        # 故障分类器（保持不变）
        self.fault_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, 7)
        )

    def forward(self, x, edge_index, edge_type, batch_idx):
        """
        注意：保留 edge_type 参数（为了兼容训练循环），但在内部忽略它
        """
        # 节点特征编码
        mask = torch.zeros_like(x, dtype=torch.bool)
        x = self.embedding.forward(x, mask, edge_index, edge_type, True)[:, 0]

        # 🔥 图推理（使用 GATv2，不使用 edge_type）
        x = self.transformer_encoder.forward(x, edge_index)

        # 图级别池化
        graph_representation = global_mean_pool(x, batch_idx)

        # 故障分类
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

            batch.x, fault_labels = inject_single_graph_fault_and_label(batch, features, num_parameters=num_parameters,
                                                                        device=device)
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


if __name__ == '__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=128)
        parser.add_argument("--epochs", dest="epochs", type=int, default=NUM_TRAINING_EPOCH)
        parser.add_argument("--checkpoint", dest="checkpoint", type=str, default=None)
        return parser.parse_args()


    args = get_args()
    HIDDEN_DIM = args.hidden_dim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("====> Running on", device)
    snapshot = ModelSnapshot(__file__)

    # Data loading
    sem = BgpSemantics()
    dataset = FactBaseSemanticsDataset(sem, "bgp-ospf-dataset-sub", num_samples=64)
    print("====> Dataset Size", len(dataset))

    num_validation_samples = 16
    training_dataset, validation_dataset = dataset[num_validation_samples:], dataset[:num_validation_samples]

    print("====> Validation Dataset Size", len(validation_dataset))

    # Feature Settings
    predicate_declarations = sem.decls()
    prog = FactBase(predicate_declarations)
    feature = prog.feature_registry.feature

    excluded_feature_indices = set([1])
    features = prog.feature_registry.get_all_features()
    print(prog.predicate_declarations)
    print(features)

    NUM_FAULT_CLASSES = 7

    # 🔥 创建 GATv2 故障诊断模型
    model = Model_fd_GATv2(features, HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices).to(device)
    model.feature = feature

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    writer = snapshot.writer()
    print("Model parameters:", len(list(model.parameters())))

    batch_size = 4
    num_samples_per_epoch = 32
    num_batches_per_epoch = int(num_samples_per_epoch / batch_size)

    visual_loss = []
    visual_accuracy = []

    for epoch in tqdm(range(args.epochs), leave=False):
        training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        step_writer = StepWriter(writer, epoch)

        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for i, batch in tqdm(enumerate(training_loader), leave=False, total=num_batches_per_epoch,
                             desc=f"Epoch {epoch}"):
            if i > num_batches_per_epoch: break

            optimizer.zero_grad()

            batch = batch.to(device)
            batch.x = batch.x.unsqueeze(1)
            batch.edge_index = reflexive(bidirectional(batch.edge_index), num_nodes=batch.x.size(0))
            batch.edge_type = reflexive_bidirectional_edge_type(batch.edge_type, batch.x.size(0))

            batch.x, fault_labels = inject_single_graph_fault_and_label(batch, feature,
                                                                        num_parameters=NUM_FAULT_CLASSES, device=device)

            fault_predictions = model.forward(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            loss = criterion(fault_predictions, fault_labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(fault_predictions.data, 1)
            total += fault_labels.size(0)
            correct += (predicted == fault_labels).sum().item()

        avg_loss = epoch_loss / num_batches_per_epoch
        accuracy = 100 * correct / total if total > 0 else 0

        step_writer.add_scalar("Loss/Train", avg_loss)
        step_writer.add_scalar("Accuracy/Train", accuracy)

        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

        visual_loss.append(avg_loss)
        visual_accuracy.append(accuracy)

        if epoch % 20 == 0 or epoch < 20 or epoch == NUM_TRAINING_EPOCH-1:
            eval_fd(model, validation_dataset, feature, NUM_FAULT_CLASSES, step_writer, prefix="Val", device=device)
            results = eval_fd_detailed(model, validation_dataset, feature, NUM_FAULT_CLASSES, step_writer, prefix="Val",
                                       device=device)

            np.savez(f"./results/val_GATv2_epoch{epoch}.npz",
                     losses=results['losses'],
                     predictions=results['predictions'],
                     labels=results['labels'],
                     correct=results['correct'])

            uid = os.environ.get("EXP_ID", "FD")
            torch.save([model.state_dict(), HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices], f"models/{uid}-GATv2-epoch{epoch}.pt")

        writer.flush()

    # uid = os.environ.get("EXP_ID", "fault_localization")
    # os.makedirs("models", exist_ok=True)
    # torch.save([model.state_dict(), HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices], f"models/{uid}-model-GATv2-final.pt")

    print("Training completed!")