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

from torch_geometric.nn import GATConv, GCNConv, GatedGraphConv
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
NUM_LEN_DATASET = 1024
NUM_LEN_DATASET_EVAL = max(100, int(NUM_LEN_DATASET*0.25))

# determines factbase consistency
# (input: Semantic checker and network configuration data)
# (output: Summary of network configuration consistency check results)
def check_factbase(semantics, data):
    program = FactBase.from_data(data) # transfer the data from "PyG" to "FactBase"
    _, summary = semantics.check(program, ignore_missing_fwd_facts=True, return_summary=True) # FactBase → NetworkX → Prot()
    return summary

def combine_dict(dicts):
    flatten = lambda seq: reduce(lambda a,b: a.union(b), seq, set())
    keys = flatten([set(d.keys()) for d in dicts])
    values_for_key = lambda k: [d[k] for d in dicts if k in d.keys()]
    return dict([(k,torch.tensor(values_for_key(k))) for k in keys])

class AsyncFactBaseChecker: # multi-process of check_factbase();
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

class EdgeTypeTransformerLayer(torch.nn.Module): #
    def __init__(self, hidden_dim, d_inner, num_edge_types, n_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = d_inner

        # initialise layers
        self.layers = []
        for j in range(num_edge_types):
            l = GATConv(self.hidden_dim, self.hidden_dim, n_heads, False)
            #l = GCNConv(self.hidden_dim, self.hidden_dim, aggr="max")
            #l = MaxGraphLayer(self.hidden_dim)
            self.layers.append(l)
            self.add_module("layer_edge_type_" + str(j), l)

        # initialise norm
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
            index = edge_index[:,indices]
            return index

        def layer(l, x, ei):
            return l(x, ei)

        # edge type attention
        x2 = torch.stack([layer(l, x.view(-1, self.hidden_dim), edge_index_for_type(t)) for t,l in enumerate(self.layers)], axis=0).sum(axis=0)
        x = x + self.drop1(x2)
        x = self.norm1.forward(x.view(-1, self.hidden_dim)).view(x.shape)
        x = self.linear2(self.drop(torch.relu(self.linear1(x))))
        x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)

        return x

class SimpleGATLayer(torch.nn.Module):
    """
    纯 GAT 层，不区分边类型
    对所有边使用同一个注意力机制
    """

    def __init__(self, hidden_dim, d_inner, n_heads=8, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = d_inner

        # 只创建一个 GAT 层，处理所有边
        self.gat_layer = GATConv(self.hidden_dim, self.hidden_dim, n_heads, concat=False)

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
        注意：这里移除了 edge_type 参数
        """
        # GAT 注意力传播（所有边一起处理，不区分类型）
        x2 = self.gat_layer(x.view(-1, self.hidden_dim), edge_index)

        # 残差连接 + LayerNorm
        x = x + self.drop1(x2)
        x = self.norm1.forward(x.view(-1, self.hidden_dim)).view(x.shape)

        # FFN
        x = x + self.drop2(self.linear2(self.drop(torch.relu(self.linear1(x)))))
        x = self.norm2(x.view(-1, self.hidden_dim)).view(x.shape)

        return x

# Their GraphTransformer is learning: "how the network works":
# 1) Structural relationships in network topology: which routers are connected and what are the types of connections;
# 2) Propagation patterns of routing protocols: how BGP/OSPF information propagates within the network;
# 3) Dependencies between configuration parameters: how a router's OSPF weight affects the routing decisions of other routers;
# 4) Global constraints required by the specification: how various parameters should be coordinated to meet reachability requirements;
class EdgeTypeGraphTransformer(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, num_edge_types):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.propagation_layers = [EdgeTypeTransformerLayer(hidden_dim, 4*hidden_dim, num_edge_types, 8) for i in range(num_layers)]
        for i, l in enumerate(self.propagation_layers):
            self.add_module("prop_layer_" + str(i), l)

    def forward(self, x, edge_index, edge_type):
        for l in self.propagation_layers:
            x = l.forward(x, edge_index, edge_type)
            x = torch.relu(x)
        return x

class SimpleGATEncoder(torch.nn.Module):
    """
    纯 GAT 编码器，堆叠多层 SimpleGATLayer
    不使用边类型信息
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.propagation_layers = [
            SimpleGATLayer(hidden_dim, 4 * hidden_dim, n_heads=8)
            for i in range(num_layers)
        ]

        for i, l in enumerate(self.propagation_layers):
            self.add_module("prop_layer_" + str(i), l)

    def forward(self, x, edge_index):
        """
        注意：这里移除了 edge_type 参数
        """
        for l in self.propagation_layers:
            x = l.forward(x, edge_index)  # 不再传递 edge_type
            x = torch.relu(x)
        return x

# Encode the original features of network nodes (router ID, IP address, protocol parameters, etc.) into high-dimensional vectors
# this class should be the "EMB function" in paper's Section 4.2
class PredicateGraphEmbedding(torch.nn.Module):
    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices=set(), num_layers=6):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self.encoder = NodeFeatureEmbedding(hidden_dim, features, excluded_feature_indices=excluded_feature_indices)
        self.decoder = NodeFeatureDecoder(hidden_dim, features, excluded_feature_indices=excluded_feature_indices)

    def forward(self, x, mask, edge_index, edge_type, reliable_masking):
        return self.encoder.forward(x, mask, reliable_masking, positional_encoding=False)

class Model_fd(torch.nn.Module): # maybe compare with the original "class Model()" in "synthesis task" to understand this code
    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices):
        super().__init__()

        self.hidden_dim = hidden_dim # hidden dimension of the neural network controls the model capacity
        self.num_edge_types = num_edge_types # The number of edge types in the graph (4 in the code)

        self.embedding = PredicateGraphEmbedding(features, hidden_dim, self.num_edge_types, excluded_feature_indices) # Create a node embedding layer to convert the original features into high-dimensional vectors

        self.transformer_encoder = EdgeTypeGraphTransformer(self.hidden_dim, 2, num_edge_types)

        # C sompared with the synthesis tasks, we delete all the decoders

        # In the following, we define the fault classifier, which is a simple multilayer perceptron.
            # Input: The overall representation vector of the graph.
            # Output: Scores (logits) for the seven fault classes.
        self.fault_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),   # Fully connected layer: 128 -> 64
            torch.nn.ReLU(),                                # Activation Function
            torch.nn.Dropout(0.2),                          # To prevent overfitting, randomly discard 20%
            torch.nn.Linear(hidden_dim // 2, 7)             # Final classification layer: 64 -> 7 fault categories
        )

    def forward(self, x, edge_index, edge_type, batch_idx): # this "forward" is for classification task
        # x:            Node feature matrix [number of nodes, number of features]
        # edge_index:   Edge connection relationship [2, number of edges]
        # edge_type:    The type of each edge [number of edges]
        # batch_idx:    Which graph each node belongs to [number of nodes] (due to batch processing)

        # if edge_index.dim() == 3: # If edge_index is 3-dimensional (batch dimension), take the first batch
        #     edge_index = edge_index[0]
        #     edge_type = edge_type[0]

        # if x.dim() == 2: # If x is 2D [number of nodes, number of features], it is expanded to 3D [number of nodes, 1, number of features]; This is because the embedding layer expects 3D input.
        #     x = x.unsqueeze(1)

        # if edge_index.dim() == 3:
        #     if edge_index.size(0) != 1:
        #         raise ValueError("Batched edge_index not supported yet")
        #     edge_index = edge_index[0]
        #     edge_type = edge_type[0]
        # assert x.size(1) == 1, f"Expected sequence length 1, got {x.size(1)}"

        mask = torch.zeros_like(x, dtype=torch.bool) # 创建一个全零的mask（表示不遮盖任何特征）
        x = self.embedding.forward(x, mask, edge_index, edge_type, True)[:, 0] # 特征编码
        # [:, 0] removes the dimension just added, changing from [number of nodes, 1, hidden_dim] back to [number of nodes, hidden_dim]
        # The final True parameter indicates the use of a reliable masking method

        x = self.transformer_encoder.forward(x, edge_index, edge_type) # 图推理
        # Core Steps: Inference via Graph Transformers
        # Let each node's representation incorporate information from neighboring nodes
        # Learn dependencies in the network topology

        graph_representation = global_mean_pool(x, batch_idx) # Graph-level pooling
        fault_logits = self.fault_classifier(graph_representation) # Fault classification

        return fault_logits

class Model_fd_classical_GAT(torch.nn.Module):
    """
    使用经典 GAT 的故障诊断模型（baseline）
    不使用边类型信息
    """

    def __init__(self, features, hidden_dim, num_edge_types, excluded_feature_indices):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types  # 保留这个参数（为了兼容性），但不使用

        # Embedding 层（保持不变）
        self.embedding = PredicateGraphEmbedding(
            features, hidden_dim, self.num_edge_types, excluded_feature_indices
        )

        # 使用简单的 GAT 编码器（不区分边类型）
        self.transformer_encoder = SimpleGATEncoder(self.hidden_dim, num_layers=2)

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
        # 注意：embedding 内部可能还需要 edge_type，所以这里还是传递

        # 图推理（不使用 edge_type）
        x = self.transformer_encoder.forward(x, edge_index)  # 关键：不传递 edge_type

        # 图级别池化
        graph_representation = global_mean_pool(x, batch_idx)

        # 故障分类
        fault_logits = self.fault_classifier(graph_representation)

        return fault_logits

# def generate_fake_fault_labels(batch): # Temporarily generate a fake fault label
#     batch_size = batch.num_graphs   # PyG 自带的属性，表示batch里有多少个graph
#     return torch.randint(0, 7, (batch_size,), device=batch.x.device)

def inject_single_graph_fault_and_label(batch, feature, num_parameters=7, device='cpu'):
    """
    batch: PyG Data batch
    feature: prog.feature_registry.feature
    num_parameters: 总共可出错参数数量
    """
    batch_size = batch.num_graphs
    labels = torch.randint(0, num_parameters, (batch_size,), device=device)  # 每个图随机选择一个参数出错

    # 对每个图进行注入
    for i in range(batch_size):
        fault_idx = labels[i].item() # 哪个参数出错
        # 根据 index 找到 feature 对应的张量列
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

        # 只修改属于第i个图的节点
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

            # === 和训练一致：注入一个 fault 并得到标签 ===
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
    criterion = torch.nn.CrossEntropyLoss(reduction='none')  # ← 改成 'none'，返回每个样本的loss

    # 用于存储每个样本的结果
    all_losses = []  # 每个样本的loss
    all_predictions = []  # 每个样本的预测结果
    all_labels = []  # 每个样本的真实标签
    all_correct = []  # 每个样本是否预测正确 (True/False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = batch.x.unsqueeze(1)
            batch.edge_index = reflexive(bidirectional(batch.edge_index), num_nodes=batch.x.size(0))
            batch.edge_type = reflexive_bidirectional_edge_type(batch.edge_type, batch.x.size(0))

            # 注入故障
            batch.x, fault_labels = inject_single_graph_fault_and_label(
                batch, features, num_parameters=num_parameters, device=device)

            # 预测
            fault_predictions = model.forward(batch.x, batch.edge_index, batch.edge_type, batch.batch)

            # 计算每个样本的loss（不求和）
            losses = criterion(fault_predictions, fault_labels)  # shape: [batch_size]

            # 获取预测结果
            _, predicted = torch.max(fault_predictions.data, 1)

            # 判断是否预测正确
            correct = (predicted == fault_labels)  # shape: [batch_size], bool tensor

            # 保存到列表（转换为numpy便于后续处理）
            all_losses.extend(losses.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(fault_labels.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())

    # 转换为numpy数组
    all_losses = np.array(all_losses)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_correct = np.array(all_correct)

    # 计算统计量
    mean_loss = all_losses.mean()
    std_loss = all_losses.std()
    accuracy = 100 * all_correct.sum() / len(all_correct)

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"{prefix} Results:")
    print(f"  Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"  Accuracy:  {accuracy:.2f}% ({all_correct.sum()}/{len(all_correct)})")
    print(f"  Min Loss:  {all_losses.min():.4f}")
    print(f"  Max Loss:  {all_losses.max():.4f}")
    print(f"{'=' * 60}\n")

    # 写入TensorBoard
    step_writer.add_scalar(f"{prefix}/Loss_Mean", mean_loss)
    step_writer.add_scalar(f"{prefix}/Loss_Std", std_loss)
    step_writer.add_scalar(f"{prefix}/Accuracy", accuracy)

    # 返回详细结果
    results = {
        'losses': all_losses,  # [100] 每个样本的loss
        'predictions': all_predictions,  # [100] 每个样本的预测
        'labels': all_labels,  # [100] 每个样本的真实标签
        'correct': all_correct,  # [100] 每个样本是否正确
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'accuracy': accuracy
    }

    return results

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

    # === 这里你要设定 fault classification 的类别数量（比如 7） ===
    NUM_FAULT_CLASSES = 7

    # Creating a fault diagnosis model
    # model = Model_fd(features, HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices).to(device)
    model = Model_fd_classical_GAT(features, HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices).to(device)
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

    # main loop
    visual_loss = []
    visual_accuracy = []
    for epoch in tqdm(range(args.epochs), leave=False):
        training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        step_writer = StepWriter(writer, epoch)

        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for i, batch in tqdm(enumerate(training_loader), leave=False, total=num_batches_per_epoch, desc=f"Epoch {epoch}"):
            if i > num_batches_per_epoch: break

            optimizer.zero_grad()

            # Data preprocessing
            batch = batch.to(device)
            batch.x = batch.x.unsqueeze(1)

            # Graph structure enhancement: 1) "bidirectional" adds an edge; 2) "reflexive" adds self-loop
            batch.edge_index = reflexive(bidirectional(batch.edge_index), num_nodes=batch.x.size(0))
            batch.edge_type = reflexive_bidirectional_edge_type(batch.edge_type, batch.x.size(0))

            batch.x, fault_labels = inject_single_graph_fault_and_label(batch, feature, num_parameters=NUM_FAULT_CLASSES, device=device)
            # fault_labels = generate_fake_fault_labels(batch).to(device) # Generate temporary tags

            # Forward propagation
            fault_predictions = model.forward(batch.x, batch.edge_index, batch.edge_type, batch.batch) # Forward propagation
            loss = criterion(fault_predictions, fault_labels) # calculate the loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            # statistics of results
            epoch_loss += loss.item()
            _, predicted = torch.max(fault_predictions.data, 1)
            total += fault_labels.size(0)
            correct += (predicted == fault_labels).sum().item()

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / num_batches_per_epoch
        accuracy = 100 * correct / total if total > 0 else 0

        # Recording the metrics
        step_writer.add_scalar("Loss/Train", avg_loss)
        step_writer.add_scalar("Accuracy/Train", accuracy)

        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

        # for visualise in the paper
        visual_loss.append(avg_loss)
        visual_accuracy.append(accuracy)

    ######################### run the evaluation code in another script ############################
        # Validation by "eval_fd()"
        if epoch % 20 == 0 or epoch < 20 or epoch == NUM_TRAINING_EPOCH-1:
            eval_fd(model, validation_dataset, feature, NUM_FAULT_CLASSES, step_writer, prefix="Val", device=device)
            results = eval_fd_detailed(model, validation_dataset, feature, NUM_FAULT_CLASSES, step_writer, prefix="Val", device=device)

            # 保存详细结果到文件
            np.savez(f"./results/val_GAT2018_epoch{epoch}.npz",
                     losses=results['losses'],
                     predictions=results['predictions'],
                     labels=results['labels'],
                     correct=results['correct'])

            # Save the model
            uid = os.environ.get("EXP_ID", "FD")
            torch.save([model.state_dict(), HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices], f"models/{uid}-GAT2018-epoch{epoch}.pt")

        writer.flush()

    # uid = os.environ.get("EXP_ID", "fault_localization")
    # os.makedirs("models", exist_ok=True)
    # torch.save([model.state_dict(), HIDDEN_DIM, NUM_EDGE_TYPES, excluded_feature_indices], f"models/{uid}-model-GAT2018-final.pt")

    print("Training completed!")