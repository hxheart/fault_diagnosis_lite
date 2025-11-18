import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# ===== 滑动平均函数 =====
def smooth_curve(data, window_size=10):
    """
    使用滑动窗口对曲线进行平滑处理
    
    参数:
        data: 原始数据数组
        window_size: 滑动窗口大小（默认10）
    
    返回:
        平滑后的数据
    """
    if len(data) < window_size:
        return data
    
    # 使用卷积进行滑动平均
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='valid')
    
    # 为了保持数组长度一致，在开头补充原始数据
    pad_size = len(data) - len(smoothed)
    smoothed = np.concatenate([data[:pad_size], smoothed])
    
    return smoothed

# ===== IEEE双栏格式配置 =====
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['lines.markersize'] = 3

# ===== 第一部分：读取训练曲线数据 =====
num_runs_begin = 0
num_runs_end   = 5

SMOOTH_WINDOW = 3  # 建议范围：5-20 # 滑动平均窗口大小（可以调整这个参数来控制平滑程度）

# 定义4个算法的配置
algorithms = {
    'GAT_2018': {
        'loss_pattern': './fig_data/visual_GAT_2018_loss_{}.npy',
        'acc_pattern': './fig_data/visual_GAT_2018_accuracy_{}.npy',
        'label': 'GAT',
        'color': '#4E79A7',  # 蓝灰色
    },
    'GATv2_2022': {
        'loss_pattern': './fig_data/visual_GATv2_2022_loss_{}.npy',
        'acc_pattern': './fig_data/visual_GATv2_2022_accuracy_{}.npy',
        'label': 'GATv2',
        'color': '#E15759',  # 红色
    },
    'EtaGAT': {
        'loss_pattern': './fig_data/visual_EtaGAT_loss_{}.npy',
        'acc_pattern': './fig_data/visual_EtaGAT_accuracy_{}.npy',
        'label': 'EtaGAT',
        'color': '#59A14F',  # 绿色
    },
    'EtaGATv2': {
        'loss_pattern': './fig_data/visual_EtaGATv2_loss_{}.npy',
        'acc_pattern': './fig_data/visual_EtaGATv2_accuracy_{}.npy',
        'label': 'EtaGATv2',
        'color': '#A23B72',  # 紫红色
    }
}

# 读取所有算法的数据
algo_data = {}
for algo_name, config in algorithms.items():
    losses = []
    accuracies = []
    
    for i in range(num_runs_begin, num_runs_end):
        loss_path = config['loss_pattern'].format(i)
        acc_path = config['acc_pattern'].format(i)
        
        if not os.path.exists(loss_path) or not os.path.exists(acc_path):
            print(f"⚠️  找不到文件: {loss_path} 或 {acc_path}")
            continue
        
        # 读取数据并进行滑动平均
        loss_data = np.load(loss_path)
        acc_data = np.load(acc_path)
        
        # 对每次运行的数据进行平滑
        losses.append(smooth_curve(loss_data, SMOOTH_WINDOW))
        accuracies.append(smooth_curve(acc_data, SMOOTH_WINDOW))
    
    if losses and accuracies:
        algo_data[algo_name] = {
            'losses': np.array(losses),
            'accuracies': np.array(accuracies),
            'mean_loss': np.mean(losses, axis=0),
            'std_loss': np.std(losses, axis=0),
            'mean_accuracy': np.mean(accuracies, axis=0),
            'std_accuracy': np.std(accuracies, axis=0),
            'config': config
        }
        print(f"✅ {algo_name}: 成功加载 {len(losses)} 次运行的数据（滑动窗口={SMOOTH_WINDOW}）")
    else:
        print(f"❌ {algo_name}: 未找到有效数据")

# ===== 绘制训练曲线图 =====
# ===== 绘制两个独立图：Loss 和 Accuracy =====

# ---------- 图1：Loss ----------
fig_loss = plt.figure(figsize=(4.2, 3.2))
ax_loss = plt.gca()
ax_loss.set_xlabel('Training epochs')
ax_loss.set_ylabel('Loss')

lines_loss = []

for algo_name, data in algo_data.items():
    config = data['config']
    epochs = range(len(data['mean_loss']))
    color = config['color']
    markevery = max(1, len(epochs) // 15)

    line_loss = ax_loss.plot(
        epochs, data['mean_loss'],
        color=color,
        label=config['label'],
        linestyle='-',
        linewidth=1.2,
        markevery=markevery
    )
    ax_loss.fill_between(
        epochs,
        data['mean_loss'] - data['std_loss'],
        data['mean_loss'] + data['std_loss'],
        alpha=0.15,
        color=color
    )
    lines_loss.extend(line_loss)

max_epochs = max([len(data['mean_accuracy']) for data in algo_data.values()])

ax_loss.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax_loss.set_xlim(0, max_epochs)
ax_loss.set_xlim(0, 400)
ax_loss.legend(loc='best', framealpha=0.9, ncol=1)
plt.tight_layout()
plt.savefig('./fig1_training_loss_4algo_moving_avg.pdf', dpi=300, bbox_inches='tight')
print("✅ 图1已保存: fig1_training_loss_4algo_moving_avg.pdf")


# ---------- 图2：Accuracy ----------
fig_acc = plt.figure(figsize=(4.2, 3.2))
ax_acc = plt.gca()
ax_acc.set_xlabel('Training epochs')
ax_acc.set_ylabel('Average accuracy (%)')

lines_acc = []

for algo_name, data in algo_data.items():
    config = data['config']
    epochs = range(len(data['mean_accuracy']))
    color = config['color']
    markevery = max(1, len(epochs) // 15)

    line_acc = ax_acc.plot(
        epochs, data['mean_accuracy'],
        color=color,
        label=config['label'],
        linestyle='-',
        linewidth=1.2,
        markevery=markevery
    )
    ax_acc.fill_between(
        epochs,
        data['mean_accuracy'] - data['std_accuracy'],
        data['mean_accuracy'] + data['std_accuracy'],
        alpha=0.15,
        color=color
    )
    lines_acc.extend(line_acc)

# 在epoch=20处添加竖线
ax_acc.axvline(x=20, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

ax_acc.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax_acc.set_xlim(0,  max_epochs)
ax_acc.set_xlim(0,  60)
ax_acc.set_ylim(20, 100)
ax_acc.legend(loc='best', framealpha=0.9, ncol=1)
plt.tight_layout()
plt.savefig('./fig2_training_accuracy_4algo_moving_avg.pdf', dpi=300, bbox_inches='tight')
print("✅ 图2已保存: fig2_training_accuracy_4algo_moving_avg.pdf")

# plt.show()


# ===== 第二部分：无滑动平均版本 =====
algo_data = {}
for algo_name, config in algorithms.items():
    losses = []
    accuracies = []
    
    for i in range(num_runs_begin, num_runs_end):
        loss_path = config['loss_pattern'].format(i)
        acc_path = config['acc_pattern'].format(i)
        
        if not os.path.exists(loss_path) or not os.path.exists(acc_path):
            print(f"⚠️  找不到文件: {loss_path} 或 {acc_path}")
            continue
        
        losses.append(np.load(loss_path))
        accuracies.append(np.load(acc_path))
    
    if losses and accuracies:
        algo_data[algo_name] = {
            'losses': np.array(losses),
            'accuracies': np.array(accuracies),
            'mean_loss': np.mean(losses, axis=0),
            'std_loss': np.std(losses, axis=0),
            'mean_accuracy': np.mean(accuracies, axis=0),
            'std_accuracy': np.std(accuracies, axis=0),
            'config': config
        }
        print(f"✅ {algo_name}: 成功加载 {len(losses)} 次运行的数据")
    else:
        print(f"❌ {algo_name}: 未找到有效数据")

# ---------- 图3：Loss (无滑动平均) ----------
fig_loss = plt.figure(figsize=(4.2, 3.2))
ax_loss = plt.gca()
ax_loss.set_xlabel('Training epochs')
ax_loss.set_ylabel('Loss')

lines_loss = []

for algo_name, data in algo_data.items():
    config = data['config']
    epochs = range(len(data['mean_loss']))
    color = config['color']
    markevery = max(1, len(epochs) // 15)

    line_loss = ax_loss.plot(
        epochs, data['mean_loss'],
        color=color,
        label=config['label'],
        linestyle='-',
        linewidth=1.2,
        markevery=markevery
    )
    ax_loss.fill_between(
        epochs,
        data['mean_loss'] - data['std_loss'],
        data['mean_loss'] + data['std_loss'],
        alpha=0.15,
        color=color
    )
    lines_loss.extend(line_loss)

max_epochs = max([len(data['mean_accuracy']) for data in algo_data.values()])

ax_loss.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax_loss.set_xlim(0, max_epochs)
ax_loss.set_xlim(0, 400)
ax_loss.legend(loc='best', framealpha=0.9, ncol=1)
plt.tight_layout()
plt.savefig('./fig1_training_loss_4algo.pdf', dpi=300, bbox_inches='tight')
print("✅ 图3已保存: fig1_training_loss_4algo.pdf")


# ---------- 图4：Accuracy (无滑动平均) ----------
fig_acc = plt.figure(figsize=(4.2, 3.2))
ax_acc = plt.gca()
ax_acc.set_xlabel('Training epochs')
ax_acc.set_ylabel('Average accuracy (%)')

lines_acc = []

for algo_name, data in algo_data.items():
    config = data['config']
    epochs = range(len(data['mean_accuracy']))
    color = config['color']
    markevery = max(1, len(epochs) // 15)

    line_acc = ax_acc.plot(
        epochs, data['mean_accuracy'],
        color=color,
        label=config['label'],
        linestyle='-',
        linewidth=1.2,
        markevery=markevery
    )
    ax_acc.fill_between(
        epochs,
        data['mean_accuracy'] - data['std_accuracy'],
        data['mean_accuracy'] + data['std_accuracy'],
        alpha=0.15,
        color=color
    )
    lines_acc.extend(line_acc)

# 在epoch=20处添加竖线
ax_acc.axvline(x=20, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

ax_acc.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax_acc.set_xlim(0, max_epochs)
ax_acc.set_xlim(0, 100)
ax_acc.legend(loc='best', framealpha=0.9, ncol=1)
plt.tight_layout()
plt.savefig('./fig2_training_accuracy_4algo.pdf', dpi=300, bbox_inches='tight')
print("✅ 图4已保存: fig2_training_accuracy_4algo.pdf")

# plt.show()