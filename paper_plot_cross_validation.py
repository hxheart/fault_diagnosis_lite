import numpy as np
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ===== 字体大小配置 =====
FONTSIZE_TITLE = 16          # 总标题
FONTSIZE_SUBTITLE = 14       # 子图标题
FONTSIZE_LABEL = 24          # 轴标签
FONTSIZE_TICK = 20           # 刻度标签
FONTSIZE_LEGEND = 30         # 图例
FONTSIZE_TEXT = 14           # 文本提示

# ===== 配置参数 =====
# 文件名使用的算法名（用于读取数据）
algorithms_file = ['GAT2018', 'GATv2', 'EtaGAT', 'EtaGATv2']
# 显示使用的算法名（用于图例）
algorithms_display = ['GAT', 'GATv2', 'EtaGAT', 'EtaGATv2']

datasets = ['origin', 'larger', 'real']
dataset_titles = {
    'origin': 'Original Dataset',
    'larger': 'Larger Dataset', 
    'real': 'Real-World Dataset'
}

# 只显示这些epoch
EPOCHS_TO_SHOW = [2, 4, 6, 8]

# 颜色方案 - 更浅的颜色配合图案
colors = {
    'GAT': '#6B9BD1',      # 浅蓝色
    'GATv2': '#F28B8C',    # 浅红色
    'EtaGAT': '#7DC97D',   # 浅绿色
    'EtaGATv2': '#C77BA8'  # 浅紫色
}

# 添加填充图案
hatches = {
    'GAT': '',        # 无图案
    'GATv2': '//',    # 右斜线
    'EtaGAT': '\\\\', # 左斜线
    'EtaGATv2': 'xx'  # 交叉线
}

# ===== 加载数据函数 =====
def load_algorithm_data(algorithm_file, dataset):
    """
    加载指定算法和数据集的验证结果
    
    Args:
        algorithm_file: 算法文件名 (GAT2018, GATv2, EtaGAT, EtaGATv2)
        dataset: 数据集名称 (origin, larger, real)
    
    Returns:
        epochs: epoch列表
        mean_accs: 平均准确率
        std_accs: 准确率标准差（如果没有则为0）
    """
    filepath = f'./fig_data/summary_{algorithm_file}_{dataset}.npz'
    
    if not os.path.exists(filepath):
        print(f"⚠️  文件不存在: {filepath}")
        return None, None, None
    
    try:
        data = np.load(filepath)
        
        # 根据你的存储格式读取数据
        if 'epochs' in data and 'accuracies' in data:
            epochs = data['epochs']
            accuracies = data['accuracies']
            
            # 检查是否有数据
            if len(epochs) == 0 or len(accuracies) == 0:
                print(f"⚠️  数据为空: {filepath}")
                return None, None, None
            
            # accuracies 已经是百分比格式
            mean_accs = accuracies
            
            # 如果有标准差数据就用，否则设为0
            if 'std_accuracies' in data:
                std_accs = data['std_accuracies']
                print(f"✅ 加载成功: {algorithm_file} - {dataset}, {len(epochs)} epochs, 标准差范围: [{np.min(std_accs):.3f}, {np.max(std_accs):.3f}]")
            else:
                std_accs = np.zeros_like(mean_accs)
                print(f"✅ 加载成功: {algorithm_file} - {dataset}, {len(epochs)} epochs, 无标准差数据")
        
        else:
            print(f"⚠️  文件格式不符合预期: {filepath}")
            print(f"   可用的键: {list(data.keys())}")
            return None, None, None
        
        return epochs, mean_accs, std_accs
        
    except Exception as e:
        print(f"❌ 加载失败: {filepath}, 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ===== 绘制柱状图 =====
def plot_validation_comparison(save_path='./results/validation_comparison.pdf'):
    """
    绘制三个柱状图，每个对应一个数据集，图例放在顶部横向排列
    """
    
    # 创建图形 (1行3列)，为顶部图例留出空间
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # 用于存储图例句柄
    legend_handles = []
    legend_labels = []
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # 收集所有算法在该数据集上的数据
        all_data = {}
        all_epochs_set = set()
        
        for algo_file, algo_display in zip(algorithms_file, algorithms_display):
            epochs, mean_accs, std_accs = load_algorithm_data(algo_file, dataset)
            if epochs is not None and len(epochs) > 0:
                all_data[algo_display] = {
                    'epochs': epochs,
                    'mean_accs': mean_accs,
                    'std_accs': std_accs
                }
                all_epochs_set.update(epochs)
        
        if not all_data:
            print(f"⚠️  {dataset} 数据集没有可用数据")
            ax.set_xlabel('Training Epoch', fontsize=FONTSIZE_LABEL)
            ax.set_ylabel('Validation Accuracy (%)', fontsize=FONTSIZE_LABEL)
            continue
        
        # 获取所有epoch并排序，只保留要显示的epoch
        all_epochs_sorted = sorted(all_epochs_set)
        epochs_display = [ep for ep in all_epochs_sorted if ep in EPOCHS_TO_SHOW]
        
        if not epochs_display:
            print(f"⚠️  {dataset} 数据集没有要显示的epoch")
            ax.text(0.5, 0.5, f'No data available for selected epochs',
                   ha='center', va='center', transform=ax.transAxes, fontsize=FONTSIZE_TEXT)
            ax.set_xlabel('Training Epoch', fontsize=FONTSIZE_LABEL)
            ax.set_ylabel('Validation Accuracy (%)', fontsize=FONTSIZE_LABEL)
            continue
        
        n_epochs = len(epochs_display)
        n_algos = len(all_data)
        
        # 设置柱状图参数
        x = np.arange(n_epochs)
        width = 0.8 / n_algos  # 动态调整柱子宽度
        
        # 为每个算法绘制柱子
        for i, (algo, data) in enumerate(all_data.items()):
            offset = (i - n_algos/2 + 0.5) * width
            
            # 为当前算法创建对齐到 epochs_display 的数据
            algo_mean_accs = []
            algo_std_accs = []
            algo_epochs = data['epochs']
            
            for ep in epochs_display:
                if ep in algo_epochs:
                    ep_idx = np.where(algo_epochs == ep)[0][0]
                    algo_mean_accs.append(data['mean_accs'][ep_idx])
                    algo_std_accs.append(data['std_accs'][ep_idx])
                else:
                    # 如果该算法没有这个epoch的数据，填充nan（不显示）
                    algo_mean_accs.append(np.nan)
                    algo_std_accs.append(0)
            
            algo_mean_accs = np.array(algo_mean_accs)
            algo_std_accs = np.array(algo_std_accs)
            
            # 只绘制非nan的数据
            valid_mask = ~np.isnan(algo_mean_accs)
            
            if valid_mask.any():
                # 确保标准差不全为0时才显示误差棒
                show_error = np.any(algo_std_accs[valid_mask] > 0)
                
                bars = ax.bar(x[valid_mask] + offset, 
                             algo_mean_accs[valid_mask], 
                             width,
                             yerr=algo_std_accs[valid_mask] if show_error else None,
                             label=algo,
                             color=colors[algo],
                             alpha=0.7,  # 增加透明度
                             edgecolor='black',
                             linewidth=1.0,
                             hatch=hatches[algo],  # 添加填充图案
                             capsize=5,  # 增大误差棒的帽子大小
                             error_kw={'linewidth': 2, 'ecolor': 'black', 'alpha': 0.8, 'capthick': 2})
                
                # 只在第一个子图收集图例信息
                if idx == 0 and algo not in legend_labels:
                    legend_handles.append(bars)
                    legend_labels.append(algo)
        
        # 设置图表属性
        ax.set_xlabel('Epoch', fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=FONTSIZE_LABEL)
        ax.set_xticks(x)
        ax.set_xticklabels(epochs_display, fontsize=FONTSIZE_TICK)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    
    # 在图形顶部添加横向图例
    if legend_handles:
        fig.legend(legend_handles, legend_labels, 
                  loc='upper center',           # 位置在上方中央
                  bbox_to_anchor=(0.5, 0.98),   # 锚点在顶部中央
                  ncol=4,                        # 4列横向排列
                  fontsize=FONTSIZE_LEGEND,
                  framealpha=0.95,
                  edgecolor='black',
                  fancybox=True,
                  frameon=True)
    
    # 调整子图布局，为顶部图例留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    
    print(f"\n✅ 柱状图已保存到:")
    print(f"   - {save_path}")
    print(f"   - {save_path.replace('.png', '.pdf')}")
    
    plt.show()


# ===== 打印统计表格 =====
def print_statistics_table():
    """
    打印详细的统计表格
    """
    print("\n" + "="*100)
    print("📊 验证结果统计表格")
    print("="*100)
    
    for dataset in datasets:
        print(f"\n📁 {dataset_titles[dataset]}:")
        print("-"*100)
        print(f"{'Algorithm':<15} {'Final Acc (%)':<20} {'Best Acc (%)':<20} {'Avg Acc (%)':<20}")
        print("-"*100)
        
        for algo_file, algo_display in zip(algorithms_file, algorithms_display):
            epochs, mean_accs, std_accs = load_algorithm_data(algo_file, dataset)
            if epochs is not None and len(epochs) > 0:
                final_acc = mean_accs[-1]
                final_std = std_accs[-1]
                best_acc = np.max(mean_accs)
                avg_acc = np.mean(mean_accs)
                
                if final_std > 0:
                    final_str = f"{final_acc:.2f}±{final_std:.2f}"
                else:
                    final_str = f"{final_acc:.2f}"
                
                print(f"{algo_display:<15} {final_str:<20} {best_acc:.2f}{'':16} {avg_acc:.2f}")
            else:
                print(f"{algo_display:<15} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
        
        print("-"*100)
    
    print("="*100)


# ===== 主函数 =====
if __name__ == '__main__':
    print("🎨 开始生成多算法验证结果对比图...\n")
    
    # 生成柱状图
    plot_validation_comparison(save_path='./validation_comparison.pdf')
    
    # 打印统计表格
    print_statistics_table()
    
    print("\n✅ 所有可视化完成！")
    print("\n💡 说明:")
    print(f"   - 显示的epoch: {EPOCHS_TO_SHOW}")
    print("   - 每个子图对应一个测试数据集 (origin/larger/real)")
    print("   - 横轴为训练epoch")
    print("   - 纵轴为验证准确率 (%)")
    print("   - 误差棒显示标准差（如果有的话）")
    print("   - 填充图案: GAT(无), GATv2(//), EtaGAT(\\\\), EtaGATv2(xx)")
    print("   - 图例放在顶部横向排列，节省空间")
    print("   - 如果某算法在某数据集上没有数据，该算法的柱子会被跳过")