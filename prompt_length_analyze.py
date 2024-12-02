import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
import os
import sys
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from transformers import PreTrainedTokenizerFast

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 添加tokenizer初始化函数
def init_tokenizer(tokenizer_path: str = "./") -> PreTrainedTokenizerFast:
    """初始化Qwen tokenizer"""
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        print("成功加载tokenizer")
        return tokenizer
    except Exception as e:
        print(f"加载tokenizer失败: {str(e)}")
        raise

def calculate_prompt_length(item: Dict, tokenizer: PreTrainedTokenizerFast) -> int:
    """计算提示词token长度"""
    choices = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item['options'])])
    prompt_text = f"{item['text']}\nQuestion: {item['question']}\n{choices}\nAnswer:"
    return len(tokenizer.encode(prompt_text))

def load_dataset(file_path: Path) -> List[Dict]:
    """加载数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def check_plotting_environment():
    """检查绘图环境配置"""
    try:
        import scienceplots
        print("SciencePlots 版本:", scienceplots.__version__)
        
        # 测试 LaTeX
        plt.style.use(['science', 'ieee'])
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, r'$\alpha_{test}$')
        plt.close()
        
        print("LaTeX 环境检查通过")
        return True
    except Exception as e:
        print(f"环境检查失败: {str(e)}")
        print("\n请确保已经安装：")
        print("1. MiKTeX (https://miktex.org/download)")
        print("2. 必要的 LaTeX 包：cm-super, dvipng, type1cm")
        print("3. SciencePlots: pip install scienceplots --upgrade")
        return False

def set_plot_style():
    """Set global plotting style for scientific visualization"""
    try:
        import scienceplots
        plt.style.use(['science', 'ieee'])
        print("成功应用 SciencePlots 样式")
        
        # 检查系统可用字体
        from matplotlib.font_manager import findfont, FontProperties
        available_fonts = []
        for font in ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Nimbus Sans L']:
            try:
                if findfont(FontProperties(family=font)) is not None:
                    available_fonts.append(font)
            except:
                continue
        
        if not available_fonts:
            print("警告: 未找到首选字体，使用系统默认字体")
            font_family = ['sans-serif']
        else:
            print(f"使用字体: {available_fonts[0]}")
            font_family = [available_fonts[0]] + ['sans-serif']
        
        plt.rcParams.update({
            # 字体设置
            'font.family': font_family,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.title_fontsize': 11,
            
            # 图表元素
            'figure.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linestyle': '--',
            'axes.linewidth': 1.0,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'xtick.major.size': 4.0,
            'ytick.major.size': 4.0,
            'xtick.minor.size': 2.0,
            'ytick.minor.size': 2.0,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            
            # 边框和轴线
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            
            # 图例设置
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': '0.8',
            'legend.borderpad': 0.4,
            'legend.borderaxespad': 0.5,
            'legend.handlelength': 2.0,
            'legend.handletextpad': 0.5,
            'legend.columnspacing': 1.0,
            
            # 图表间距
            'figure.subplot.left': 0.125,
            'figure.subplot.right': 0.9,
            'figure.subplot.bottom': 0.11,
            'figure.subplot.top': 0.88,
            'figure.subplot.wspace': 0.2,
            'figure.subplot.hspace': 0.2,
            
            # 保存设置
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'savefig.dpi': 300,
            
            # 禁用 LaTeX
            'text.usetex': False,
            'mathtext.default': 'regular'
        })
        return True
        
    except Exception as e:
        print(f"无法应用 SciencePlots 样式: {str(e)}")
        print("使用备用样式设置")
        
        plt.style.use('seaborn-paper')
        plt.rcParams.update({
            'font.family': ['sans-serif'],
            # ... (其他设置保持不变)
        })
        return False

def plot_boxplot(df: pd.DataFrame, output_dir: Path):
    """Create publication-quality boxplot"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set color palette with better visibility
    colors = sns.color_palette("husl", n_colors=len(df['dataset'].unique()))
    
    # Create boxplot with enhanced style
    bp = sns.boxplot(x='dataset', y='length', data=df, 
                palette=colors,
                medianprops=dict(color="red", linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='gray', 
                              markersize=3, alpha=0.5, markeredgecolor='none'),
                boxprops=dict(alpha=0.8, edgecolor='black', linewidth=1),
                whiskerprops=dict(linewidth=1.2, linestyle='--'),
                capprops=dict(linewidth=1.2))
    
    # Enhance grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set labels and title with enhanced style
    plt.title('Distribution of Prompt Lengths', pad=20, weight='bold', size=14)
    plt.xlabel('Dataset', labelpad=10, weight='bold', size=12)
    plt.ylabel('Length (characters)', labelpad=10, weight='bold', size=12)
    
    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right', weight='bold')
    
    # Add statistical annotations with enhanced style
    for idx, dataset in enumerate(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]['length']
        stats = {
            'median': np.median(dataset_data),
            'q1': np.percentile(dataset_data, 25),
            'q3': np.percentile(dataset_data, 75)
        }
        plt.text(idx, 
                stats['median'],
                f"Median: {int(stats['median']):,}\nQ1: {int(stats['q1']):,}\nQ3: {int(stats['q3']):,}",
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=9,
                bbox=dict(facecolor='white', 
                         alpha=0.95,
                         edgecolor='gray',
                         boxstyle='round,pad=0.5',
                         linewidth=0.5))
    
    # Add minor ticks and format axis
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Enhance spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / "prompt_length_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_histogram(df: pd.DataFrame, output_dir: Path):
    """Create publication-quality split histogram"""
    fig = plt.figure(figsize=(12, 8))
    
    # Create main plot and subplot with specific ratios
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    ax_main = plt.subplot(gs[0])
    ax_sub = plt.subplot(gs[1])
    
    # Set color palette
    colors = sns.color_palette("husl", n_colors=len(df['dataset'].unique()))
    
    # Calculate ranges
    main_range = np.percentile(df['length'], 95)
    bins = np.linspace(0, main_range, 40)
    
    # Plot main distribution with enhanced style
    has_data = False
    for dataset, color in zip(sorted(df['dataset'].unique()), colors):
        dataset_data = df[df['dataset'] == dataset]['length']
        main_data = dataset_data[dataset_data <= main_range]
        if len(main_data) > 0:
            try:
                sns.kdeplot(data=main_data,
                           ax=ax_main,
                           label=dataset,
                           color=color,
                           alpha=0.7,
                           linewidth=2,
                           fill=True,
                           warn_singular=False)  # 禁用奇异性警告
                has_data = True
            except Exception as e:
                print(f"Warning: Could not plot KDE for {dataset} in main plot: {str(e)}")
    
    # Plot tail distribution using histograms instead of KDE
    bins_tail = np.linspace(main_range, df['length'].max(), 20)
    for dataset, color in zip(sorted(df['dataset'].unique()), colors):
        dataset_data = df[df['dataset'] == dataset]['length']
        tail_data = dataset_data[dataset_data > main_range]
        if len(tail_data) > 0:
            ax_sub.hist(tail_data,
                       bins=bins_tail,
                       color=color,
                       alpha=0.7,
                       label=dataset)
    
    # Only add legend if we have data to show
    if has_data:
        ax_main.legend(title='Dataset',
                      title_fontsize=11,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left',
                      frameon=True,
                      fancybox=True,
                      edgecolor='0.8',
                      fontsize=10)
    
    # Enhance grid and style
    for ax in [ax_main, ax_sub]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    # Style main plot
    ax_main.set_title('Distribution of Prompt Lengths', pad=15, weight='bold', size=14)
    ax_main.set_xlabel('')
    ax_main.set_ylabel('Density', labelpad=10, weight='bold', size=12)
    
    # Style subplot
    ax_sub.set_xlabel('Length (characters)', labelpad=10, weight='bold', size=12)
    ax_sub.set_ylabel('Count\n(tail)', labelpad=10, weight='bold', size=12)  # Changed from Density to Count
    
    # Format axis labels
    for ax in [ax_main, ax_sub]:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.tick_params(labelsize=10)
    
    # Add break marks with enhanced style
    d = .015
    kwargs = dict(transform=ax_main.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax_main.plot((-d, +d), (-d, +d), **kwargs)
    ax_main.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax_sub.transAxes)
    ax_sub.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_sub.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    # Set figure size and spacing
    fig.set_size_inches(15, 8)
    plt.subplots_adjust(right=0.85, hspace=0.1)
    
    # Save figure
    plt.savefig(output_dir / "prompt_length_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_stats_markdown(df: pd.DataFrame, output_dir: Path):
    """保存统计信息为markdown格式"""
    stats_file = output_dir / "prompt_length_stats.md"
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("# 提示词长度统计信息\n\n")
        
        # 总体统计
        f.write("## 总体统计\n\n")
        f.write("| 统计指标 | 数值 |\n")
        f.write("|---------|------|\n")
        f.write(f"| 总样本数 | {len(df):,} |\n")
        f.write(f"| 数据集数量 | {len(df['dataset'].unique())} |\n")
        f.write(f"| 最短长度 | {int(df['length'].min()):,} |\n")
        f.write(f"| 最长长度 | {int(df['length'].max()):,} |\n")
        f.write(f"| 平均长度 | {int(df['length'].mean()):,} |\n")
        f.write(f"| 中位数长度 | {int(df['length'].median()):,} |\n")
        f.write("\n")
        
        # 各数据集统计
        f.write("## 数据集详细统计\n\n")
        f.write("| 数据集 | 样本数量 | 最小长度 | 最大长度 | 平均长度 | 中位数 | 标准差 | Q1 (25%) | Q3 (75%) |\n")
        f.write("|--------|----------|----------|----------|----------|---------|---------|-----------|----------|\n")
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_data = df[df['dataset'] == dataset]['length']
            f.write(f"| {dataset} | {len(dataset_data):,} | "
                   f"{int(dataset_data.min()):,} | {int(dataset_data.max()):,} | "
                   f"{int(dataset_data.mean()):,} | {int(np.median(dataset_data)):,} | "
                   f"{int(dataset_data.std()):,} | {int(np.percentile(dataset_data, 25)):,} | "
                   f"{int(np.percentile(dataset_data, 75)):,} |\n")
        
        # 长度分布区间计
        f.write("\n## 长度分布区间统计\n\n")
        bins = [0, 500, 1000, 2000, 5000, float('inf')]
        labels = ['0-500', '501-1000', '1001-2000', '2001-5000', '5000+']
        
        f.write("| 数据集 | " + " | ".join(labels) + " |\n")
        f.write("|--------|" + "|".join(["-" * 10] * len(labels)) + "|\n")
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_data = df[df['dataset'] == dataset]['length']
            hist, _ = np.histogram(dataset_data, bins=bins)
            percentages = hist / len(dataset_data) * 100
            f.write(f"| {dataset} | " + " | ".join([f"{p:.1f}%" for p in percentages]) + " |\n")

def plot_stacked_histogram(df: pd.DataFrame, output_dir: Path):
    """Create publication-quality stacked histogram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set color palette
    colors = sns.color_palette("Set2", n_colors=len(df['dataset'].unique()))
    
    # Calculate bins
    bins = np.linspace(0, np.percentile(df['length'], 99), 40)  # Use 99th percentile for better visualization
    
    # Get datasets and sort by median length
    datasets = sorted(df['dataset'].unique(),
                     key=lambda x: df[df['dataset'] == x]['length'].median())
    
    # Create stacked histogram
    plt.hist([df[df['dataset'] == dataset]['length'] for dataset in datasets],
             bins=bins,
             label=datasets,
             stacked=True,
             alpha=0.8,
             color=colors,
             edgecolor='white',
             linewidth=0.5)
    
    # Set labels and title
    plt.title('Stacked Distribution of Prompt Lengths', pad=15)
    plt.xlabel('Length (characters)', labelpad=10)
    plt.ylabel('Frequency', labelpad=10)
    
    # Optimize legend
    plt.legend(title='Dataset',
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              frameon=True,
              fancybox=True,
              edgecolor='0.8')
    
    # Format x-axis with thousand separators
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / "prompt_length_stacked_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_dataset_composition(df: pd.DataFrame, output_dir: Path):
    """Create pie chart for dataset composition"""
    # Calculate dataset proportions
    dataset_counts = df['dataset'].value_counts()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot main pie chart
    colors = sns.color_palette("husl", n_colors=len(dataset_counts))
    wedges, texts, autotexts = ax1.pie(dataset_counts.values, 
                                      labels=dataset_counts.index,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85,
                                      wedgeprops=dict(width=0.5, edgecolor='white'))
    
    # Enhance pie chart style
    plt.setp(autotexts, size=9, weight='bold')
    plt.setp(texts, size=10)
    
    # Add title
    ax1.set_title('Dataset Composition', pad=20, weight='bold', size=14)
    
    # Create length distribution pie chart
    bins = [0, 500, 1000, 2000, 5000, float('inf')]
    labels = ['0-500', '501-1000', '1001-2000', '2001-5000', '5000+']
    
    # Calculate length distribution
    hist, _ = np.histogram(df['length'], bins=bins)
    total = len(df)
    percentages = hist / total * 100
    
    # Plot length distribution pie chart
    colors = sns.color_palette("Set2", n_colors=len(labels))
    wedges, texts, autotexts = ax2.pie(percentages,
                                      labels=labels,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85,
                                      wedgeprops=dict(width=0.5, edgecolor='white'))
    
    # Enhance pie chart style
    plt.setp(autotexts, size=9, weight='bold')
    plt.setp(texts, size=10)
    
    # Add title
    ax2.set_title('Length Distribution', pad=20, weight='bold', size=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / "dataset_composition.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_split_boxplot(df: pd.DataFrame, output_dir: Path):
    """Create split boxplots for different length ranges"""
    # 计算每个数据集的中位数长度
    median_lengths = df.groupby('dataset')['length'].median()
    
    # 根据中位数将数据集分成两组
    threshold = median_lengths.median()
    short_datasets = median_lengths[median_lengths <= threshold].index
    long_datasets = median_lengths[median_lengths > threshold].index
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])
    
    # 设置颜色
    colors = sns.color_palette("husl", n_colors=len(df['dataset'].unique()))
    color_dict = dict(zip(df['dataset'].unique(), colors))
    
    # 绘制短文本数据集的箱线图
    short_data = df[df['dataset'].isin(short_datasets)]
    sns.boxplot(x='dataset', y='length', data=short_data, 
                ax=ax1,
                palette=[color_dict[d] for d in short_datasets],
                medianprops=dict(color="red", linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='gray', 
                              markersize=3, alpha=0.5, markeredgecolor='none'),
                boxprops=dict(alpha=0.8, edgecolor='black', linewidth=1))
    
    # 绘制长文本数据集的箱线图
    long_data = df[df['dataset'].isin(long_datasets)]
    sns.boxplot(x='dataset', y='length', data=long_data, 
                ax=ax2,
                palette=[color_dict[d] for d in long_datasets],
                medianprops=dict(color="red", linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='gray', 
                              markersize=3, alpha=0.5, markeredgecolor='none'),
                boxprops=dict(alpha=0.8, edgecolor='black', linewidth=1))
    
    # 设置标题和标签
    ax1.set_title('Short Text Datasets (Length ≤ {:,})'.format(int(threshold)), 
                 pad=20, weight='bold', size=14)
    ax2.set_title('Long Text Datasets (Length > {:,})'.format(int(threshold)), 
                 pad=20, weight='bold', size=14)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('Dataset', labelpad=10, weight='bold', size=12)
        ax.set_ylabel('Length (characters)', labelpad=10, weight='bold', size=12)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # 添加统计注释
    for ax, data in [(ax1, short_data), (ax2, long_data)]:
        for idx, dataset in enumerate(data['dataset'].unique()):
            dataset_data = data[data['dataset'] == dataset]['length']
            stats = {
                'median': np.median(dataset_data),
                'q1': np.percentile(dataset_data, 25),
                'q3': np.percentile(dataset_data, 75)
            }
            ax.text(idx, 
                   stats['median'],
                   f"Median: {int(stats['median']):,}\nQ1: {int(stats['q1']):,}\nQ3: {int(stats['q3']):,}",
                   horizontalalignment='center',
                   verticalalignment='bottom',
                   fontsize=9,
                   bbox=dict(facecolor='white', 
                            alpha=0.95,
                            edgecolor='gray',
                            boxstyle='round,pad=0.5',
                            linewidth=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / "prompt_length_split_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_nested_pie_chart(df: pd.DataFrame, output_dir: Path):
    """Create nested pie chart for dataset composition and length distribution"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 计算数据集比例（外环）
    dataset_counts = df['dataset'].value_counts()
    
    # 计算每数据集的长度分布（内环）
    bins = [0, 500, 1000, 2000, 5000, float('inf')]
    labels = ['0-500', '501-1000', '1001-2000', '2001-5000', '5000+']
    
    # 准备内环数据
    inner_data = []
    inner_colors = []
    inner_labels = []
    
    # 设置颜色方案
    outer_colors = sns.color_palette("husl", n_colors=len(dataset_counts))
    length_colors = sns.color_palette("Set2", n_colors=len(bins)-1)
    
    for dataset, outer_color in zip(dataset_counts.index, outer_colors):
        dataset_data = df[df['dataset'] == dataset]['length']
        hist, _ = np.histogram(dataset_data, bins=bins)
        for count, label, color in zip(hist, labels, length_colors):
            if count > 0:  # 只添加非零数据
                inner_data.append(count)
                inner_colors.append(color)
                inner_labels.append(f"{dataset}\n{label}")
    
    # 绘制嵌套饼图
    # 外环：数据集分布
    outer_wedges, outer_texts, outer_autotexts = ax.pie(dataset_counts.values,
                                                       radius=1.3,
                                                       labels=dataset_counts.index,
                                                       colors=outer_colors,
                                                       autopct='%1.1f%%',
                                                       pctdistance=0.85,
                                                       wedgeprops=dict(width=0.5, edgecolor='white'))
    
    # 内环：长度分布
    inner_wedges, inner_texts, inner_autotexts = ax.pie(inner_data,
                                                       radius=0.8,
                                                       labels=inner_labels,
                                                       colors=inner_colors,
                                                       autopct='%1.1f%%',
                                                       pctdistance=0.75,
                                                       wedgeprops=dict(width=0.5, edgecolor='white'))
    
    # 设置文本样式
    plt.setp(outer_autotexts, size=9, weight='bold')
    plt.setp(outer_texts, size=10)
    plt.setp(inner_autotexts, size=8)
    plt.setp(inner_texts, size=8)
    
    # 添加图例
    length_legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='none') 
                            for color in length_colors]
    ax.legend(length_legend_elements, labels,
             title='Length Ranges',
             loc='center left',
             bbox_to_anchor=(1, 0, 0.5, 1))
    
    # 添加标题
    plt.title('Dataset Composition and Length Distribution', 
             pad=20, size=14, weight='bold')
    
    plt.savefig(output_dir / "nested_pie_chart.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_merged_dataset():
    """分析合并后的数据集"""
    merged_file = Path("converted/merged_dataset.jsonl")
    if not merged_file.exists():
        print("未找到合并数据集文件")
        return
    
    print("\n开始分析合并数据集...")
    
    # 初始化tokenizer
    try:
        tokenizer = init_tokenizer()
    except Exception as e:
        print(f"初始化tokenizer失败: {str(e)}")
        return
    
    # 加载数据
    data = load_dataset(merged_file)
    print(f"合并数据集样本总数: {len(data)}")
    
    # 计算token长度并创建DataFrame
    lengths = [(calculate_prompt_length(item, tokenizer), item.get('source', 'unknown')) 
              for item in data]
    df = pd.DataFrame(lengths, columns=['length', 'dataset'])
    
    # 创建输出目录
    output_dir = Path("analysis/merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制箱线图
    plot_boxplot(df, output_dir)
    print(f"合并数据集箱线图已保存至: {output_dir}/prompt_length_boxplot.png")
    
    # 绘制累加直方图
    plot_stacked_histogram(df, output_dir)
    print(f"合并数据集累加直方图已保存至: {output_dir}/prompt_length_stacked_histogram.png")
    
    # 保存统计信息
    save_stats_markdown(df, output_dir)
    print(f"合并数据集统计信息已保存至: {output_dir}/prompt_length_stats.md")
    
    # 添加饼图分析
    plot_dataset_composition(df, output_dir)
    print(f"数据集组成饼图已保存至: {output_dir}/dataset_composition.png")
    
    # 使用嵌套饼图替换原来的饼图
    plot_nested_pie_chart(df, output_dir)
    print(f"嵌套饼图已保存至: {output_dir}/nested_pie_chart.png")
    
    # 额外添加数据集组成分析
    composition_file = output_dir / "dataset_composition.md"
    with open(composition_file, 'w', encoding='utf-8') as f:
        f.write("# 数据集组成分析\n\n")
        
        # 计算每个数据集的样本数量和占比
        dataset_counts = df['dataset'].value_counts()
        total_samples = len(df)
        
        f.write("## 各数据集样本数量及占比\n\n")
        f.write("| 数据集 | 样本数量 | 占比 |\n")
        f.write("|--------|----------|------|\n")
        
        for dataset, count in dataset_counts.items():
            percentage = (count / total_samples) * 100
            f.write(f"| {dataset} | {count:,} | {percentage:.2f}% |\n")
        
        # 添加总计行
        f.write(f"| **总计** | {total_samples:,} | 100% |\n\n")
        
        # 添加长度区间分布
        f.write("\n## 长度区间分布\n\n")
        bins = [0, 500, 1000, 2000, 5000, float('inf')]
        labels = ['0-500', '501-1000', '1001-2000', '2001-5000', '5000+']
        
        # 计算总体的长度分布
        total_hist, _ = np.histogram(df['length'], bins=bins)
        total_percentages = total_hist / len(df) * 100
        
        f.write("### 总体分布\n\n")
        f.write("| 长度区间 | 样本数量 | 占比 |\n")
        f.write("|----------|----------|------|\n")
        for label, count, percentage in zip(labels, total_hist, total_percentages):
            f.write(f"| {label} | {count:,} | {percentage:.2f}% |\n")
        
        # 计算各数据集的长度分布
        f.write("\n### 各数据集长度分布\n\n")
        f.write("| 数据集 | " + " | ".join(labels) + " |\n")
        f.write("|--------|" + "|".join(["-" * 10] * len(labels)) + "|\n")
        
        for dataset in sorted(df['dataset'].unique()):
            dataset_data = df[df['dataset'] == dataset]['length']
            hist, _ = np.histogram(dataset_data, bins=bins)
            percentages = hist / len(dataset_data) * 100
            f.write(f"| {dataset} | " + " | ".join([f"{p:.1f}%" for p in percentages]) + " |\n")
    
    print(f"数据集组成分析已保存至: {composition_file}")

def analyze_prompt_lengths():
    """分析各数据集的提示词长度分布"""
    # 设置全局绘图样式
    set_plot_style()
    
    # 切换到脚本所在目录
    os.chdir(SCRIPT_DIR)
    
    # 初始化tokenizer
    try:
        tokenizer = init_tokenizer()
    except Exception as e:
        print(f"初始化tokenizer失败: {str(e)}")
        return
    
    # 获取converted目录下所有jsonl文件
    dataset_dir = Path("converted")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"目录不存在: {dataset_dir.absolute()}")
    
    # 获取所有jsonl文件，但排除merged_dataset.jsonl
    dataset_files = [f for f in dataset_dir.glob("*.jsonl") 
                    if f.name != "merged_dataset.jsonl"]
    
    if not dataset_files:
        raise FileNotFoundError(f"在 {dataset_dir.absolute()} 中未找到任何有效的数据集文件")
    
    print(f"工作目录: {os.getcwd()}")
    print(f"找到以下数据集文件:")
    for f in dataset_files:
        print(f"- {f.name}")
    
    # 收集所有数据集的长度信息
    all_lengths = []
    
    for file_path in dataset_files:
        dataset_name = file_path.stem
        try:
            data = load_dataset(file_path)
            print(f"\n处理数据集 {dataset_name}:")
            print(f"样本数量: {len(data)}")
            
            # 使用tokenizer计算token长度
            lengths = [calculate_prompt_length(item, tokenizer) for item in data]
            
            # 添加到总数据中，包含数据集名称
            all_lengths.extend([(length, dataset_name) for length in lengths])
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
            continue
    
    if not all_lengths:
        raise ValueError("没有成功处理任何数据集")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_lengths, columns=['length', 'dataset'])
    
    # 创建输出目录
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 绘制图表
    plot_boxplot(df, output_dir)
    print(f"箱线图已保存至: {output_dir}/prompt_length_boxplot.png")
    
    plot_histogram(df, output_dir)
    print(f"直方图已保存至: {output_dir}/prompt_length_histogram.png")
    
    # 保存统计信息
    save_stats_markdown(df, output_dir)
    print(f"统计信息已保存至: {output_dir}/prompt_length_stats.md")
    
    # 使用新的分离箱线图替换原来的箱线图
    plot_split_boxplot(df, output_dir)
    print(f"分离箱线图已保存至: {output_dir}/prompt_length_split_boxplot.png")
    
    # 在完成单个数据集分析后，添加对合并数据集的分析
    analyze_merged_dataset()

if __name__ == "__main__":
    try:
        analyze_prompt_lengths()
    except Exception as e:
        print(f"分析过程出错: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        print(traceback.format_exc()) 