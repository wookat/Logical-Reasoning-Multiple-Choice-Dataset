import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LabelBalancer:
    def __init__(self, input_dir: str, output_dir: str):
        """初始化标签平衡器
        
        Args:
            input_dir: 输入数据目录(mixed目录)
            output_dir: 输出数据目录
        """
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / input_dir
        self.output_dir = self.base_dir / output_dir
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir = self.output_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集文件名
        self.dataset_files = [
            "full_train.jsonl",
            "peft_train.jsonl",
            "eval.jsonl",
        ]
        
        # 验证文件是否存在
        self._verify_files()
    
    def _verify_files(self):
        """验证所有需要的数据文件是否存在"""
        missing_files = []
        for filename in self.dataset_files:
            if not (self.input_dir / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(
                f"以下数据文件不存在:\n" + 
                "\n".join(missing_files) + 
                f"\n请确保这些文件位于 {self.input_dir} 目录下"
            )
    
    def _load_dataset(self, filename: str) -> List[Dict]:
        """加载数据集
        
        Args:
            filename: 数据集文件名
            
        Returns:
            数据列表
        """
        data = []
        with open(self.input_dir / filename, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def _save_dataset(self, data: List[Dict], filename: str):
        """保存数据集
        
        Args:
            data: 数据列表
            filename: 输出文件名
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"已保存平衡后的数据集到: {output_path}")
    
    def _analyze_distribution(self, data: List[Dict]) -> Dict:
        """分析标签分布
        
        Args:
            data: 数据列表
            
        Returns:
            分布统计信息
        """
        # 按选项数量分组
        grouped_data = defaultdict(list)
        for item in data:
            opt_count = len(item['options'])
            grouped_data[opt_count].append(item)
        
        # 统计每组的标签分布
        distribution = {}
        for opt_count, items in grouped_data.items():
            label_counts = defaultdict(int)
            for item in items:
                label_counts[item['label']] += 1
            distribution[opt_count] = {
                'total': len(items),
                'label_counts': dict(label_counts)
            }
        
        return distribution
    
    def _balance_by_shuffling(self, data: List[Dict]) -> Tuple[List[Dict], Dict, Dict]:
        """通过调整选项顺序来平衡标签分布，保持正确答案内容不变
        
        Args:
            data: 原始数据列表
            
        Returns:
            平衡后的数据列表, 平衡前分布, 平衡后分布
        """
        # 记录原始分布
        original_distribution = self._analyze_distribution(data)
        
        # 按选项数量分组
        grouped_data = defaultdict(list)
        balanced_data = []  # 最终的平衡数据
        
        # 第一步：按选项数量分组
        for item in data:
            opt_count = len(item['options'])
            grouped_data[opt_count].append(item)
        
        # 对每组分别平衡
        for opt_count, items in grouped_data.items():
            # 计算理想的每个标签样本数
            total_samples = len(items)
            target_per_label = total_samples // opt_count
            remaining = total_samples % opt_count
            
            # 初始化标签计数
            current_counts = defaultdict(int)
            group_balanced_data = []  # 当前组的平衡数据
            
            # 处理每个样本
            for item in items:
                # 计算当前最少使用的标签
                min_label = min(range(opt_count), 
                              key=lambda x: current_counts[x])
                
                # 检查是否需要使用这个标签
                if current_counts[min_label] < target_per_label + (1 if remaining > min_label else 0):
                    target_label = min_label
                else:
                    # 找到下一个未满的标签
                    target_label = None
                    for label in range(opt_count):
                        if current_counts[label] < target_per_label + (1 if remaining > label else 0):
                            target_label = label
                            break
                    if target_label is None:
                        continue  # 所有标签都已满
                
                # 创建新的样本
                balanced_item = item.copy()
                current_label = item['label']
                correct_answer = item['options'][current_label]
                
                # 创建新的选项列表
                new_options = item['options'].copy()
                
                # 只有当目标标签和当前标签不同时才交换
                if current_label != target_label:
                    # 交换选项位置
                    new_options[current_label] = new_options[target_label]
                    new_options[target_label] = correct_answer
                
                # 更新样本
                balanced_item['options'] = new_options
                balanced_item['label'] = target_label
                
                # 验证正确答案是否保持不变
                if balanced_item['options'][balanced_item['label']] != correct_answer:
                    logging.error(f"答案内容不匹配: 原始答案={correct_answer}, "
                                f"平衡后答案={balanced_item['options'][balanced_item['label']]}")
                    continue  # 跳过这个样本而不是抛出错误
                
                group_balanced_data.append(balanced_item)
                current_counts[target_label] += 1
            
            # 将当前组的平衡数据添加到最终结果
            balanced_data.extend(group_balanced_data)
        
        # 记录平衡后的分布
        balanced_distribution = self._analyze_distribution(balanced_data)
        
        # 记录处理结果
        logging.info(f"原始样本数: {len(data)}")
        logging.info(f"平衡后样本数: {len(balanced_data)}")
        
        return balanced_data, original_distribution, balanced_distribution
    
    def _balance_labels(self, data: List[Dict]) -> Tuple[List[Dict], Dict, Dict]:
        """平衡标签分布(使用选项重排序方法)
        
        Args:
            data: 原始数据列表
            
        Returns:
            平衡后的数据列表, 平衡前分布, 平衡后分布
        """
        return self._balance_by_shuffling(data)
    
    def _plot_distributions(self, 
                          original_dist: Dict, 
                          balanced_dist: Dict, 
                          filename: str):
        """绘制分布对比图
        
        Args:
            original_dist: 原始分布
            balanced_dist: 平衡后的分布
            filename: 输出文件名
        """
        # 为每个选项数量创建一个子图
        opt_counts = sorted(original_dist.keys())
        n_plots = len(opt_counts)
        
        # 处理只有一个选项数量的情况
        if n_plots == 1:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes = [axes]  # 将axes包装成二维数组形式
        else:
            fig, axes = plt.subplots(n_plots, 2, figsize=(15, 5*n_plots))
        
        fig.suptitle('Label Distribution Before and After Balancing')
        
        for idx, opt_count in enumerate(opt_counts):
            # 获取原始分布和平衡后分布的所有标签
            orig_data = original_dist[opt_count]
            bal_data = balanced_dist[opt_count]
            
            # 合并所有可能的标签
            all_labels = sorted(set(orig_data['label_counts'].keys()) | 
                              set(bal_data['label_counts'].keys()))
            
            # 原始分布
            counts = [orig_data['label_counts'].get(label, 0) for label in all_labels]
            if n_plots == 1:
                ax = axes[0]
            else:
                ax = axes[idx, 0]
            ax.bar(all_labels, counts)
            ax.set_title(f'Original Distribution ({opt_count} options)')
            ax.set_xlabel('Label')
            ax.set_ylabel('Count')
            
            # 平衡后的分布
            counts = [bal_data['label_counts'].get(label, 0) for label in all_labels]
            if n_plots == 1:
                ax = axes[1]
            else:
                ax = axes[idx, 1]
            ax.bar(all_labels, counts)
            ax.set_title(f'Balanced Distribution ({opt_count} options)')
            ax.set_xlabel('Label')
            ax.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / f'{filename}_distribution.png')
        plt.close()
    
    def _generate_report(self, 
                        filename: str, 
                        original_dist: Dict, 
                        balanced_dist: Dict):
        """生成分析报告
        
        Args:
            filename: 数据集文件名
            original_dist: 原始分布
            balanced_dist: 平衡后的分布
        """
        report = [f"# {filename} 标签分布分析报告\n"]
        
        # 添加总体统计
        orig_total = sum(data['total'] for data in original_dist.values())
        bal_total = sum(data['total'] for data in balanced_dist.values())
        report.extend([
            "## 总体统计",
            f"- 原始样本总数: {orig_total}",
            f"- 平衡后样本总数: {bal_total}",
            "- 样本数量保持不变,通过调整选项顺序实现标签平衡\n"
        ])
        
        # 按选项数量分组统计
        report.append("## 各选项数量组的分布统计")
        for opt_count in sorted(original_dist.keys()):
            orig_data = original_dist[opt_count]
            bal_data = balanced_dist[opt_count]
            
            report.extend([
                f"\n### {opt_count}个选项的样本",
                f"- 原始样本数: {orig_data['total']}",
                f"- 平衡后样本数: {bal_data['total']}",
                "\n原始标签分布:",
                "| 标签 | 样本数 | 占比 |",
                "| --- | --- | --- |"
            ])
            
            # 原始分布
            for label, count in sorted(orig_data['label_counts'].items()):
                percentage = (count / orig_data['total']) * 100
                report.append(f"| {label} | {count} | {percentage:.2f}% |")
            
            report.extend([
                "\n平衡后标签分布:",
                "| 标签 | 样本数 | 占比 |",
                "| --- | --- | --- |"
            ])
            
            # 平衡后分布
            for label, count in sorted(bal_data['label_counts'].items()):
                percentage = (count / bal_data['total']) * 100
                report.append(f"| {label} | {count} | {percentage:.2f}% |")
        
        # 保存报告
        report_path = self.analysis_dir / f'{filename}_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logging.info(f"分析报告已保存至: {report_path}")
    
    def balance_datasets(self):
        """处理所有数据集"""
        for filename in self.dataset_files:
            logging.info(f"处理数据集: {filename}")
            
            # 加载数据
            data = self._load_dataset(filename)
            
            # 平衡标签
            balanced_data, original_dist, balanced_dist = self._balance_labels(data)
            
            # 保存平衡后的数据集
            output_filename = f"balanced_{filename}"
            self._save_dataset(balanced_data, output_filename)
            
            # 生成分布对比图
            self._plot_distributions(
                original_dist, 
                balanced_dist, 
                output_filename.replace('.jsonl', '')
            )
            
            # 生成分析报告
            self._generate_report(
                output_filename.replace('.jsonl', ''),
                original_dist,
                balanced_dist
            )
            
            logging.info(f"完成数据集 {filename} 的处理")

def main():
    # 设置随机种子
    random.seed(42)
    
    try:
        # 创建标签平衡器
        balancer = LabelBalancer(
            input_dir="mixed",
            output_dir="balanced"
        )
        
        # 处理所有数据集
        balancer.balance_datasets()
        
        logging.info("所有数据集处理完成!")
        
    except Exception as e:
        logging.error(f"处理数据集时出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 