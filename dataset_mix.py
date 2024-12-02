import os
import json
import random
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LogicDatasetCreator:
    def __init__(self, input_dir: str, output_dir: str):
        """初始化数据集创建器
        
        Args:
            input_dir: 输入数据目录(converted目录)
            output_dir: 输出数据目录
        """
        # 设置基本路径
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / input_dir
        self.output_dir = self.base_dir / output_dir
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
            
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.output_dir  
        self.analysis_dir = self.output_dir / "analysis"  
        self.logs_dir = self.output_dir / "logs"  
        
        for dir_path in [self.dataset_dir, self.analysis_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 设置总目标样本数
        self.total_target_samples = 30000  # 目标总样本数
        
        # 定义数据集文件名映射
        self.dataset_files = {
            "logiqa": "logiqa.jsonl",
            "logiqa_val": "logiqa_val.jsonl",
            "reclor_train": "reclor_train.jsonl",
            "reclor_val": "reclor_val.jsonl",
            "balacopa": "balacopa.jsonl",
            "arc_challenge": "arc_challenge.jsonl",
            "arc_easy": "arc_easy.jsonl",
            "race": "race.jsonl",
            "winogrande": "winogrande.jsonl",
            "boolean_train": "boolean_train.jsonl",
            "boolean_dev": "boolean_dev.jsonl"
        }
        
        # 数据集分组配置
        self.dataset_groups = {
            "balacopa": ["balacopa"],                # BalaCOPA
            "reclor": ["reclor_train", "reclor_val"], # ReClor系列
            "logiqa": ["logiqa", "logiqa_val"],      # LogiQA系列
            "arc": ["arc_challenge", "arc_easy"],     # ARC系列
            "race": ["race"],                        # RACE
            "winogrande": ["winogrande"],            # Winogrande
            "boolean": ["boolean_train", "boolean_dev"] # Boolean系列
        }
        
        # 每个组在总样本中的目标比例
        self.group_ratios = {
            "balacopa": 0.10,  # BalaCOPA占10%
            "reclor": 0.18,    # ReClor系列总共占18%
            "logiqa": 0.27,    # LogiQA系列总共占27%
            "arc": 0.15,       # ARC系列总共占15%
            "race": 0.15,      # RACE占15%
            "winogrande": 0.05,# Winogrande占5%
            "boolean": 0.05    # Boolean系列总共占5%
        }
        
        # 最终数据集的划分比例
        self.split_ratios = {
            "full_train": 0.9,  # 90%用于全量训练
            "peft_train": 0.45, # 45%用于PEFT训练（是全量训练集的一半）
            "eval": 0.1        # 10%用于评估
        }
        
        # 分层采样的特征配置
        self.stratify_config = {
            "length_bins": 5,  # 文本长度分箱数
            "length_labels": ['very_short', 'short', 'medium', 'long', 'very_long']
        }
        
        # 随机种子设置
        self.random_seed = 42
        random.seed(self.random_seed)
        
        # 验证文件
        self._verify_files()
        
        # 添加无效样本记录目录
        self.invalid_samples_dir = self.output_dir / "invalid_samples"
        self.invalid_samples_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """设置日志"""
        # 获取根日志记录器
        logger = logging.getLogger()
        
        # 清除现有的处理器
        logger.handlers.clear()
        
        # 创建新的处理器
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"dataset_creation_{current_time}.log"
        
        handlers = [
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def _get_sample_fingerprint(self, item: Dict) -> tuple:
        """生成样本的唯一指纹，并记录指纹信息"""
        text = item['text'].strip().lower()
        question = item['question'].strip().lower()
        options = tuple(opt.strip().lower() for opt in item['options'])
        
        # 计算文本的相似度指纹
        fingerprint = (text, question, options)
        
        # 如果是第一次遇到这个指纹，记录详细信息
        if not hasattr(self, '_fingerprint_details'):
            self._fingerprint_details = {}
        
        if fingerprint not in self._fingerprint_details:
            self._fingerprint_details[fingerprint] = []
        
        self._fingerprint_details[fingerprint].append({
            'source': item['source'],
            'text_length': len(text),
            'question_length': len(question),
            'options_count': len(options),
            'label': item['label']
        })
        
        return fingerprint

    def _validate_data_item(self, item: Dict, dataset_name: str) -> Tuple[bool, str]:
        """数据项验证，增加内容重复和有效性检查
        
        Returns:
            Tuple[bool, str]: (是否有效, 无效原因)
        """
        # 基础字段验证
        required_fields = ['text', 'question', 'options', 'label', 'source']
        for field in required_fields:
            if field not in item:
                if field == 'source':
                    item['source'] = dataset_name
                else:
                    return False, f"缺少必需字段: {field}"
        
        # text和question的内容验证
        text = item['text'].strip() if isinstance(item['text'], str) else ''
        question = item['question'].strip() if isinstance(item['question'], str) else ''
        
        # 检查text和question至少有一个有实际内容
        if not text and not question:
            return False, "text和question都为空"
        
        # 选项验证
        if not isinstance(item['options'], list) or not item['options']:
            return False, "选项不是列表或为空"
        
        # 选项内容验证 - 确保选项非空
        if any(not isinstance(opt, str) or not opt.strip() for opt in item['options']):
            return False, "存在空选项或选项格式不正确"
        
        # 选项重复检查
        options = [opt.strip().lower() for opt in item['options']]
        unique_options = set(options)
        if len(unique_options) < len(options):
            return False, "存在重复选项"
        
        # 标签验证
        if not isinstance(item['label'], int):
            return False, "标签不是整数类型"
        if item['label'] < 0 or item['label'] >= len(item['options']):
            return False, f"标签值 {item['label']} 超出选项范围 [0, {len(item['options'])-1}]"
        
        return True, ""

    def _stratify_sample(self, data: List[Dict], target_size: int) -> List[Dict]:
        """分层采样
        
        Args:
            data: 数据列表
            target_size: 目标样本数
            
        Returns:
            采样后的数据列表
        """
        # 创建分层特征
        df = pd.DataFrame(data)
        df['opt_count'] = df['options'].apply(len)
        df['text_length'] = df['text'].apply(len)
        df['length_bin'] = pd.qcut(df['text_length'], q=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'])
        
        # 组合分层特
        df['strata'] = df.apply(lambda x: f"{x['source']}_{x['opt_count']}_{x['length_bin']}", axis=1)
        
        # 计算每个层的目标样本数
        strata_counts = df['strata'].value_counts()
        total_samples = len(df)
        strata_ratios = {stratum: count/total_samples for stratum, count in strata_counts.items()}
        
        # 分层采样
        sampled_data = []
        for stratum, ratio in strata_ratios.items():
            stratum_target = int(target_size * ratio)
            stratum_data = df[df['strata'] == stratum]
            if len(stratum_data) > stratum_target:
                sampled_stratum = stratum_data.sample(n=stratum_target, random_state=42)
            else:
                sampled_stratum = stratum_data
            sampled_data.extend(sampled_stratum.to_dict('records'))
        
        return sampled_data

    def _calculate_total_length(self, item: Dict) -> int:
        """计算样本的总长度
        
        Args:
            item: 数据项
            
        Returns:
            总长度
        """
        text_len = len(item['text'])
        question_len = len(item['question'])
        options_len = sum(len(opt) for opt in item['options'])
        return text_len + question_len + options_len

    def _filter_and_replace_outliers(self, data: List[Dict], group_available: Dict[str, List[Dict]], 
                                   group_name: str) -> List[Dict]:
        """过滤异常长度样本并尝试替换
        
        Args:
            data: 当前组的数据
            group_available: 所有可用的原始数据
            group_name: 组名
            
        Returns:
            处理后的数据列表
        """
        # 计算所有样本的长度
        lengths = [self._calculate_total_length(item) for item in data]
        
        # 计算箱线图参数
        q1 = np.percentile(lengths, 25)
        q3 = np.percentile(lengths, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        # 记录异常样本信息
        outliers_info = {
            'group_name': group_name,
            'total_samples': len(data),
            'length_stats': {
                'q1': q1,
                'median': np.median(lengths),
                'q3': q3,
                'iqr': iqr,
                'upper_bound': upper_bound
            },
            'outliers': []
        }
        
        # 分离正常样本和异常样本
        normal_samples = []
        available_replacements = []
        
        # 从原始数据中找出可用的替换样本
        for item in group_available[group_name]:
            length = self._calculate_total_length(item)
            if length <= upper_bound:
                available_replacements.append(item)
        
        # 处理每个样本
        for item, length in zip(data, lengths):
            if length <= upper_bound:
                normal_samples.append(item)
            else:
                outliers_info['outliers'].append({
                    'length': length,
                    'text_length': len(item['text']),
                    'question_length': len(item['question']),
                    'options_length': sum(len(opt) for opt in item['options']),
                    'source': item['source']
                })
        
        # 尝试补充样本
        samples_needed = len(data) - len(normal_samples)
        if samples_needed > 0 and available_replacements:
            # 从可用的替换样本中随机选择
            replacements = random.sample(
                available_replacements, 
                min(samples_needed, len(available_replacements))
            )
            normal_samples.extend(replacements)
            
            outliers_info['replacements_added'] = len(replacements)
        else:
            outliers_info['replacements_added'] = 0
        
        # 保存异常样本信息
        outliers_file = self.analysis_dir / f'{group_name}_length_outliers.json'
        with open(outliers_file, 'w', encoding='utf-8') as f:
            json.dump(outliers_info, f, ensure_ascii=False, indent=2)
        
        logging.info(f"{group_name}组长度异常样本处理:")
        logging.info(f"  - 原始样本数: {len(data)}")
        logging.info(f"  - 异常样本数: {len(outliers_info['outliers'])}")
        logging.info(f"  - 补充样本数: {outliers_info['replacements_added']}")
        logging.info(f"  - 最终样本数: {len(normal_samples)}")
        
        return normal_samples

    def create_datasets(self):
        """创建所有混合数据集"""
        logging.info("开始创建混合数据集...")
        
        # 1. 按组加载并混合数据集
        all_data = []
        used_samples = set()
        
        # 首先收集每个组的可用样本数
        group_available = {}
        for group_name, datasets in self.dataset_groups.items():
            group_data = []
            duplicate_count = 0
            logging.info(f"处理{group_name}组数据...")
            
            # 加载该组所有数据集
            for dataset_name in datasets:
                data = self.load_dataset(dataset_name)
                for item in data:
                    fingerprint = self._get_sample_fingerprint(item)
                    if fingerprint not in used_samples:
                        used_samples.add(fingerprint)
                        group_data.append(item)
                    else:
                        duplicate_count += 1
            
            group_available[group_name] = group_data
            logging.info(f"{group_name}组统计:")
            logging.info(f"  - 原始样本数: {sum(len(self.load_dataset(d)) for d in datasets)}")
            logging.info(f"  - 重复样本数: {duplicate_count}")
            logging.info(f"  - 去重后可用样本数: {len(group_data)}")
            
            # 保存去重信息
            duplicate_info = {
                'group_name': group_name,
                'datasets': datasets,
                'original_count': sum(len(self.load_dataset(d)) for d in datasets),
                'duplicate_count': duplicate_count,
                'available_count': len(group_data)
            }
            
            duplicate_file = self.invalid_samples_dir / f"{group_name}_duplicates.json"
            with open(duplicate_file, 'w', encoding='utf-8') as f:
                json.dump(duplicate_info, f, ensure_ascii=False, indent=2)
        
        # 2. 按比例采样每个组的数据
        for group_name, ratio in self.group_ratios.items():
            target_samples = int(self.total_target_samples * ratio)
            available_samples = group_available[group_name]
            
            if len(available_samples) < target_samples:
                logging.warning(
                    f"{group_name}组可用样本数({len(available_samples)})少于目标样本数({target_samples})"
                )
                sampled_data = available_samples
            else:
                sampled_data = random.sample(available_samples, target_samples)
            
            all_data.extend(sampled_data)
            logging.info(f"{group_name}组采样后样本数: {len(sampled_data)}")
        
        # 3. 处理异常长度样本
        logging.info("开始处理异常长度样本...")
        lengths = [self._calculate_total_length(item) for item in all_data]
        
        # 计算箱线图参数
        q1 = np.percentile(lengths, 25)
        q3 = np.percentile(lengths, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        # 记录异常样本信息
        outliers_info = {
            'total_samples': len(all_data),
            'length_stats': {
                'q1': q1,
                'median': np.median(lengths),
                'q3': q3,
                'iqr': iqr,
                'upper_bound': upper_bound
            },
            'outliers': []
        }
        
        # 分离正常样本和异常样本
        normal_data = []
        outlier_data = []
        
        for item, length in zip(all_data, lengths):
            if length <= upper_bound:
                normal_data.append(item)
            else:
                outlier_data.append({
                    'item': item,
                    'length': length,
                    'text_length': len(item['text']),
                    'question_length': len(item['question']),
                    'options_length': sum(len(opt) for opt in item['options']),
                    'source': item['source']
                })
                outliers_info['outliers'].append(outlier_data[-1])
        
        # 保存异常样本信息
        outliers_file = self.analysis_dir / 'length_outliers.json'
        with open(outliers_file, 'w', encoding='utf-8') as f:
            json.dump(outliers_info, f, ensure_ascii=False, indent=2)
        
        logging.info(f"异常长度样本处理结果:")
        logging.info(f"  - 原始样本数: {len(all_data)}")
        logging.info(f"  - 异常样本数: {len(outlier_data)}")
        logging.info(f"  - 保留样本数: {len(normal_data)}")
        
        # 4. 打乱数据顺序
        random.shuffle(normal_data)
        
        # 5. 划分数据集
        total_samples = len(normal_data)
        eval_size = int(total_samples * self.split_ratios['eval'])
        train_data = normal_data[:-eval_size]
        eval_data = normal_data[-eval_size:]
        
        # 6. 创建训练集
        full_train_size = int(total_samples * self.split_ratios['full_train'])
        peft_train_size = int(total_samples * self.split_ratios['peft_train'])
        
        full_train_data = train_data[:full_train_size]
        peft_train_data = train_data[:peft_train_size]
        
        # 7. 保存数据集
        datasets = {
            'full_train.jsonl': full_train_data,
            'peft_train.jsonl': peft_train_data,
            'eval.jsonl': eval_data
        }
        
        for filename, data in datasets.items():
            self._save_dataset(data, filename)
            self._analyze_and_plot(data, filename.replace('.jsonl', ''))
            self._analyze_shuffle_quality(filename.replace('.jsonl', ''), data)
            logging.info(f"保存数据集 {filename}: {len(data)} 样本")
        
        logging.info("所有数据集创建完成!")

    def _analyze_and_plot(self, data: List[Dict], name: str):
        """分析数据集分布并生成可视化
        
        Args:
            data: 数据列表
            name: 数据集名称
        """
        # 1. 基础统计
        stats = {
            'total_samples': len(data),
            'source_dist': {},
            'opt_count_dist': {},
            'length_dist': {},
            'label_dist': {}
        }
        
        # 2. 计算各种分布
        text_lengths = []  # 收集所有文本长度
        for item in data:
            # 来源分布
            stats['source_dist'][item['source']] = stats['source_dist'].get(item['source'], 0) + 1
            # 选项数量分布
            opt_count = len(item['options'])
            stats['opt_count_dist'][opt_count] = stats['opt_count_dist'].get(opt_count, 0) + 1
            # 收集文本长度
            text_len = len(item['text'])
            text_lengths.append(text_len)
            # 标签分布
            stats['label_dist'][item['label']] = stats['label_dist'].get(item['label'], 0) + 1
        
        # 处理文本长度分布
        try:
            # 尝试使用qcut进行分箱
            length_bins = pd.qcut(text_lengths, q=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'], duplicates='drop')
            for length_bin in length_bins:
                stats['length_dist'][str(length_bin)] = stats['length_dist'].get(str(length_bin), 0) + 1
        except ValueError:
            # 如果分箱失败，使用简单的统计
            stats['length_dist'] = {
                'min': min(text_lengths),
                'max': max(text_lengths),
                'mean': sum(text_lengths) / len(text_lengths),
                'median': sorted(text_lengths)[len(text_lengths)//2]
            }
        
        # 3. 绘制分布图
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'Distribution Analysis - {name}')
        
        # 来源分布
        pd.Series(stats['source_dist']).plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Source Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 选项数量分布
        pd.Series(stats['opt_count_dist']).plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Options Count Distribution')
        
        # 长度分布
        if isinstance(next(iter(stats['length_dist'].values())), (int, float)):
            # 如果是数值统计，绘制直方图
            axes[1,0].hist(text_lengths, bins=30)
            axes[1,0].set_title('Text Length Distribution')
        else:
            # 如果是分箱统计，绘制条形图
            pd.Series(stats['length_dist']).plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Text Length Distribution')
        
        # 标签分布
        pd.Series(stats['label_dist']).plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Label Distribution')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / f'{name}_distribution.png')
        plt.close()
        
        # 4. 保存统计信息
        with open(self.analysis_dir / f'{name}_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def _verify_files(self):
        """验证所有需要的数据文件是否存在"""
        missing_files = []
        existing_files = []
        
        for dataset_name, filename in self.dataset_files.items():
            file_path = self.input_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
            else:
                existing_files.append(filename)
        
        # 打印现有文件和目录信息以便调试
        logging.info(f"当前目录: {os.getcwd()}")
        logging.info(f"输入目录: {self.input_dir}")
        logging.info("已找到的文件:")
        for file in existing_files:
            logging.info(f"  - {file}")
        
        if missing_files:
            raise FileNotFoundError(
                f"以下数据文件不存在:\n" + 
                "\n".join(missing_files) + 
                f"\n请确保这些文件位于 {self.input_dir} 目录下"
            )

    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """加载数据集文件"""
        filename = self.dataset_files[dataset_name]
        file_path = self.input_dir / filename
        
        try:
            data = []
            invalid_samples = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        is_valid, invalid_reason = self._validate_data_item(item, dataset_name)
                        
                        if is_valid:
                            data.append(item)
                        else:
                            invalid_samples.append({
                                'line_number': line_num,
                                'item': item,
                                'reason': invalid_reason
                            })
                    except json.JSONDecodeError:
                        invalid_samples.append({
                            'line_number': line_num,
                            'item': line.strip(),
                            'reason': "JSON解析错误"
                        })
            
            # 保存无效样本信息
            if invalid_samples:
                invalid_file = self.invalid_samples_dir / f"{dataset_name}_invalid.json"
                with open(invalid_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'dataset': dataset_name,
                        'total_samples': line_num,
                        'valid_samples': len(data),
                        'invalid_samples': len(invalid_samples),
                        'invalid_records': invalid_samples
                    }, f, ensure_ascii=False, indent=2)
            
            logging.info(f"从 {filename} 加载了 {line_num} 条数据，其中有效数据 {len(data)} 条，"
                        f"无效数据 {len(invalid_samples)} 条")
            if invalid_samples:
                logging.info(f"无效样本详情已保存至: {invalid_file}")
            
            return data
            
        except Exception as e:
            logging.error(f"加载数据集 {filename} 时出错: {str(e)}")
            raise

    def _save_dataset(self, data: List[Dict], filename: str):
        """保存数据集，并重新排列索引，清理不需要的字段，保留source字段
        
        Args:
            data: 数据列表
            filename: 输出文件名
        """
        output_path = self.dataset_dir / filename
        try:
            # 定义需要保留的字段
            required_fields = ['text', 'question', 'options', 'label', 'source', 'index']
            
            cleaned_data = []
            for idx, item in enumerate(data):
                # 创建新的数据项，只保留必要字段
                cleaned_item = {
                    'index': idx,
                    'text': item['text'],
                    'question': item['question'],
                    'options': item['options'],
                    'label': item['label'],
                    'source': item['source']
                }
                cleaned_data.append(cleaned_item)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in cleaned_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
            logging.info(f"成功保存数据集到: {output_path}共 {len(cleaned_data)} 条数据，索引范围 [0-{len(cleaned_data)-1}]")
        except Exception as e:
            logging.error(f"保存数据集时出错: {str(e)}")
            raise

    def _save_fingerprint_analysis(self):
        """保存指纹分析结果"""
        if hasattr(self, '_fingerprint_details'):
            analysis = {
                'total_unique_fingerprints': len(self._fingerprint_details),
                'fingerprint_details': {
                    str(k): v for k, v in self._fingerprint_details.items() 
                    if len(v) > 1  # 只保存重复的指纹信息
                }
            }
            
            fingerprint_file = self.analysis_dir / 'fingerprint_analysis.json'
            with open(fingerprint_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

    def _analyze_shuffle_quality(self, name: str, dataset: List[Dict]):
        """分析数据集打乱质量并生成报告
        
        Args:
            name: 数据集名称
            dataset: 数据列表
        """
        # 1. 获取所有可能的数据源
        all_sources = sorted(set(item['source'] for item in dataset))
        
        # 2. 按chunk分析source分布
        sources = [item['source'] for item in dataset]
        chunk_size = 100
        chunks = [sources[i:i+chunk_size] for i in range(0, len(sources), chunk_size)]
        
        # 3. 创建分布图
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle(f'Shuffle Quality Analysis - {name}')
        
        # 3.1 绘制前10个chunk的分布热力图
        chunk_dists = []
        for chunk in chunks[:10]:  # 只取前10个chunk
            # 计算每个source的出现次数
            dist = {source: chunk.count(source) for source in all_sources}
            chunk_dists.append(dist)
        
        # 转换为DataFrame，确保所有source都存在
        dist_df = pd.DataFrame(chunk_dists, columns=all_sources).fillna(0)
        
        # 绘制热力图
        sns.heatmap(dist_df, ax=axes[0], cmap='YlOrRd', annot=True, fmt='.0f')
        axes[0].set_title('Source Distribution in First 10 Chunks')
        axes[0].set_xlabel('Source')
        axes[0].set_ylabel('Chunk Index')
        
        # 3.2 绘制整体分布趋势
        # 使用滑动窗口计算趋势
        window_size = 5
        trends = []
        source_trends = {source: [] for source in all_sources}
        
        # 计算每个窗口内各source的平均出现次数
        for i in range(0, len(chunks), window_size):
            window_chunk = sources[i:i+window_size*chunk_size]
            for source in all_sources:
                avg_count = window_chunk.count(source) / window_size
                source_trends[source].append(avg_count)
        
        # 绘制趋势线
        x = range(len(next(iter(source_trends.values()))))
        for source in all_sources:
            axes[1].plot(x, source_trends[source], label=source, marker='o')
        
        axes[1].set_title('Source Distribution Trend (Moving Average)')
        axes[1].set_xlabel('Window Index (5 chunks per window)')
        axes[1].set_ylabel('Average Count per Chunk')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True)
        
        # 调整布局以适应图例
        plt.tight_layout()
        plt.savefig(self.analysis_dir / f'{name}_shuffle_quality.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 4. 生成分析报告
        report = [
            f"# {name} 数据集打乱质量分析报告\n",
            "## 1. 数据基本信息",
            f"- 总样本数: {len(dataset)}",
            f"- Chunk大小: {chunk_size}",
            f"- 总Chunk数: {len(chunks)}",
            f"- 数据源数量: {len(all_sources)}\n",
            "## 2. 各Chunk的分布统计"
        ]
        
        # 添加前5个chunk的详细统计
        for i, chunk in enumerate(chunks[:5]):
            chunk_dist = {source: chunk.count(source) for source in all_sources}
            total = len(chunk)
            
            report.extend([
                f"\n### Chunk {i+1}",
                "| 数据来源 | 样本数 | 占比 |",
                "| --- | --- | --- |"
            ])
            
            for source in all_sources:
                count = chunk_dist[source]
                percentage = (count / total) * 100
                report.append(f"| {source} | {count} | {percentage:.2f}% |")
        
        # 添加分布均匀性分析
        source_counts = pd.Series(sources).value_counts()
        source_std = source_counts.std()
        source_cv = source_std / source_counts.mean()  # 变异系数
        
        report.extend([
            "\n## 3. 分布均匀性分析",
            f"- 标准差: {source_std:.2f}",
            f"- 变异系数: {source_cv:.2f}",
            "- 评估: " + ("良好" if source_cv < 0.5 else "需要改进"),
            "\n## 4. 数据源分布",
            "| 数据来源 | 总样本数 | 总体占比 |",
            "| --- | --- | --- |"
        ])
        
        total_samples = len(sources)
        for source in all_sources:
            count = sources.count(source)
            percentage = (count / total_samples) * 100
            report.append(f"| {source} | {count} | {percentage:.2f}% |")
        
        # 保存报告
        report_path = self.analysis_dir / f'{name}_shuffle_analysis.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

def main():
    random.seed(42)
    try:
        creator = LogicDatasetCreator(
            input_dir="converted",
            output_dir="mixed"
        )
        creator.create_datasets()
        
    except Exception as e:
        logging.error(f"创建数据集时出错: {str(e)}")
        raise 

if __name__ == "__main__":
    main()