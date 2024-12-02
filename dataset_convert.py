import json
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import os
import sys
import warnings
from collections import Counter
import locale
import re

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 切换到脚本所在目录
os.chdir(SCRIPT_DIR)

# 设置默认编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def check_environment():
    """检查运行环境和必要的依赖"""
    print(f"工作目录: {os.getcwd()}")
    print(f"Python 版本: {sys.version}")
    print(f"Pandas 版本: {pd.__version__}")
    
    # 检查必要的依赖
    missing_deps = []
    try:
        import pyarrow
        print(f"PyArrow 版本: {pyarrow.__version__}")
    except ImportError:
        missing_deps.append("pyarrow")
        
    try:
        import fastparquet
        print(f"Fastparquet 版本: {fastparquet.__version__}")
    except ImportError:
        missing_deps.append("fastparquet")
    
    if missing_deps:
        print("\n缺少必要的依赖包，请安装：")
        print("pip install " + " ".join(missing_deps))
        return False
        
    return True

def check_file_exists(path: str) -> bool:
    """检查文件是否存在"""
    file_path = Path(path)
    exists = file_path.exists()
    if not exists:
        print(f"警告: 文件不存在 - {path}")
    return exists

def is_valid_option(option: str) -> bool:
    """检查选项是否有效
    允许:
    1. 字母数字
    2. 特定的序号字符(①②③④⑤等)
    3. 包含上述字符的选项
    """
    # 去除空白字符
    option = option.strip()
    
    # 如果选项为空，返回False
    if not option:
        return False
    
    # 定义有效的特殊序号字符
    valid_special_chars = {
        '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩',
        '⑪', '⑫', '⑬', '⑭', '⑮', '⑯', '⑰', '⑱', '⑲', '⑳',
        '㉑', '㉒', '㉓', '㉔', '㉕', '㉖', '㉗', '㉘', '㉙', '㉚'
    }
    
    # 检查是否包含至少一个字母、数字或有效的特殊序号字符
    has_valid_char = False
    for char in option:
        if char.isalnum() or char in valid_special_chars:
            has_valid_char = True
            break
            
    return has_valid_char

# 基础转换器
class BaseConverter(ABC):
    def __init__(self, input_path: str):
        self.input_path = input_path
        
    @abstractmethod
    def convert(self) -> List[Dict[str, Any]]:
        pass
        
    def save(self, data: List[Dict], output_dir: str, name: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式
        try:
            def convert_numpy(obj):
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj
                
            json_path = Path(output_dir) / f"{name}.jsonl"
            with open(json_path, 'w', encoding='utf-8') as f:
                for item in data:
                    converted_item = convert_numpy(item)
                    f.write(json.dumps(converted_item, ensure_ascii=False) + '\n')
            print(f"已保存JSON格式: {json_path}")
        except Exception as e:
            print(f"保存JSON格式失败: {str(e)}")
            
        # 只在有parquet支持的情况下尝试保存parquet格式
        try:
            import pyarrow  # 检查是否有pyarrow支持
            df = pd.DataFrame(data)
            parquet_path = Path(output_dir) / f"{name}.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"已保存Parquet格式: {parquet_path}")
        except ImportError:
            print("跳过保存Parquet格式：缺少pyarrow支持")
        except Exception as e:
            print(f"保存Parquet格式失败: {str(e)}")

    def validate_data(self, data: List[Dict]) -> bool:
        """验证转换后的数据格式"""
        required_keys = {"text", "question", "options", "label"}
        
        for idx, item in enumerate(data):
            # 检查必需字段
            if not all(key in item for key in required_keys):
                print(f"数据缺少必需字段: {required_keys - set(item.keys())}")
                print(f"问题样本索引: {idx}")
                return False
                
            # 验证options和label的对应关系
            if not isinstance(item["options"], list):
                print(f"options必须是列表类型，当前类型: {type(item['options'])}")
                print(f"问题样本索引: {idx}")
                return False
                
            if not isinstance(item["label"], int):
                print(f"label必须是整数类型，当前类型: {type(item['label'])}")
                print(f"问题样本索引: {idx}")
                return False
                
            # 验证label范围
            if item["label"] < 0:
                print(f"label值 {item['label']} 小于0")
                print(f"问题样本索引: {idx}")
                return False
                
            if item["label"] >= len(item["options"]):
                print(f"label值 {item['label']} 超出options范围 [0, {len(item['options'])-1}]")
                print(f"问题样本索引: {idx}")
                print(f"当前options: {item['options']}")
                return False
            
            # 验证文本字段不为空
            if not item["text"] or not isinstance(item["text"], str):
                print(f"text字段必须是非空字符串")
                print(f"问题样本索引: {idx}")
                return False
                
            if not item["question"] or not isinstance(item["question"], str):
                print(f"question字段必须是非空字符串")
                print(f"问题样本索引: {idx}")
                return False
                
            # 验证options不为空且每个选项都是字符串
            if not item["options"]:
                print(f"options不能为空列表")
                print(f"问题样本索引: {idx}")
                return False
                
            if not all(isinstance(opt, str) and opt.strip() for opt in item["options"]):
                print(f"所有options必须是非空字符串")
                print(f"问题样本索引: {idx}")
                print(f"当前options: {item['options']}")
                return False
                
        return True

    def filter_invalid_samples(self, data: List[Dict]) -> List[Dict]:
        """过滤掉不合格的数据样本，并将异常样本导出到json文件"""
        valid_data = []
        invalid_stats = {
            "重复选项": [],
            "选项数不足": [],
            "text和question相同": [],
            "text和question为空": [],
            "label超出范围": [],
            "重复样本": [],
            "无效选项": [],
            "其他错误": [],
            "选项去重保留": []
        }
        
        # 用于检测重复样本
        seen_samples = set()
        
        for idx, item in enumerate(data):
            try:
                # 添加样本索引
                item["index"] = idx
                
                # 基础字段检查
                if not all(key in item for key in {"text", "question", "options", "label"}):
                    invalid_stats["其他错误"].append(item)
                    continue
                
                # 验证options基本类型
                if not isinstance(item["options"], list) or not item["options"]:
                    invalid_stats["其他错误"].append(item)
                    continue
                    
                if not all(isinstance(opt, str) and opt.strip() for opt in item["options"]):
                    invalid_stats["其他错误"].append(item)
                    continue
                
                # 检查选项是否有效（不能只包含特殊符号）
                options = [str(opt).strip() for opt in item["options"]]
                if not all(is_valid_option(opt) for opt in options):
                    invalid_stats["无效选项"].append(item)
                    continue
                
                # 处理选项 - 修改这部分代码
                options = [str(opt).strip() for opt in item["options"]]  # 确保所有选项都是字符串并去除空白
                
                # 检查选项是否有重复 (不区分大小写)
                lower_options = [opt.lower() for opt in options]
                option_counts = Counter(lower_options)
                
                if len(option_counts) != len(options):  # 如果有重复选项
                    # 获取唯一选项（保持原始大小写）
                    seen = set()
                    unique_options = []
                    for opt, lower_opt in zip(options, lower_options):
                        if lower_opt not in seen:
                            seen.add(lower_opt)
                            unique_options.append(opt)
                    
                    if len(unique_options) >= 2:
                        # 创建修改后的item
                        modified_item = item.copy()
                        modified_item["original_options"] = options.copy()
                        modified_item["original_label"] = item["label"]
                        
                        # 找到原始选中选项在去重后列表中的新位置
                        original_selected_option = options[item["label"]].lower()
                        new_label = lower_options.index(original_selected_option)
                        
                        # 更新item
                        modified_item["options"] = unique_options
                        modified_item["label"] = new_label
                        modified_item["处理说明"] = f"原有{len(options)}个选项，去重后保留{len(unique_options)}个选项"
                        
                        invalid_stats["选项去重保留"].append(modified_item)
                        item = modified_item
                    else:
                        invalid_stats["重复选项"].append(item)
                        continue
                
                # 检查选项数量
                if len(item["options"]) < 2:
                    invalid_stats["选项数不足"].append(item)
                    continue
                
                # 验证label
                if not isinstance(item["label"], int):
                    invalid_stats["label超出范围"].append(item)
                    continue
                    
                if item["label"] < 0 or item["label"] >= len(item["options"]):
                    invalid_stats["label超出范围"].append(item)
                    continue
                
                # 验证text和question
                text = str(item["text"]).strip()
                question = str(item["question"]).strip()
                
                # 检查text和question是否都为空
                if not text and not question:
                    invalid_stats["text和question为空"].append(item)
                    continue
                
                # 检查text和question是否相同
                if text.lower() == question.lower():
                    invalid_stats["text和question相同"].append(item)
                    continue
                
                # 检查重复样本
                sample_key = f"{text}|{question}|{','.join(item['options'])}"
                if sample_key in seen_samples:
                    invalid_stats["重复样本"].append(item)
                    continue
                seen_samples.add(sample_key)
                
                # 通过所有验证，添加到有效数据中
                valid_data.append(item)
                
            except Exception as e:
                print(f"处理样本 {idx} 时发生错误: {str(e)}")
                item["error_message"] = str(e)
                invalid_stats["其他错误"].append(item)
                continue
        
        # 导出异常和处理样本到json文件
        output_dir = Path("converted/invalid_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name = Path(self.input_path).stem if hasattr(self, "input_path") else "unknown"
        
        for reason, samples in invalid_stats.items():
            if samples:
                output_file = output_dir / f"{dataset_name}_{reason}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "dataset": dataset_name,
                        "reason": reason,
                        "count": len(samples),
                        "samples": samples
                    }, f, ensure_ascii=False, indent=2)
        
        # 打印详细的过滤统计信息
        total_invalid = sum(len(samples) for reason, samples in invalid_stats.items() 
                           if reason != "选项去重保留")
        if total_invalid > 0 or invalid_stats["选项去重保留"]:
            print(f"\n数据集 {dataset_name} 的处理统计:")
            for reason, samples in invalid_stats.items():
                if samples:
                    if reason == "选项去重保留":
                        print(f"- {reason}: {len(samples)} 个（这些样本经处理后被保留）")
                    else:
                        print(f"- {reason}: {len(samples)} 个")
            print(f"\n总计: 过滤 {total_invalid} 个无效样本")
            print(f"保留 {len(valid_data)} 个有效样本")
            print(f"其中 {len(invalid_stats['选项去重保留'])} 个样本经过选项去重处理后保留")
            print(f"处理记录已导出到 {output_dir} 目录")
        
        return valid_data

    def convert_and_filter(self) -> List[Dict]:
        """转换并过滤数据"""
        raw_data = self.convert()
        filtered_data = self.filter_invalid_samples(raw_data)
        return filtered_data

# RACE数据集转换器
class RaceConverter(BaseConverter):
    def convert(self) -> List[Dict]:
        df = pd.read_parquet(self.input_path)
        converted = []
        for _, row in df.iterrows():
            # 确保options是列表类型
            if isinstance(row["options"], str):
                try:
                    options = eval(row["options"])  # 安全地将字符串转换为列表
                except:
                    options = row["options"].strip('[]').split(',')
            else:
                options = row["options"]
                
            # 确保options是列表类型
            if not isinstance(options, list):
                options = list(options)
                
            converted.append({
                "text": row["article"],
                "question": row["question"],
                "options": options,
                "label": ord(str(row["answer"]).upper()) - ord('A')
            })
        return converted

# AI2 ARC数据集转换器 
class ArcConverter(BaseConverter):
    def convert(self) -> List[Dict]:
        df = pd.read_parquet(self.input_path)
        converted = []
        for _, row in df.iterrows():
            # 处理choices字段
            if isinstance(row["choices"], str):
                try:
                    choices = eval(row["choices"])
                except:
                    choices = {"text": []}  # 默认空列表
            else:
                choices = row["choices"]
                
            # 确保获取选项列表
            if isinstance(choices, dict) and "text" in choices:
                options = choices["text"]
            else:
                options = choices
                
            # 确保options是列表类型
            if not isinstance(options, list):
                if isinstance(options, str):
                    options = options.strip('[]').split(',')
                else:
                    options = list(options)
                
            converted.append({
                "text": "",  # 设置为空字符串
                "question": str(row["question"]),
                "options": options,
                "label": ord(str(row["answerKey"]).upper()) - ord('A')
            })
        return converted

# LogiQA数据集转换器
class LogiQAConverter(BaseConverter):
    def convert(self) -> List[Dict]:
        data = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "text": item["text"],
                    "question": item["question"],
                    "options": item["options"],
                    "label": item["answer"]
                })
        return data

# Reclor数据集转换器
class ReclorConverter(BaseConverter):
    def convert(self) -> List[Dict]:
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [{
            "text": item["context"],
            "question": item["question"],
            "options": item["answers"],
            "label": item["label"]
        } for item in data]

# Winogrande数据集转换器
class WinograndeConverter(BaseConverter):
    def convert(self) -> List[Dict]:
        data = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "text": item["sentence"],
                    "question": "Which of the following options is the most appropriate one to fill in the blank?",
                    "options": [item["option1"], item["option2"]],
                    "label": int(item["answer"]) - 1
                })
        return data

# BalaCOPA数据集转换器
class BalacopaConverter(BaseConverter):
    def convert(self) -> List[Dict]:
        tree = ET.parse(self.input_path)
        root = tree.getroot()
        data = []
        for item in root.findall('item'):
            data.append({
                "text": item.find('p').text,
                "question": f"Which may be the {item.get('asks-for')}?",
                "options": [item.find('a1').text, item.find('a2').text],
                "label": int(item.get('most-plausible-alternative')) - 1
            })                        
        return data

# Boolean Questions数据集转换器
class BooleanQuestionsConverter(BaseConverter):
    def convert(self) -> List[Dict]:
        data = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question = item["question"]
                question = question[0].upper() + question[1:] + "?"  # 首字母大写并添加问号
                
                data.append({
                    "text": item["passage"],
                    "question": question,
                    "options": ["True", "False"],
                    "label": 0 if item["answer"] else 1
                })
        return data

def merge_datasets(output_dir: Path):
    """合并所有转换后的数据集"""
    all_data = []
    
    # 读取所有jsonl文件，但排除merged_dataset.jsonl
    for file in output_dir.glob("*.jsonl"):
        if file.name != "merged_dataset.jsonl":  # 排除已经合并的文件
            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
                # 添加数据集来源标记，使用原始文件名（不包含扩展名）
                for item in data:
                    item["source"] = file.stem
                all_data.extend(data)
    
    # 保存合并后的数据集
    merged_path = output_dir / "merged_dataset.jsonl"
    with open(merged_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n已合并所有数据集到: {merged_path}")
    print(f"总样本数: {len(all_data)}")

def main():
    # 环境检查
    if not check_environment():
        sys.exit(1)
        
    # 确保当前目录正确
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 数据集配置
    converters = [
        (RaceConverter, "converting/race.parquet", "race"),
        (ArcConverter, "converting/ARC-Challenge.parquet", "arc_challenge"),
        (ArcConverter, "converting/ARC-Easy.parquet", "arc_easy"),
        (LogiQAConverter, "converting/LogiQA2.0.txt", "logiqa"),
        (LogiQAConverter, "converting/LogiQA2.0-test.txt", "logiqa_val"),
        (ReclorConverter, "converting/reclor-train.json", "reclor_train"),
        (ReclorConverter, "converting/reclor-val.json", "reclor_val"),
        (WinograndeConverter, "converting/winogrande.jsonl", "winogrande"),
        (BalacopaConverter, "converting/balacopa.xml", "balacopa"),
        (BooleanQuestionsConverter, "converting/boolean-questions-train.jsonl", "boolean_train"),
        (BooleanQuestionsConverter, "converting/boolean-questions-dev.jsonl", "boolean_dev")
    ]
    
    # 验证文件存在
    for _, input_path, name in converters:
        check_file_exists(input_path)
    
    # 创建输出目录
    output_dir = Path("converted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 添加数据集统计
    dataset_stats = {}
    
    # 执行转换
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for converter_class, input_path, name in converters:
            try:
                if not check_file_exists(input_path):
                    continue
                    
                print(f"\n正在转换 {name}...")
                converter = converter_class(input_path)
                
                # 使用新的转换和过滤方法
                data = converter.convert_and_filter()
                
                # 如果没有有效数据，跳过保存
                if not data:
                    print(f"警告: {name} 没有有效数据")
                    continue
                
                # 统计数据集信息
                dataset_stats[name] = {
                    "样本数量": len(data),
                    "平均文本长度": sum(len(item["text"]) for item in data) / len(data),
                    "平均选项数": sum(len(item["options"]) for item in data) / len(data),
                    "选项数分布": dict(Counter(len(item["options"]) for item in data))
                }
                
                converter.save(data, str(output_dir), name)
                print(f"完成转换 {name}")
            except Exception as e:
                print(f"转换 {name} 失败: {str(e)}")
                continue
    
    # 输出统计信息
    print("\n数据集统计信息:")
    for name, stats in dataset_stats.items():
        print(f"\n{name}:")
        for key, value in stats.items():
            if key == "选项数分布":
                print(f"  {key}: {value}")  # 字典类型直接打印
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")  # 数值类型使用.2f格式化
            else:
                print(f"  {key}: {value}")  # 其他类型直接打印
    
    # # 合并数据集
    # merge_datasets(output_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)

# 使用说明
# 1. 先尝试卸载当前的pyarrow:
#    pip uninstall pyarrow
# 
# 2. 重新安装指定版本:
#    pip install pyarrow==12.0.1