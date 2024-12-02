import json
import logging
import re
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import os
from generate_reasoning_steps import first_option_postprocess

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 在文件开头添加配置部分
CONFIG = {
    # 输入输出配置
    "IO": {
        "input_path": "dataset/peft_train_strict/peft_train_strict_reasoning.jsonl",
        "output_dir": "dataset/peft_train_strict/validation_results",
        "encoding": "utf-8"
    },
    
    # 验证标准配置
    "VALIDATION": {
        "min_reasoning_length": 0,  # 最小推理文本长度
        "max_reasoning_length": 200000,  # 最大推理文本长度
        "required_step_count": 0,  # 最少需要的推理步骤数
    }
}

class ReasoningValidator:
    def __init__(self, validation_config: Dict = None):
        """初始化验证器
        
        Args:
            validation_config: 验证配置参数
        """
        self.validation_config = validation_config or CONFIG["VALIDATION"]
        self.validation_results = {
            "total": 0,
            "valid_answer": 0,
            "has_step_by_step": 0,
            "has_conclusion": 0,
            "invalid_samples": []
        }

    def validate_answer(self, reasoning: str, correct_label: int) -> bool:
        """验证生成的答案是否正确
        
        Args:
            reasoning: 生成的推理文本
            correct_label: 正确答案的索引(0-3对应A-D)
            
        Returns:
            bool: 答案是否正确
        """
        extracted_answer = first_option_postprocess(reasoning, 'ABCDEF')
        if not extracted_answer:
            return False
            
        predicted_label = ord(extracted_answer) - ord('A')
        return predicted_label == correct_label

    def has_step_by_step_reasoning(self, reasoning: str) -> bool:
        """检查是否包含逐步推理过程
        
        Args:
            reasoning: 推理文本
            
        Returns:
            bool: 是否包含逐步推理
        """
        # 检查是否包含表示步骤的关键词
        step_patterns = [
            r"[Ll]et's think step by step",
            r"步骤[：:]\s*\d+",
            r"[Ss]tep\s*\d+",
            r"首先",
            r"其次",
            r"最后",
            r"第[一二三四五]",
            r"[Ff]irst",
            r"[Ss]econd",
            r"[Tt]hird",
            r"[Ff]inally"
        ]
        
        for pattern in step_patterns:
            if re.search(pattern, reasoning):
                return True
        return False

    def has_conclusion(self, reasoning: str) -> bool:
        """检查是否包含结论
        
        Args:
            reasoning: 推理文本
            
        Returns:
            bool: 是否包含结论
        """
        conclusion_patterns = [
            r"[Tt]herefore",
            r"[Tt]hus",
            r"[Tt]he answer is",
            r"[Tt]he correct answer is",
            r"所以",
            r"因此",
            r"综上所述",
            r"答案[是为]",
            r"选择",
            r"选项"
        ]
        
        for pattern in conclusion_patterns:
            if re.search(pattern, reasoning):
                return True
        return False

    def validate_sample(self, sample: Dict) -> Tuple[bool, List[str]]:
        """验证单个样本
        
        Args:
            sample: 包含推理步骤的样本
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 问题列表)
        """
        problems = []
        reasoning = sample["reasoning"]
        
        # 验证文本长度
        if len(reasoning) < self.validation_config["min_reasoning_length"]:
            problems.append(f"推理文本过短 ({len(reasoning)} 字符)")
        elif len(reasoning) > self.validation_config["max_reasoning_length"]:
            problems.append(f"推理文本过长 ({len(reasoning)} 字符)")
        
        # 验证答案正确性
        if not self.validate_answer(reasoning, sample["label"]):
            problems.append("答案提取失败或不正确")
            
        # 验证推理步骤数量
        step_count = self.count_reasoning_steps(reasoning)
        if step_count < self.validation_config["required_step_count"]:
            problems.append(f"推理步骤数量不足 (需要 {self.validation_config['required_step_count']} 步，实际 {step_count} 步)")
            
        # 验证是否包含逐步推理
        if not self.has_step_by_step_reasoning(reasoning):
            problems.append("缺少逐步推理过程")
            
        # 验证是否包含结论
        if not self.has_conclusion(reasoning):
            problems.append("缺少明确的结论")
            
        return len(problems) == 0, problems

    def count_reasoning_steps(self, reasoning: str) -> int:
        """计算推理步骤数量
        
        Args:
            reasoning: 推理文本
            
        Returns:
            int: 推理步骤数量
        """
        step_patterns = [
            r"步骤\s*\d+",
            r"[Ss]tep\s*\d+",
            r"第[一二三四五]步",
            r"[Ff]irst",
            r"[Ss]econd",
            r"[Tt]hird",
            r"[Ff]ourth",
            r"[Ff]ifth"
        ]
        
        step_count = 0
        for pattern in step_patterns:
            step_count += len(re.findall(pattern, reasoning))
        return step_count

    def validate_dataset(self, input_path: str) -> Dict:
        """验证整个数据集
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            Dict: 验证结果统计
        """
        logging.info(f"开始验证数据集: {input_path}")
        
        # 读取数据集
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
                
        self.validation_results["total"] = len(samples)
        
        # 验证每个样本
        for sample in tqdm(samples, desc="验证样本"):
            is_valid, problems = self.validate_sample(sample)
            
            if is_valid:
                self.validation_results["valid_answer"] += 1
            else:
                invalid_sample = {
                    "question": sample["question"],
                    "reasoning": sample["reasoning"],
                    "problems": problems
                }
                self.validation_results["invalid_samples"].append(invalid_sample)
                
            if self.has_step_by_step_reasoning(sample["reasoning"]):
                self.validation_results["has_step_by_step"] += 1
                
            if self.has_conclusion(sample["reasoning"]):
                self.validation_results["has_conclusion"] += 1
                
        return self.validation_results

    def save_validation_results(self, results: Dict, output_dir: str):
        """保存验证结果
        
        Args:
            results: 验证结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存统计信息
        stats = {
            "total_samples": results["total"],
            "valid_answers": results["valid_answer"],
            "samples_with_steps": results["has_step_by_step"],
            "samples_with_conclusion": results["has_conclusion"],
            "invalid_samples_count": len(results["invalid_samples"]),
            "valid_answer_rate": results["valid_answer"] / results["total"] * 100,
            "step_by_step_rate": results["has_step_by_step"] / results["total"] * 100,
            "conclusion_rate": results["has_conclusion"] / results["total"] * 100
        }
        
        with open(os.path.join(output_dir, "validation_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        # 保存无效样本
        if results["invalid_samples"]:
            with open(os.path.join(output_dir, "invalid_samples.json"), 'w', encoding='utf-8') as f:
                json.dump(results["invalid_samples"], f, indent=2, ensure_ascii=False)
                
        logging.info(f"验证结果已保存至: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='验证推理步骤数据集')
    parser.add_argument(
        '--input_path', 
        type=str, 
        default=CONFIG["IO"]["input_path"],
        help='输入数据集路径'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=CONFIG["IO"]["output_dir"],
        help='输出目录'
    )
    parser.add_argument(
        '--min_length', 
        type=int, 
        default=CONFIG["VALIDATION"]["min_reasoning_length"],
        help='最小推理文本长度'
    )
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=CONFIG["VALIDATION"]["max_reasoning_length"],
        help='最大推理文本长度'
    )
    parser.add_argument(
        '--required_steps', 
        type=int, 
        default=CONFIG["VALIDATION"]["required_step_count"],
        help='最少需要的推理步骤数'
    )
    
    args = parser.parse_args()
    
    # 更新配置
    CONFIG["VALIDATION"]["min_reasoning_length"] = args.min_length
    CONFIG["VALIDATION"]["max_reasoning_length"] = args.max_length
    CONFIG["VALIDATION"]["required_step_count"] = args.required_steps
    
    validator = ReasoningValidator(CONFIG["VALIDATION"])
    results = validator.validate_dataset(args.input_path)
    validator.save_validation_results(results, args.output_dir)
    
    # 打印验证结果摘要
    total = results["total"]
    logging.info(
        f"\n验证完成! 结果统计:\n"
        f"- 总样本数: {total}\n"
        f"- 有效答案: {results['valid_answer']} ({results['valid_answer']/total*100:.2f}%)\n"
        f"- 包含步骤: {results['has_step_by_step']} ({results['has_step_by_step']/total*100:.2f}%)\n"
        f"- 包含结论: {results['has_conclusion']} ({results['has_conclusion']/total*100:.2f}%)\n"
        f"- 无效样本: {len(results['invalid_samples'])}\n"
        f"验证配置:\n"
        f"- 最小长度: {CONFIG['VALIDATION']['min_reasoning_length']}\n"
        f"- 最大长度: {CONFIG['VALIDATION']['max_reasoning_length']}\n"
        f"- 最少步骤: {CONFIG['VALIDATION']['required_step_count']}\n"
    )

if __name__ == "__main__":
    main() 