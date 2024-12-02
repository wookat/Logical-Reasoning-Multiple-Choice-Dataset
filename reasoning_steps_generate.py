import json
import time
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ReasoningStepsGenerator:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", max_retries: int = 2):
        """初始化推理步骤生成器
        
        Args:
            api_key: DeepSeek API密钥
            base_url: DeepSeek API基础URL
            max_retries: 最大重试次数
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries
        self.system_prompt = """You are a logical reasoning expert. For each question, provide a detailed logical reasoning chain in JSON format.

The reasoning should follow these specific steps:
1. For deductive reasoning: Identify general principles and apply them to reach specific conclusions
2. For inductive reasoning: Find patterns from specific observations to make generalizations
3. For causal reasoning: Analyze cause-effect relationships and their implications
4. For analogical reasoning: Compare similar situations to draw parallels
5. For conditional reasoning: Evaluate if-then relationships and their consequences

EXAMPLE INPUT:
Text: All mammals are warm-blooded. Whales are mammals.
Question: Are whales warm-blooded?
Options:
A. Yes, because they are mammals
B. No, because they live in cold water
C. It depends on the season
D. Cannot be determined

EXAMPLE JSON OUTPUT:
{
    "reasoning_type": "deductive",
    "given_facts": [
        "All mammals are warm-blooded",
        "Whales are mammals"
    ],
    "logical_steps": [
        "Major premise: All mammals are warm-blooded",
        "Minor premise: Whales are mammals",
        "Application: If all mammals are warm-blooded, and whales are mammals, then whales must be warm-blooded"
    ],
    "conclusion": "Since whales are mammals, and all mammals are warm-blooded, whales must be warm-blooded",
    "answer": "A",
    "confidence": "high",
    "reasoning_pattern": "If A implies B, and X is A, then X is B"
}

Ensure your analysis follows formal logical principles and clearly shows each step of reasoning."""

    def generate_reasoning(self, text: str, question: str, options: List[str], correct_answer: str) -> Tuple[Optional[Dict], bool]:
        """为单个问题生成推理步骤，并验证答案
        
        Args:
            text: 问题背景文本
            question: 具体问题
            options: 选项列表
            correct_answer: 正确答案
        
        Returns:
            (推理步骤字典, 是否需要人工处理)
        """
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        prompt = f"Text: {text}\nQuestion: {question}\nOptions:\n{options_str}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000,
                    response_format={'type': 'json_object'},
                    stream=False
                )
                
                # 解析JSON响应
                reasoning = json.loads(response.choices[0].message.content)
                
                # 验证JSON格式
                required_fields = ['reasoning_type', 'given_facts', 'logical_steps', 'conclusion', 'answer', 'confidence', 'reasoning_pattern']
                if not all(field in reasoning for field in required_fields):
                    logger.warning(f"响应缺少必要字段: {reasoning}")
                    continue
                
                # 验证答案是否正确
                model_answer = reasoning['answer'].strip().upper()
                correct_answer = str(correct_answer).strip().upper()
                
                if model_answer == correct_answer:
                    return reasoning, False
                
                logger.warning(f"第{attempt + 1}次尝试答案不匹配: 模型={model_answer}, 正确={correct_answer}")
                
            except Exception as e:
                logger.error(f"生成推理步骤时出错: {str(e)}")
                if attempt == self.max_retries:
                    return None, True
        
        # 如果所有重试都失败了
        logger.warning(f"需要人工处理的样本:\nText: {text}\nQuestion: {question}\nOptions: {options_str}\nCorrect Answer: {correct_answer}")
        return None, True

    def process_dataset(self, input_path: str, output_path: str, manual_review_path: str):
        """处理整个数据集
        
        Args:
            input_path: 输入数据文件路径
            output_path: 输出数据文件路径
            manual_review_path: 需要人工审核的数据保存路径
        """
        logger.info(f"开始处理数据集: {input_path}")
        
        # 读取数据
        if input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            df = pd.DataFrame(data)
        
        total_samples = len(df)
        logger.info(f"总共需要处理 {total_samples} 个样本")
        
        # 初始化数据列表
        enhanced_data = []
        manual_review_data = []
        
        for idx, row in tqdm(df.iterrows(), total=total_samples):
            # 处理选项，确保它是列表
            options = row['options']
            if isinstance(options, str):
                options = json.loads(options)
            elif hasattr(options, 'tolist'):
                options = options.tolist()
            
            # 生成推理步骤
            reasoning, needs_review = self.generate_reasoning(
                text=row['text'],
                question=row['question'],
                options=options,
                correct_answer=chr(65 + row['label'])  # 将数字标签转换为字母
            )
            
            if needs_review:
                manual_review_data.append({
                    'text': row['text'],
                    'question': row['question'],
                    'options': options,
                    'label': row['label'],
                    'correct_answer': chr(65 + row['label'])
                })
                continue
            
            # 构建增强后的数据样本
            enhanced_sample = {
                'text': row['text'],
                'question': row['question'],
                'options': options,
                'label': row['label'],
                'reasoning_steps': reasoning
            }
            enhanced_data.append(enhanced_sample)
            
            # 定期保存数据
            if (idx + 1) % 100 == 0:
                self._save_data(enhanced_data, output_path)
                self._save_data(manual_review_data, manual_review_path)
                logger.info(f"已处理 {idx + 1} 个样本")
            
            # 添加延迟以避免API限制
            time.sleep(0.5)
        
        # 最终保存
        self._save_data(enhanced_data, output_path)
        self._save_data(manual_review_data, manual_review_path)
        
        logger.info(f"数据处理完成:")
        logger.info(f"- 成功处理的样本: {len(enhanced_data)}")
        logger.info(f"- 需要人工审核的样本: {len(manual_review_data)}")
        logger.info(f"- 成功率: {len(enhanced_data)/total_samples*100:.2f}%")

    def _save_data(self, data: List[Dict], output_path: str):
        """保存数据到文件"""
        if not data:  # 如果数据为空，不进行保存
            return
            
        if output_path.endswith('.parquet'):
            df = pd.DataFrame(data)
            df.to_parquet(output_path, index=False)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 配置参数
    api_key = "<Your-DeepSeek-API-Key>"  # 替换为您的API密钥
    input_path = "path/to/your/input/data"  # 替换为您的输入数据路径
    output_path = "path/to/your/output/data"  # 替换为您的输出数据路径
    manual_review_path = "path/to/manual_review_data"  # 替换为需要人工审核的数据保存路径
    
    # 创建生成器实例
    generator = ReasoningStepsGenerator(api_key=api_key)
    
    # 处理数据集
    generator.process_dataset(input_path, output_path, manual_review_path)

if __name__ == "__main__":
    main() 