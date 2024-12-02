import json
import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import aiohttp
import asyncio
from tqdm import tqdm
import os
import argparse
from dataclasses import dataclass
from aiohttp import ClientSession, TCPConnector

# 配置参数
CONFIG = {
    # API相关配置
    "API": {
        "model": "deepseek-chat",
        "temperature": 0,
        "max_tokens": 1024,
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-d68b2c5a741140ac982d9a7853eb5f22",  # 替换为你的API密钥
    },
    
    # 重试配置
    "RETRY": {
        "max_retries": 3,
        "retry_delay": 5,  # 秒
    },
    
    # 批处理配置
    "BATCH": {
        "save_interval": 10,  # 每处理多少条样本保存一次
    },
    
    # 输出文件配置
    "OUTPUT": {
        "failed_samples_filename": "failed_samples.json",
        "encoding": "utf-8",
        "output_dir": "dataset/generated",  # 输出目录
    },
    
    # 数据处理配置
    "DATA": {
        "input_path": "dataset/mixed/full_train_low.jsonl",  # 默认输入路径
        "start_idx": 0,  # 起始索引
        "end_idx": None,  # 结束索引，None表示处理到结束
    },
    
    # 并发配置
    "CONCURRENT": {
        "max_concurrent_requests": 20,   # 最大并发请求数
        "batch_size": 50,               # 每批处理的样本数
        "rate_limit": 100,              # 每分钟最大请求数
        "request_interval": 0.5,        # 降低单个请求间隔
    },
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 系统提示词模板
SYSTEM_PROMPT = """You are an expert in logical reasoning, specializing in solving single-choice questions. Before addressing any question, you will take a deep breath, calmly analyze the problem, and proceed step by step to dissect the question and evaluate the options. Your response will be clear, well-structured, and logically rigorous. Always conclude your answer with the following format:
‘The answer is: (X) [Full content of the option]’, where X represents the option letter (A/B/C/D). Your analysis should reflect professionalism and lead to the most logical answer."""

def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            outputs = match.group(1)
            for i in options:
                if i in outputs:
                    return i
    return ''

@dataclass
class Sample:
    """表示待处理的样本"""
    text: str
    question: str
    options: List[str]
    label: int
    index: int  # 用于追踪原始顺序

class AsyncReasoningGenerator:
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = CONFIG["API"]["base_url"]
        self.failed_samples = []
        self.semaphore = asyncio.Semaphore(CONFIG["CONCURRENT"]["max_concurrent_requests"])
        self.rate_limiter = asyncio.Semaphore(CONFIG["CONCURRENT"]["rate_limit"])
        self.last_request_time = 0
        
    async def wait_for_rate_limit(self):
        """等待请求间隔"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < CONFIG["CONCURRENT"]["request_interval"]:
            await asyncio.sleep(CONFIG["CONCURRENT"]["request_interval"] - elapsed)
        self.last_request_time = time.time()
        
    async def generate_reasoning_for_sample(
        self,
        session: ClientSession,
        sample: Sample,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """为单个样本异步生成推理步骤"""
        
        formatted_options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample.options)])
        prompt = f"{sample.text}\nQuestion: {sample.question}\nOptions:\n{formatted_options}\nAnswer:\nTake a deep breath and work on this problem step-by-step."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": CONFIG["API"]["temperature"],
            "max_tokens": CONFIG["API"]["max_tokens"]
        }

        for attempt in range(max_retries):
            try:
                async with self.semaphore:  # 限制并发请求数
                    async with self.rate_limiter:  # 限制请求速率
                        await self.wait_for_rate_limit()  # 添加请求间隔
                        
                        async with session.post(
                            f"{self.base_url}/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=30  # 添加超时设置
                        ) as response:
                            if response.status == 418:  # 特别处理 418 错误
                                logging.warning(f"请求被限制，等待更长时间后重试...")
                                await asyncio.sleep(30)  # 遇到418错误时等待30秒
                                continue
                                
                            if response.status != 200:
                                raise Exception(f"API请求失败: {response.status}")
                            
                            result = await response.json()
                            reasoning = result["choices"][0]["message"]["content"]
                            
                            # 验证答案
                            if self.validate_answer(reasoning, sample.label):
                                return {
                                    "text": sample.text,
                                    "question": sample.question,
                                    "options": sample.options,
                                    "label": sample.label,
                                    "reasoning": reasoning,
                                    "index": sample.index
                                }
                            
                            if attempt < max_retries - 1:
                                await asyncio.sleep(CONFIG["RETRY"]["retry_delay"])
                                continue
                                
                            self.failed_samples.append({
                                "text": sample.text,
                                "question": sample.question,
                                "options": sample.options,
                                "label": sample.label,
                                "generated_reasoning": reasoning,
                                "failure_reason": "答案验证失败",
                                "index": sample.index
                            })
                            return None
                            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * CONFIG["RETRY"]["retry_delay"]
                    logging.warning(f"请求失败，等待 {wait_time} 秒后重试: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                    
                self.failed_samples.append({
                    "text": sample.text,
                    "question": sample.question,
                    "options": sample.options,
                    "label": sample.label,
                    "failure_reason": f"API错误: {str(e)}",
                    "index": sample.index
                })
                return None

    def validate_answer(self, reasoning: str, correct_label: int) -> bool:
        """验证生成的答案是否正确"""
        extracted_answer = first_option_postprocess(reasoning, 'ABCD')
        if not extracted_answer:
            return False
        predicted_label = ord(extracted_answer) - ord('A')
        return predicted_label == correct_label

class AsyncResultWriter:
    """异步结果写入器"""
    def __init__(self, output_path: str, failed_samples_path: str):
        self.output_path = output_path
        self.failed_samples_path = failed_samples_path
        self.results = []
        self.failed_samples = []
        self.lock = asyncio.Lock()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 初始化文件
        with open(self.output_path, 'w', encoding=CONFIG["OUTPUT"]["encoding"]) as f:
            pass
        with open(self.failed_samples_path, 'w', encoding=CONFIG["OUTPUT"]["encoding"]) as f:
            json.dump([], f, ensure_ascii=False)
            
    async def add_result(self, result: Dict[str, Any]):
        """实时添加并保存结果"""
        async with self.lock:
            self.results.append(result)
            # 移除临时索引
            result_to_save = result.copy()
            del result_to_save["index"]
            
            # 立即追加写入结果
            with open(self.output_path, 'a', encoding=CONFIG["OUTPUT"]["encoding"]) as f:
                f.write(json.dumps(result_to_save, ensure_ascii=False) + '\n')
            
            # 打印进度
            logging.info(
                f"成功处理样本 - 当前进度: {len(self.results)} 条, "
                f"失败: {len(self.failed_samples)} 条"
            )
                
    async def add_failed_sample(self, sample: Dict[str, Any]):
        """实时添加并保存失败样本"""
        async with self.lock:
            self.failed_samples.append(sample)
            # 移除临时索引
            sample_to_save = sample.copy()
            del sample_to_save["index"]
            
            # 更新失败样本文件
            with open(self.failed_samples_path, 'w', encoding=CONFIG["OUTPUT"]["encoding"]) as f:
                json.dump(self.failed_samples, f, ensure_ascii=False, indent=2)
            
            # 打印失败信息
            logging.warning(
                f"样本处理失败 - "
                f"问题: {sample.get('question', '未知')} - "
                f"原因: {sample.get('failure_reason', '未知')}"
            )

async def process_batch(
    session: ClientSession,
    generator: AsyncReasoningGenerator,
    batch: List[Sample],
    result_writer: AsyncResultWriter
) -> None:
    """并发处理一批数据"""
    tasks = []
    for sample in batch:
        task = generator.generate_reasoning_for_sample(session, sample)
        tasks.append((sample, asyncio.create_task(task)))
    
    for sample, task in tasks:
        try:
            result = await task
            if result:
                await result_writer.add_result(result)
            elif generator.failed_samples:
                await result_writer.add_failed_sample(generator.failed_samples[-1])
        except Exception as e:
            logging.error(f"处理样本时出错 - 问题: {sample.question[:50]}... - 错误: {str(e)}")

async def process_dataset_async(
    input_path: str,
    api_key: str,
    start_idx: int = 0,
    end_idx: int = None
) -> None:
    """异步处理数据集"""
    output_path, failed_samples_path = get_output_paths(input_path)
    
    # 读取数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    end_idx = end_idx or len(data)
    data_to_process = data[start_idx:end_idx]
    
    # 转换为Sample对象
    samples = [
        Sample(
            text=item['text'],
            question=item['question'],
            options=item['options'],
            label=item['label'],
            index=i
        )
        for i, item in enumerate(data_to_process)
    ]
    
    # 设置批处理大小
    BATCH_SIZE = CONFIG["CONCURRENT"]["batch_size"]
    batches = [samples[i:i + BATCH_SIZE] for i in range(0, len(samples), BATCH_SIZE)]
    
    generator = AsyncReasoningGenerator(api_key=api_key)
    result_writer = AsyncResultWriter(output_path, failed_samples_path)
    
    start_time = time.time()
    
    # 使用aiohttp的ClientSession进行并发请求
    async with aiohttp.ClientSession(
        connector=TCPConnector(limit=CONFIG["CONCURRENT"]["max_concurrent_requests"])
    ) as session:
        with tqdm(total=len(samples), desc="Processing samples") as pbar:
            for batch in batches:
                await process_batch(session, generator, batch, result_writer)
                pbar.update(len(batch))
    
    # 计算处理时间和统计信息
    end_time = time.time()
    processing_time = end_time - start_time
    total_samples = len(samples)
    success_count = len(result_writer.results)
    failed_count = len(result_writer.failed_samples)
    success_rate = success_count / total_samples * 100
    
    logging.info(
        f"\n处理完成! 统计信息:\n"
        f"- 总处理时间: {processing_time:.2f}秒\n"
        f"- 平均每题时间: {processing_time/total_samples:.2f}秒\n"
        f"- 总样本数: {total_samples}\n"
        f"- 成功处理: {success_count} 条 ({success_rate:.2f}%)\n"
        f"- 失败样本: {failed_count} 条\n"
        f"- 成功结果保存至: {output_path}\n"
        f"- 失败样本保存至: {failed_samples_path}"
    )

def get_output_paths(input_path: str) -> Tuple[str, str]:
    """根据输入路径生成输出路径
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        Tuple[str, str]: (输出文件路径, 失败样本文件路径)
    """
    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 创建输出目录
    output_dir = os.path.join(CONFIG["OUTPUT"]["output_dir"], input_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件路径
    output_path = os.path.join(output_dir, f"{input_filename}_reasoning.jsonl")
    failed_samples_path = os.path.join(output_dir, CONFIG["OUTPUT"]["failed_samples_filename"])
    
    return output_path, failed_samples_path

def main():
    parser = argparse.ArgumentParser(description='生成推理步骤')
    parser.add_argument('--input_path', type=str, help='输入数据集路径')
    parser.add_argument('--api_key', type=str, help='DeepSeek API密钥')
    parser.add_argument('--start_idx', type=int, help='起始索引')
    parser.add_argument('--end_idx', type=int, help='结束索引')
    
    args = parser.parse_args()
    
    input_path = args.input_path or CONFIG["DATA"]["input_path"]
    api_key = args.api_key or CONFIG["API"]["api_key"]
    start_idx = args.start_idx if args.start_idx is not None else CONFIG["DATA"]["start_idx"]
    end_idx = args.end_idx if args.end_idx is not None else CONFIG["DATA"]["end_idx"]
    
    asyncio.run(process_dataset_async(
        input_path=input_path,
        api_key=api_key,
        start_idx=start_idx,
        end_idx=end_idx
    ))

if __name__ == "__main__":
    main() 