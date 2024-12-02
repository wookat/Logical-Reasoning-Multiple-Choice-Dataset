from openai import OpenAI
import json
import asyncio 
import aiohttp
from tqdm import tqdm
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = "sk-b7b406654e6240bb8c3e0468cb8d3e0e"
BASE_URL = "https://api.deepseek.com/beta/v1/chat/completions"  # 直接使用API endpoint

SYSTEM_PROMPT = """
You are a logical reasoning expert. For each multiple choice question, please analyze and output your analysis in JSON format.

Please provide:
1. A step-by-step reasoning process showing your thought process to reach the answer
2. The correct answer (A, B, C, D, etc.)
3. A difficulty level assessment:
   1 - Basic (Simple one-step reasoning)
   2 - Intermediate (Two to three steps reasoning)
   3 - Advanced (Complex multi-step reasoning)

4. Question type (e.g., Scientific, Mathematical, Logical, Reading, Temporal, Comparative, Causal, Ethical)

5. Reasoning type (e.g., Deductive, Inductive, Analogical, Causal, Comparative, Sequential, Conditional)

6. Validation of your analysis (is_valid: true/false)

EXAMPLE INPUT:
Today, Alex had a busy schedule. He woke up at 8am. Mia saw Alex jogging in the park from 8am to 9am. 
When could Alex have visited the museum? 
Options: (A) 3pm to 5pm (B) 8am to 9am (C) 9am to 10am (D) 11am to 3pm

Let's think step by step:

EXAMPLE JSON OUTPUT:
{
    "reasoning_process": [
        "Let's analyze Alex's schedule systematically:",
        "1. We know Alex was jogging from 8am to 9am",
        "2. Looking at the options:",
        "   - Option B (8am-9am): Impossible, he was jogging",
        "   - Option C (9am-10am): Need to check other activities",
        "   - Option D (11am-3pm): Need to check other activities",
        "   - Option A (3pm-5pm): This time slot is currently unaccounted for",
        "3. By process of elimination, only 3pm-5pm is possible",
        "4. This makes logical sense as it's the only time slot without a confirmed activity"
    ],
    "correct_answer": "A",
    "difficulty_level": 2,
    "question_type": "Temporal",
    "reasoning_type": "Sequential",
    "is_valid": true
}
"""

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def construct_question(data: Dict[str, Any]) -> str:
    text = data['text'].strip()
    question = data['question'].strip()
    options = [f"({chr(65+i)}) {opt}" for i, opt in enumerate(data['options'])]
    
    full_question = ""
    if text:
        full_question += f"{text}\n\n"
    if question:
        full_question += f"{question}\n"
    full_question += "\nOptions: " + " ".join(options)
    full_question += "\n\nLet's think step by step:"
    
    return full_question

async def analyze_question_async(session: aiohttp.ClientSession, question: str) -> Dict[str, Any]:
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 1000
        }
        
        async with session.post(BASE_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return json.loads(result['choices'][0]['message']['content'])
            else:
                error_text = await response.text()
                logger.error(f"API Error: {response.status} - {error_text}")
                return None
                
    except Exception as e:
        logger.error(f"Error analyzing question: {e}")
        return None

def verify_answer(result: Dict[str, Any], correct_label: int) -> bool:
    if result is None:
        return False
    
    predicted_answer = result['correct_answer']
    correct_answer = chr(65 + correct_label)
    
    is_correct = predicted_answer == correct_answer
    if not is_correct:
        logger.warning(f"Predicted answer: {predicted_answer}, Correct answer: {correct_answer}")
    return is_correct

async def process_batch(session: aiohttp.ClientSession, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks = []
    for data in batch:
        question = construct_question(data)
        task = analyze_question_async(session, question)
        tasks.append((data, asyncio.create_task(task)))
    
    results = []
    for data, task in tasks:
        try:
            result = await task
            if result:
                is_correct = verify_answer(result, data['label'])
                results.append({
                    'question_id': data.get('index', len(results)),
                    'source': data.get('source', 'unknown'),
                    'analysis': result,
                    'is_correct': is_correct,
                    'original_data': data
                })
        except Exception as e:
            logger.error(f"Error processing question {data.get('index')}: {e}")
    
    return results

async def main_async():
    # 加载数据集
    dataset = load_jsonl('dataset/mixed/peft_train_strict_low.jsonl')
    logger.info(f"加载数据集完成，共 {len(dataset)} 条数据")
    
    # 设置并发参数
    BATCH_SIZE = 20  # 增大批处理大小
    
    all_results = []
    batches = [dataset[i:i + BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]
    
    start_time = time.time()
    
    # 使用aiohttp.ClientSession进行并发请求
    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(dataset), desc="Processing questions") as pbar:
            for batch in batches:
                batch_results = await process_batch(session, batch)
                all_results.extend(batch_results)
                pbar.update(len(batch))
    
    # 计算统计信息
    end_time = time.time()
    processing_time = end_time - start_time
    correct_count = sum(1 for r in all_results if r['is_correct'])
    accuracy = correct_count / len(dataset)
    
    # 按推理类型统计准确率
    type_stats = {}
    for result in all_results:
        r_type = result['analysis']['reasoning_type']
        if r_type not in type_stats:
            type_stats[r_type] = {'total': 0, 'correct': 0}
        type_stats[r_type]['total'] += 1
        if result['is_correct']:
            type_stats[r_type]['correct'] += 1
    
    # 输出详细统计信息
    logger.info(f"\n分析完成!")
    logger.info(f"处理时间: {processing_time:.2f}秒")
    logger.info(f"平均每题时间: {processing_time/len(dataset):.2f}秒")
    logger.info(f"总题数: {len(dataset)}")
    logger.info(f"正确数: {correct_count}")
    logger.info(f"总体准确率: {accuracy:.2%}")
    
    logger.info("\n各类型题目统计:")
    for r_type, stats in type_stats.items():
        type_accuracy = stats['correct'] / stats['total']
        logger.info(f"{r_type}: {stats['correct']}/{stats['total']} = {type_accuracy:.2%}")
    
    # 保存结果
    output_file = 'analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n结果已保存至: {output_file}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    import time
    main()



