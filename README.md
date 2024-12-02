
# 逻辑推理数据集处理工具

这是一个用于处理和转换多个逻辑推理数据集的工具集。该工具可以将不同格式的数据集统一转换为标准格式，并进行数据集的混合、平衡和分析。

## 功能特性

- 支持多种数据集格式的转换
- 自动数据清洗和验证
- 数据集混合与采样
- 标签平衡处理
- 详细的数据分析报告
- 可视化分析图表生成

## 支持的数据集

- LogiQA
- ReClor
- Winogrande
- ARC (Challenge & Easy)
- RACE
- BalaCOPA
- Boolean Questions

## 数据处理流程

1. **数据转换** (dataset_convert.py)
   - 将各种格式的数据集转换为统一的 JSONL 格式
   - 进行数据清洗和验证
   - 生成初步的数据统计信息

2. **数据集混合** (dataset_mix.py)
   - 按照预设比例混合不同数据集
   - 处理重复样本
   - 生成训练集和评估集
   - 创建数据分布分析报告

3. **标签平衡** (balance_labels.py)
   - 分析并平衡各数据集的标签分布
   - 生成平衡后的数据集
   - 提供分布对比可视化

4. **长度分析** (prompt_length_analyze.py)
   - 分析文本长度分布
   - 生成长度分布报告和可视化图表
   - 识别异常长度样本

## 输出文件结构

```
dataset/
├── converted/          # 转换后的标准格式数据
├── mixed/             # 混合后的数据集
│   ├── full_train.jsonl
│   ├── peft_train.jsonl
│   ├── eval.jsonl
│   └── analysis/      # 分析报告和图表
├── logs/              # 处理日志
└── invalid_samples/   # 无效样本记录
```

## 数据格式规范

所有转换后的数据集采用统一的 JSONL 格式：

```json
{
    "text": "上下文文本",
    "question": "问题描述",
    "options": ["选项A", "选项B", "选项C", "选项D"],
    "label": 0,  //正确答案的索引
    "source": "数据集来源"
}
```

## 使用方法

1. 数据转换：
```bash
python dataset_convert.py
```

2. 数据集混合：
```bash
python dataset_mix.py
```

3. 标签平衡：
```bash
python balance_labels.py
```

4. 长度分析：
```bash
python prompt_length_analyze.py
```

## 数据集统计

当前处理的数据集总量：164,902 个样本，其中：
- RACE: 87,848 (53.27%)
- Winogrande: 40,394 (24.50%)
- LogiQA: 14,582 (8.84%)
- Boolean Questions: 12,697 (7.70%)
- ReClor: 5,138 (3.11%)
- ARC: 3,244 (1.96%)
- BalaCOPA: 999 (0.61%)

## 注意事项

1. 确保所有输入数据集文件位于正确的目录
2. 运行前检查环境依赖是否满足
3. 建议使用 Python 3.8 或更高版本
4. 处理大型数据集时注意内存使用

## 依赖要求

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pyarrow (可选)
- fastparquet (可选)

## 许可证

MIT License


这个 README.md 文件提供了项目的完整概述，包括功能特性、支持的数据集、处理流程、使用方法等关键信息。基于代码分析，特别是：


```528:540:dataset/dataset_convert.py
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
```


用于确认支持的数据集列表，以及：


```5:18:dataset/analysis/merged/dataset_composition.md
| 数据集 | 样本数量 | 占比 |
|--------|----------|------|
| race | 87,848 | 53.27% |
| winogrande | 40,394 | 24.50% |
| logiqa | 13,024 | 7.90% |
| boolean_train | 9,427 | 5.72% |
| reclor_train | 4,638 | 2.81% |
| boolean_dev | 3,270 | 1.98% |
| arc_easy | 2,149 | 1.30% |
| logiqa_val | 1,558 | 0.94% |
| arc_challenge | 1,095 | 0.66% |
| balacopa | 999 | 0.61% |
| reclor_val | 500 | 0.30% |
| **总计** | 164,902 | 100% |
```
