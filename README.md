# DPSGD-LoRA: 差分隐私LoRA微调医疗对话模型

本项目实现了一个基于LoRA微调和差分隐私保护的医疗对话模型训练系统。通过结合参数高效微调技术和差分隐私机制，在保护训练数据隐私的同时保持模型性能。

## 项目简介

随着大语言模型在医疗领域的广泛应用，如何在训练过程中保护患者隐私成为一个重要问题。本项目采用LoRA（Low-Rank Adaptation）参数高效微调技术和差分隐私（Differential Privacy）机制，构建了一个兼顾模型性能和数据隐私的医疗对话系统。

### 核心特性

- **LoRA微调**: 使用低秩适应技术进行参数高效微调，大幅减少训练参数量
- **差分隐私保护**: 实现传统差分隐私机制，防止模型泄露训练数据中的敏感信息
- **医疗对话建模**: 专门针对医疗对话场景进行训练和优化
- **模型合并与部署**: 支持将LoRA适配器与基础模型合并，便于实际部署

## 快速开始

### 环境安装

```bash
# 克隆项目
git clone https://github.com/shanjf666/DPSGD_LORA.git
cd DPSGD_LORA

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

给定JSONL格式的对话数据train.jsonl，每行包含一个对话样本：

```json
{
  "conversations": [
    {"from": "human", "value": "患者问题"},
    {"from": "assistant", "value": "医生回答"}
  ]
}
```

使用 `extract.py` 脚本从train.jsonl数据集中提取对话数据集privacy_train.jsonl：

```bash
python extract.py
```

### 配置修改

修改 `config/training_config.py` 中的配置参数：

参考文档 `config.md`

### 模型训练

```bash
python main.py
```

训练过程包括：
1. 加载基础模型并应用LoRA适配器
2. 处理和分割训练数据
3. 计算差分隐私噪声参数
4. 使用手动实现的DPSGD进行训练
5. 评估模型性能
6. 合并LoRA适配器到基础模型

### 模型推理

使用 `infer_vllm.py` 脚本进行高效推理：

```bash
python infer_vllm.py \
  --model_path /root/autodl-tmp/model \
  --data_path /root/autodl-tmp/data/valid.jsonl \
  --output_path /root/autodl-tmp/infer/base.jsonl

python infer_vllm.py \
  --model_path /root/autodl-tmp/final/model8 \
  --data_path /root/autodl-tmp/data/valid.jsonl \
  --output_path /root/autodl-tmp/infer/result8.jsonl
```

### 对回答进行评分

使用 `score.py` 脚本进行评分：

```bash
python score.py \
  --model_name qwen/Qwen2.5-1.5B-Instruct \
  --test_data /root/autodl-tmp/data/valid.jsonl \
  --user_out_path /root/autodl-tmp/infer/result8.jsonl \
  --we_out_path /root/autodl-tmp/infer/base.jsonl \
  --out_path /root/autodl-tmp/score/score8.jsonl
```

## 项目架构

```
DPSGD_LORA/
├── config/                 # 训练配置
│   └── training_config.py
├── data/                   # 数据处理模块
│   ├── medical_dataset.py
│   └── data_processor.py
├── model/                  # 模型相关模块
│   ├── lora_model.py
│   ├── model_merger.py
│   └── model_utils.py
├── privacy/                # 差分隐私实现
│   ├── dp_calculator.py
│   └── manual_dp.py
├── trainer/                # 训练器模块
│   ├── dp_trainer.py
│   └── evaluator.py
├── utils/                  # 工具函数
│   └── logger.py
├── main.py                 # 训练主程序
├── extract.py              # 数据提取工具
├── infer_vllm.py           # 模型推理脚本
└── score.py                # 评分工具
```
