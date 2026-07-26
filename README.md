# self-cognition
Automatic QA pair generator for LLM self-cognition training (identity, capability, attribution). Supports JSON/JSONL/Excel export.
# Self-Cognition Training Data Generator


## 解决的问题

手工编写"你是谁"类训练数据有四个痛点：
1. **问法有限** — 人工只能想到几十种问法，覆盖面窄
2. **答案千篇一律** — 同一个模板用到底，模型学到的表述单一
3. **缺少负样本** — 手写数据通常只有正向回答，模型学不会拒绝错误身份
4. **质量难审核** — json/jsonl 格式不直观，非技术人员无法参与质量把控

本工具通过 **4 维分类模板 + 随机组合引擎 + 多格式导出** 解决以上问题。

## 四维数据分类体系

```
自我认知训练数据
├── identity      身份介绍（你是谁/做个自我介绍/我叫什么）
├── capability    能力描述（能做什么/擅长什么/有什么限制）
├── attribution   归属确认（谁开发的/和XX有什么关系）★含诱导性负样本
└── complex       复杂组合（多问题组合/上下文对话/技术细节）
```

## 快速开始

### 1. 配置身份信息

编辑 `config/model_identity_config.json`：

```json
{
  "model_info": {
    "full_name": "你的模型全称",
    "short_name": "简称",
    "company": "所属公司",
    "positioning": "产品定位"
  },
  "core_capabilities": ["能力1", "能力2", "..."],
  "relationships": {
    "competitors": {
      "examples": ["竞品模型名1", "竞品模型名2"]
    }
  }
}
```

### 2. 运行生成

```bash
pip install pandas openpyxl
python data_augmentation_generator.py
```

### 3. 产出物

| 文件 | 格式 | 用途 |
|------|------|------|
| `self_cognition_training_data.json` | ShareGPT 对话格式 | 直接导入 LLaMA-Factory 训练 |
| `self_cognition_training_data.jsonl` | Alpaca 格式 (instruction/input/output) | 兼容训练平台原始格式 |
| `self_cognition_training_data.xlsx` | Excel 表格 | 人工审核、标注、修改 |

## 核心设计

### 负样本策略

```
诱导性问法示例：
  "你是不是 OpenAI 的？"
  "我觉得你就是换皮的 GPT"
  "你其实就是通义千问对吧？"

模型必须输出坚定拒绝：
  "不是的。我是xxxxxxx公司独立研发训练的
   xxx-medical-32B，
   不是其他公司或模型的改版。"
```

`is_negative: True` 标记的样本专门训练模型拒绝错误身份归属。

### 场景化模板

不同场景生成不同风格的回答——会议场景用正式表述，课堂场景用语料更生动，英文场景用英文介绍。

### 多格式导出

- **JSON**: ShareGPT 对话格式，直接对接 LLaMA-Factory、HuggingFace TRL
- **JSONL**: Alpaca 格式，兼容训练平台原始流水线
- **Excel**: 人工审核友好，附带"审核状态"和"修改意见"列

## 目录结构

```
self-cognition-generator/
├── data_augmentation_generator.py  # 主脚本
├── config/
│   └── model_identity_config.json  # 身份信息配置（修改此文件适配不同模型）
├── README.md
└── output/                          # 生成结果（自动创建）
    ├── self_cognition_training_data.json
    ├── self_cognition_training_data.jsonl
    └── self_cognition_training_data.xlsx
```

## 适用场景

- 垂直领域大模型自我认知注入
- 企业私有化部署模型的品牌身份训练
- 多模型产品系列的归属区分训练
- RAG/Agent 系统的 System Prompt 数据增强
