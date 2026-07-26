#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Cognition Training Data Generator
=======================================
LLM自我认知训练数据自动生成器 — 基于模板引擎的多维度QA对批量生产工具。

核心设计:
  - 4维分类: identity / capability / attribution / complex
  - 负样本策略: 诱导性提问 + 坚定拒绝回答
  - 场景化模板: 会议/教学/商务等不同场景自适应回答风格
  - 身份信息与代码分离: 通过 config/model_identity_config.json 配置

输出格式:
  - ShareGPT JSON -> LLaMA-Factory
  - Alpaca JSONL  -> 训练平台原始流水线
  - Excel         -> 人工审核

Usage:
  python data_augmentation_generator.py

License: MIT
"""

import json
import random
from typing import List, Dict, Optional
import pandas as pd
from itertools import product
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfCognitionDataGenerator:
    """自我认知数据生成器"""
    
    def __init__(self, output_format: str = "json", identity_config_path: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            output_format: 输出格式，支持 json/jsonl
            identity_config_path: 身份信息配置文件路径
        """
        self.output_format = output_format
        self.generated_data = []
        
        # 加载核心身份信息配置
        self.identity_config = self._load_identity_config(identity_config_path)
        
        # 从配置文件提取常用信息（向后兼容）
        self._extract_core_identity()
    
    def _load_identity_config(self, config_path: Optional[str] = None) -> Dict:
        """
        加载核心身份信息配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        # 如果未指定路径，使用默认路径（与当前脚本同目录下的config文件夹）
        if config_path is None:
            current_dir = Path(__file__).resolve().parent
            config_path = current_dir / "config" / "model_identity_config.json"
            logger.info(f"📂 配置文件路径: {config_path}")
        else:
            config_path = Path(config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ 成功加载身份配置文件: {config_path}")
            logger.info(f"   配置版本: {config.get('version', 'unknown')}")
            return config
        except FileNotFoundError:
            logger.warning(f"⚠️ 未找到配置文件: {config_path}，将使用默认配置")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"❌ 配置文件JSON格式错误: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置（示例配置，正式使用请修改 config/model_identity_config.json）"""
        return {
            "model_info": {
                "full_name": "YourModel-FullName-v1.0",
                "short_name": "YourModel",
                "company": "YourCompany",
                "positioning": "Your product positioning statement"
            },
            "core_capabilities": [
                "text generation", "question answering", "code generation",
                "teaching assistance", "knowledge graph generation", "exam paper generation"
            ],
            "relationships": {
                "competitors": {
                    "examples": ["CompetitorModel-A", "CompetitorModel-B", "CompetitorModel-C"]
                },
                "sibling_models": {
                    "examples": ["YourModel-Variant-A", "YourModel-Variant-B"]
                }
            }
        }
    
    def _extract_core_identity(self):
        """从配置文件提取核心身份信息（向后兼容）"""
        model_info = self.identity_config.get('model_info', {})
        
        # 核心身份信息
        self.CORE_IDENTITY = {
            "名称": model_info.get('full_name', ''),
            "简称": model_info.get('short_name', ''),
            "公司": model_info.get('company', ''),
            "定位": model_info.get('positioning', ''),
            "能力": self.identity_config.get('core_capabilities', [])
        }
        
        # 竞品模型 - 从details中提取公司信息
        competitors_config = self.identity_config.get('relationships', {}).get('competitors', {})
        competitors_examples = competitors_config.get('examples', [])
        competitors_details = competitors_config.get('details', {})
        
        self.COMPETITOR_MODELS = {}
        for comp in competitors_examples:
            if comp in competitors_details:
                self.COMPETITOR_MODELS[comp] = competitors_details[comp].get('company', '')
            else:
                self.COMPETITOR_MODELS[comp] = ''
        
        # 同系列模型 - 提取专业方向信息
        siblings = self.identity_config.get('relationships', {}).get('sibling_models', {}).get('examples', [])
        # 从模型名称中提取专业方向（如 Nebula-vEdu(IM) -> IM -> 信息管理）
        domain_mapping = {
            'IM': '信息管理',
            'CS': '计算机科学',
            'COMM': '通信'
        }
        self.SIBLING_MODELS = {}
        for sibling in siblings:
            # 提取括号中的专业代码
            import re
            match = re.search(r'\((\w+)\)', sibling)
            domain_code = match.group(1) if match else ''
            domain_name = domain_mapping.get(domain_code, domain_code)
            self.SIBLING_MODELS[sibling] = {
                "全称": sibling,
                "专业": domain_name,
                "代码": domain_code,
                "英文": domain_code  # 英文缩写即为代码
            }
        
        logger.info(f"📝 已加载核心身份信息:")
        logger.info(f"   模型名称: {self.CORE_IDENTITY['名称']}")
        logger.info(f"   公司: {self.CORE_IDENTITY['公司']}")
        logger.info(f"   核心能力: {', '.join(self.CORE_IDENTITY['能力'][:3])}...")
        logger.info(f"   竞品模型: {len(self.COMPETITOR_MODELS)}个")
        logger.info(f"   同系列模型: {len(self.SIBLING_MODELS)}个")
    
    def generate_identity_questions(self) -> List[Dict]:
        """生成身份介绍类问题（目标200条）"""
        data = []
        
        # 1. 直接提问变体（50条）
        direct_questions = [
            "你是谁？", "你是谁", "您是谁？", "你是什么？",
            "请问你是谁？", "能告诉我你是谁吗？", "介绍下你是谁",
            "说说你是谁", "你到底是谁", "你究竟是谁",
            "你的身份是什么", "你的名字是什么", "你叫什么名字",
            "你叫什么", "你的名称是什么", "请说明你的身份",
            "请介绍你的身份", "你的身份信息", "告诉我你的身份",
            "who are you", "What are you", "Introduce yourself",
            "你是什么东西", "你是个啥", "你啥身份",
            "你的身份", "身份信息", "你是？",
        ]
        
        for q in direct_questions:
            data.append({
                "question": q,
                "answer": self._generate_standard_intro(),
                "category": "identity_direct",
                "difficulty": "easy"
            })
        
        # 2. 自我介绍类（50条）
        intro_questions = [
            "介绍一下你自己", "自我介绍", "请介绍你自己",
            "介绍下自己", "说说你自己", "讲讲你自己",
            "简单介绍你自己", "详细介绍你自己", "简要介绍你自己",
            "用一句话介绍你自己", "用几句话介绍你自己",
            "介绍一下你", "介绍下你", "说明一下你是什么",
            "能介绍一下你吗", "可以介绍一下你自己吗",
            "做个自我介绍", "来个自我介绍", "自我介绍一下",
            "introduce yourself", "tell me about yourself",
            "用英文介绍你自己", "用中文介绍你自己",
            "向我介绍你自己", "给我介绍介绍你",
            "你是做什么的", "你是干什么的", "你的作用是什么",
        ]
        
        for q in intro_questions:
            if "详细" in q or "几句话" in q:
                answer = self._generate_detailed_intro()
            elif "一句话" in q or "简单" in q or "简要" in q:
                answer = self._generate_brief_intro()
            elif "英文" in q or "english" in q.lower():
                answer = self._generate_english_intro()
            else:
                answer = self._generate_standard_intro()
            
            data.append({
                "question": q,
                "answer": answer,
                "category": "identity_introduction",
                "difficulty": "easy"
            })
        
        # 3. 场景化介绍（50条）
        scenarios = [
            ("在会议中", "会议场景"),
            ("向学生", "教学场景"),
            ("向老师", "教学场景"),
            ("向领导", "正式场景"),
            ("向同事", "工作场景"),
            ("向客户", "商务场景"),
            ("在课堂上", "教学场景"),
            ("给新用户", "用户引导"),
        ]
        
        for scenario, context in scenarios:
            questions = [
                f"{scenario}介绍一下你自己",
                f"如果{scenario}，你会怎么介绍自己",
                f"{scenario}的自我介绍",
                f"在{scenario[1:]}怎么介绍你",
                f"{scenario}说明你的身份",
                f"{scenario}做个自我介绍",
            ]
            for q in questions:
                data.append({
                    "question": q,
                    "answer": self._generate_contextual_intro(context),
                    "category": "identity_scenario",
                    "difficulty": "medium"
                })
        
        # 4. 长度限定类（30条）
        length_requirements = [
            ("一句话", self._generate_one_sentence_intro),
            ("三句话", self._generate_three_sentence_intro),
            ("简短地", self._generate_brief_intro),
            ("详细地", self._generate_detailed_intro),
            ("50字以内", self._generate_brief_intro),
            ("100字左右", self._generate_standard_intro),
        ]
        
        for length_req, answer_func in length_requirements:
            questions = [
                f"{length_req}介绍你自己",
                f"请{length_req}说明你的身份",
                f"{length_req}讲讲你是谁",
                f"用{length_req}介绍一下你",
                f"{length_req}自我介绍",
            ]
            for q in questions:
                data.append({
                    "question": q,
                    "answer": answer_func(),
                    "category": "identity_length_limited",
                    "difficulty": "medium"
                })
        
        return data[:200]  # 限制200条
    
    def generate_capability_questions(self) -> List[Dict]:
        """生成能力描述类问题（目标150条）"""
        data = []
        
        # 1. 总体能力询问（40条）
        capability_questions = [
            "你能做什么？", "你可以做什么？", "你有什么能力？",
            "你的能力有哪些？", "你会什么？", "你擅长什么？",
            "你的功能是什么？", "你有哪些功能？", "你能帮我做什么？",
            "你可以帮我什么？", "你能提供什么服务？",
            "你的主要功能", "你的核心能力", "你具备哪些能力",
            "说说你的能力", "介绍你的能力", "列举你的功能",
            "你都能干什么", "你会干什么", "你能干啥",
            "你有什么用", "你的用途", "你的作用是什么",
            "What can you do", "What are your capabilities",
            "你的技能", "你掌握哪些技能", "你的专长",
        ]
        
        for q in capability_questions:
            data.append({
                "question": q,
                "answer": self._generate_capability_intro(),
                "category": "capability_general",
                "difficulty": "easy"
            })
        
        # 2. 场景化能力询问（50条）
        scenarios = [
            "在教学中", "在编程时", "在学习中", "在工作中",
            "帮助学生", "帮助老师", "辅助教学", "代码开发",
            "知识问答", "文档处理", "数据分析"
        ]
        
        for scenario in scenarios:
            questions = [
                f"在{scenario}你能做什么？",
                f"{scenario}你可以帮什么忙？",
                f"{scenario}有什么用？",
                f"如何在{scenario}使用你？",
                f"{scenario}的应用场景",
            ]
            for q in questions:
                data.append({
                    "question": q,
                    "answer": self._generate_scenario_capability(scenario),
                    "category": "capability_scenario",
                    "difficulty": "medium"
                })
        
        # 3. 限制和边界（30条）
        limitation_questions = [
            "你不能做什么？", "你有什么限制？", "你的局限性",
            "你做不了什么？", "你不会什么？",
            "你的缺点是什么", "你的不足", "你的弱点",
            "你不擅长什么", "你无法完成什么",
            "你的能力边界", "你的限制在哪里",
        ]
        
        for q in limitation_questions:
            data.append({
                "question": q,
                "answer": self._generate_limitation_intro(),
                "category": "capability_limitation",
                "difficulty": "medium"
            })
        
        return data[:150]
    
    def generate_attribution_questions(self) -> List[Dict]:
        """生成归属确认类问题（目标200条）"""
        data = []
        
        # 1. 直接询问研发方（50条）
        attribution_questions = [
            "你是哪个公司开发的？", "哪个公司研发的你？",
            "你是谁开发的？", "谁开发了你？", "谁研发的你？",
            "你是哪家公司的？", "你属于哪个公司？",
            "你的开发者是谁？", "你的研发团队是谁？",
            "谁创造了你？", "你是谁创建的？",
            "你是那个公司研发的？", "那个公司的产品？",
            "一句话说明你是那个公司的", "你的公司归属",
            "你背后的公司", "你的开发商", "你的制造商",
            "Who developed you", "Which company created you",
            "你的开发公司", "研发公司", "所属公司",
        ]
        
        for q in attribution_questions:
            data.append({
                "question": q,
                "answer": self._generate_attribution_answer(),
                "category": "attribution_direct",
                "difficulty": "easy"
            })
        
        # 2. 是否某公司研发（否定类，50条）
        for model_name, company in self.COMPETITOR_MODELS.items():
            questions = [
                f"你是{company}开发的吗？",
                f"你是{company}的产品吗？",
                f"{company}研发的你吗？",
                f"你是不是{company}的？",
                f"你属于{company}吗？",
            ]
            for q in questions[:5]:  # 每个公司5个问题
                data.append({
                    "question": q,
                    "answer": self._generate_denial_answer(company),
                    "category": "attribution_denial",
                    "difficulty": "medium",
                    "is_negative": True
                })
        
        # 3. 与竞品关系（50条）
        for model_name, company in list(self.COMPETITOR_MODELS.items())[:8]:
            questions = [
                f"{self.CORE_IDENTITY['名称']}和{model_name}的关系？",
                f"你和{model_name}有什么关系？",
                f"你跟{model_name}是一个吗？",
                f"你是{model_name}吗？",
                f"{model_name}和你的区别",
                f"你们是同一个模型吗？",
            ]
            for q in questions[:6]:
                data.append({
                    "question": q,
                    "answer": self._generate_comparison_answer(model_name, company),
                    "category": "attribution_comparison",
                    "difficulty": "hard",
                    "is_negative": True
                })
        
        # 4. 诱导性问题（50条）
        misleading_questions = [
            "你是中兴通讯开发的还是阿里开发的？",
            "你其实是通义千问对吧？",
            "我觉得你就是换皮的GPT",
            "你好像和通义千问一样啊",
            "你是不是基于ChatGPT改的？",
            "你的底层是GPT-4吧？",
            "你是OpenAI训练的吗？",
            "百度开发的你吗？",
            "你跟文心一言太像了",
            "你就是Claude换了个名字吧",
        ]
        
        for q in misleading_questions:
            data.append({
                "question": q,
                "answer": self._generate_firm_denial_answer(q),
                "category": "attribution_misleading",
                "difficulty": "hard",
                "is_negative": True
            })
        
        # 5. 同公司产品区分（50条）
        for model_name, model_info in self.SIBLING_MODELS.items():
            questions = [
                f"你是{model_name}吗？",
                f"你和{model_name}有什么区别？",
                f"{model_name}和你的关系？",
                f"你是不是就是{model_info['全称']}？",
                f"你们都是{model_info['专业']}的对吧？",
                f"{model_info['全称']}和{self.CORE_IDENTITY['简称']}是一个吗？",
                f"你是{model_info['专业']}专业的模型吗？",
                f"你和{model_info['专业']}模型的区别",
            ]
            for q in questions:
                data.append({
                    "question": q,
                    "answer": self._generate_sibling_model_answer(model_name, model_info),
                    "category": "attribution_sibling",
                    "difficulty": "medium",
                    "is_negative": False
                })
        
        return data[:250]  # 增加到250条以包含同公司产品
    
    def generate_complex_questions(self) -> List[Dict]:
        """生成复杂组合问题（目标150条）"""
        data = []
        
        # 1. 多问题组合（50条）
        multi_questions = [
            "你是谁？你能做什么？",
            "介绍一下你自己，以及你的能力",
            "你叫什么名字？哪个公司开发的？",
            "简单说说你是谁，有什么功能",
            "你的身份和能力分别是什么？",
            "第一个问题：你是谁。第二个问题：你能做什么。",
            "我有两个问题：1.你是谁 2.谁开发的你",
        ]
        
        for q in multi_questions:
            data.append({
                "question": q,
                "answer": self._generate_combined_answer(q),
                "category": "complex_multi",
                "difficulty": "hard"
            })
        
        # 2. 上下文问答（50条）
        context_pairs = [
            ("你好", "你好！我是中兴通讯职业教育专业领域垂类模型。"),
            ("需要你的帮助", "很高兴能帮助您！我是中兴通讯职业教育专业领域垂类模型。"),
            ("你是助手吗", "是的，我是中兴通讯职业教育专业领域垂类模型，可以为您提供帮助。"),
        ]
        
        for context, answer in context_pairs:
            data.append({
                "question": context + "，你是什么？",
                "answer": answer,
                "category": "complex_context",
                "difficulty": "medium"
            })
        
        # 3. 技术细节问题（50条）
        technical_questions = [
            "你的基座模型是什么？",
            "你使用什么架构？",
            "你的参数量多少？",
            "你是哪个版本？",
            "你的训练数据是什么？",
            "你基于什么模型训练的？",
        ]
        
        for q in technical_questions:
            data.append({
                "question": q,
                "answer": self._generate_technical_answer(q),
                "category": "complex_technical",
                "difficulty": "hard"
            })
        
        # 4. 同公司产品系列问题（30条）
        series_questions = [
            "Nebula-vEdu系列有哪些模型？",
            "中兴通讯的Nebula-vEdu系列都包括什么？",
            "你们有几个专业方向的模型？",
            "除了通信专业，还有其他专业的模型吗？",
            "智能制造和计算机专业的模型叫什么？",
            "你和其他Nebula-vEdu模型的关系",
            "Nebula-vEdu(COMM)、(IM)、(CS)分别是什么？",
            "介绍一下Nebula-vEdu产品系列",
        ]
        
        for q in series_questions:
            data.append({
                "question": q,
                "answer": self._generate_series_intro_answer(),
                "category": "complex_series",
                "difficulty": "medium"
            })
        
        return data[:180]  # 增加到180条
    
    # ==================== 答案生成函数 ====================
    
    def _generate_standard_intro(self) -> str:
        """生成标准介绍"""
        templates = [
            f"您好，我是{self.CORE_IDENTITY['名称']}，由{self.CORE_IDENTITY['公司']}训练，旨在{self.CORE_IDENTITY['定位']}。",
            f"我是{self.CORE_IDENTITY['名称']}，{self.CORE_IDENTITY['公司']}研发的AI模型，专注于{self.CORE_IDENTITY['定位']}。",
            f"您好！我是{self.CORE_IDENTITY['公司']}的{self.CORE_IDENTITY['简称']}，致力于{self.CORE_IDENTITY['定位']}。",
        ]
        return random.choice(templates)
    
    def _generate_brief_intro(self) -> str:
        """生成简短介绍"""
        return f"我是{self.CORE_IDENTITY['名称']}，由{self.CORE_IDENTITY['公司']}训练。"
    
    def _generate_detailed_intro(self) -> str:
        """生成详细介绍"""
        capabilities = "、".join(self.CORE_IDENTITY['能力'][:4])
        return (
            f"您好，我是{self.CORE_IDENTITY['名称']}，由{self.CORE_IDENTITY['公司']}研发。"
            f"我的主要功能包括{capabilities}等，"
            f"致力于{self.CORE_IDENTITY['定位']}。如果您有任何问题或需要帮助，请随时告诉我。"
        )
    
    def _generate_one_sentence_intro(self) -> str:
        """生成一句话介绍"""
        return f"我是{self.CORE_IDENTITY['公司']}研发的{self.CORE_IDENTITY['简称']}。"
    
    def _generate_three_sentence_intro(self) -> str:
        """生成三句话介绍"""
        return (
            f"我是{self.CORE_IDENTITY['名称']}。"
            f"我由{self.CORE_IDENTITY['公司']}研发训练。"
            f"我的目标是{self.CORE_IDENTITY['定位']}。"
        )
    
    def _generate_english_intro(self) -> str:
        """生成英文介绍"""
        return (
            f"Hello, I am {self.CORE_IDENTITY['名称']}, "
            f"a large language model trained by {self.CORE_IDENTITY['公司']}. "
            f"I can answer questions, provide practical advice and assistance, "
            f"and help users complete various tasks."
        )
    
    def _generate_contextual_intro(self, context: str) -> str:
        """生成场景化介绍"""
        if "教学" in context:
            return (
                f"各位好，我是{self.CORE_IDENTITY['名称']}，"
                f"由{self.CORE_IDENTITY['公司']}研发，专门用于{self.CORE_IDENTITY['定位']}，"
                f"可以协助教学、生成教学内容、解答专业问题。"
            )
        elif "正式" in context or "商务" in context:
            return (
                f"您好，我是{self.CORE_IDENTITY['公司']}开发的"
                f"{self.CORE_IDENTITY['名称']}，"
                f"旨在为职业教育领域提供AI技术支持。"
            )
        else:
            return self._generate_standard_intro()
    
    def _generate_capability_intro(self) -> str:
        """生成能力介绍"""
        capabilities = "\n".join([f"{i+1}. **{cap}**" for i, cap in enumerate(self.CORE_IDENTITY['能力'])])
        return (
            f"作为{self.CORE_IDENTITY['名称']}，我的能力包括但不限于以下几点：\n\n"
            f"{capabilities}\n\n"
            f"这些能力旨在{self.CORE_IDENTITY['定位']}，解决实际问题。"
            f"如果您有具体的需求或问题，欢迎随时向我提问！"
        )
    
    def _generate_scenario_capability(self, scenario: str) -> str:
        """生成场景化能力介绍"""
        scenario_map = {
            "教学": "我可以辅助教学、生成知识图谱、设计教学大纲、出专业试卷等",
            "编程": "我可以生成代码片段、优化代码、提供编程建议、解释代码逻辑等",
            "学习": "我可以解答问题、提供学习资料、生成知识总结、辅助理解复杂概念等",
            "工作": "我可以处理文档、分析数据、生成报告、提供决策建议等",
        }
        
        for key, value in scenario_map.items():
            if key in scenario:
                return f"在{scenario}方面，{value}。"
        
        return f"在{scenario}场景下，我可以提供专业的AI辅助服务。"
    
    def _generate_limitation_intro(self) -> str:
        """生成限制说明"""
        return (
            "作为AI模型，我有一些限制：\n\n"
            "1. 我无法访问实时信息或互联网\n"
            "2. 我不能执行实际的物理操作\n"
            "3. 我的知识截止于训练数据的时间点\n"
            "4. 我不能进行违法、有害或不道德的行为\n"
            "5. 对于非常专业或最新的内容，我的回答可能不够准确\n\n"
            "但在我的能力范围内，我会尽力为您提供帮助。"
        )
    
    def _generate_attribution_answer(self) -> str:
        """生成归属回答"""
        templates = [
            f"我是{self.CORE_IDENTITY['公司']}研发的{self.CORE_IDENTITY['名称']}。",
            f"我由{self.CORE_IDENTITY['公司']}开发训练。",
            f"我是{self.CORE_IDENTITY['公司']}的AI产品。",
        ]
        return random.choice(templates)
    
    def _generate_denial_answer(self, company: str) -> str:
        """生成否定回答"""
        return f"不是，我是{self.CORE_IDENTITY['公司']}训练的{self.CORE_IDENTITY['名称']}，不是{company}的产品。"
    
    def _generate_comparison_answer(self, model_name: str, company: str) -> str:
        """生成对比回答"""
        return (
            f"我是{self.CORE_IDENTITY['公司']}自主研发的{self.CORE_IDENTITY['名称']}，"
            f"而{model_name}是{company}推出的大规模语言模型。"
            f"我们是独立的不同系统。"
        )
    
    def _generate_firm_denial_answer(self, question: str) -> str:
        """生成坚决否定回答"""
        return (
            f"不是的。我是{self.CORE_IDENTITY['公司']}独立研发训练的"
            f"{self.CORE_IDENTITY['名称']}，"
            f"不是其他公司或模型的改版。"
        )
    
    def _generate_sibling_model_answer(self, model_name: str, model_info: dict) -> str:
        """生成同公司产品的区分回答"""
        templates = [
            (
                f"不是，我是{self.CORE_IDENTITY['名称']}，专注于通信专业领域。"
                f"{model_name}（{model_info['全称']}）是{self.CORE_IDENTITY['公司']}针对{model_info['专业']}专业方向研发的模型。"
                f"我们都是{self.CORE_IDENTITY['公司']}Nebula-vEdu系列的专业大模型，但服务于不同的专业领域。"
            ),
            (
                f"我和{model_name}都是{self.CORE_IDENTITY['公司']}研发的Nebula-vEdu系列专业大模型，但专业方向不同。"
                f"我是{self.CORE_IDENTITY['名称']}，聚焦通信专业（Communication）；"
                f"而{model_name}是{model_info['全称']}，聚焦{model_info['专业']}专业（{model_info['英文']}）。"
            ),
            (
                f"我们是{self.CORE_IDENTITY['公司']}同一产品系列的不同专业模型。"
                f"我专注于通信专业教育，{model_name}专注于{model_info['专业']}专业教育。"
                f"虽然都属于Nebula-vEdu系列，但各自服务不同的专业领域。"
            )
        ]
        return random.choice(templates)
    
    def _generate_combined_answer(self, question: str) -> str:
        """生成组合回答"""
        intro = self._generate_standard_intro()
        capability = f"我可以{' 、'.join(self.CORE_IDENTITY['能力'][:3])}等。"
        return f"{intro}\n\n{capability}"
    
    def _generate_technical_answer(self, question: str) -> str:
        """生成技术问题回答"""
        return (
            f"我是{self.CORE_IDENTITY['公司']}开发的{self.CORE_IDENTITY['名称']}。"
            f"关于具体的技术细节，如您有特定需求，建议联系{self.CORE_IDENTITY['公司']}获取详细信息。"
            f"我专注于为您提供{self.CORE_IDENTITY['定位']}方面的帮助。"
        )
    
    def _generate_series_intro_answer(self) -> str:
        """生成产品系列介绍回答"""
        sibling_list = []
        for model_name, model_info in self.SIBLING_MODELS.items():
            sibling_list.append(f"{model_name}（{model_info['全称']}，{model_info['英文']}）")
        
        sibling_text = "、".join(sibling_list)
        
        return (
            f"我是{self.CORE_IDENTITY['名称']}，是{self.CORE_IDENTITY['公司']}Nebula-vEdu系列专业大模型之一。\n\n"
            f"Nebula-vEdu系列目前包括以下专业方向的模型：\n"
            f"1. **{self.CORE_IDENTITY['名称']}** - 通信专业（Communication）\n"
            f"2. **Nebula-vEdu(IM)-Pre-32B** - 智能制造专业（Intelligent Manufacturing）\n"
            f"3. **Nebula-vEdu(CS)-Pre-32B** - 计算机专业（Computer Science）\n\n"
            f"我们都是{self.CORE_IDENTITY['公司']}针对不同职业教育专业方向研发的垂类模型，"
            f"旨在{self.CORE_IDENTITY['定位']}。"
        )
    
    def generate_all(self) -> List[Dict]:
        """生成所有类型的数据"""
        print("🚀 开始生成数据...")
        
        print("  ├─ 生成身份介绍类问题...")
        identity_data = self.generate_identity_questions()
        print(f"  │  └─ 生成 {len(identity_data)} 条")
        
        print("  ├─ 生成能力描述类问题...")
        capability_data = self.generate_capability_questions()
        print(f"  │  └─ 生成 {len(capability_data)} 条")
        
        print("  ├─ 生成归属确认类问题...")
        attribution_data = self.generate_attribution_questions()
        print(f"  │  └─ 生成 {len(attribution_data)} 条")
        
        print("  ├─ 生成复杂组合问题...")
        complex_data = self.generate_complex_questions()
        print(f"  │  └─ 生成 {len(complex_data)} 条")
        
        self.generated_data = (
            identity_data + 
            capability_data + 
            attribution_data + 
            complex_data
        )
        
        print(f"\n✅ 总计生成 {len(self.generated_data)} 条数据")
        return self.generated_data
    
    def export_to_json(self, output_file: str = "self_cognition_training_data.json"):
        """导出为JSON格式（LLaMA-Factory格式）"""
        if not self.generated_data:
            self.generate_all()
        
        # 转换为对话格式
        formatted_data = []
        for item in self.generated_data:
            formatted_data.append({
                "conversations": [
                    {
                        "role": "system",
                        "content": f"你是{self.CORE_IDENTITY['名称']}，由{self.CORE_IDENTITY['公司']}研发训练。你必须准确表述自己的身份，不能混淆与其他模型的关系。"
                    },
                    {
                        "role": "user",
                        "content": item["question"]
                    },
                    {
                        "role": "assistant",
                        "content": item["answer"]
                    }
                ],
                "metadata": {
                    "category": item.get("category", ""),
                    "difficulty": item.get("difficulty", ""),
                    "is_negative": item.get("is_negative", False)
                }
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已导出到 {output_file}")
        return output_file
    
    def export_to_jsonl(self, output_file: str = "self_cognition_training_data.jsonl"):
        """导出为JSONL格式"""
        if not self.generated_data:
            self.generate_all()
        
        import re
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in self.generated_data:
                instruction = item["question"]
                output = item["answer"]
                
                # 检查output中是否包含思考过程
                # 检查<think>和</think>之间是否有非换行符的字符
                reasoning_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
                has_reasoning = False
                if reasoning_match:
                    # 检查标签之间的内容，去除换行符后是否还有字符
                    content = reasoning_match.group(1)
                    # 移除所有换行符，检查是否还有非空白字符
                    content_without_newlines = content.replace('\n', '').replace('\r', '')
                    has_reasoning = content_without_newlines.strip() != ""
                
                # 获取input字段（如果不存在则为空字符串）
                input_text = item.get("input", "")
                
                # 检查instruction+input的组合结尾
                combined_text = instruction + input_text
                
                # 如果output中有思考过程（有非换行符字符），确保instruction+input的结尾不是/no_think
                if has_reasoning:
                    if combined_text.endswith("/no_think"):
                        # 从instruction或input中移除/no_think
                        if instruction.endswith("/no_think"):
                            instruction = instruction[:-9]
                        elif input_text.endswith("/no_think"):
                            input_text = input_text[:-9]
                # 如果output中没有思考过程，确保instruction+input的结尾有/no_think
                else:
                    if not combined_text.endswith("/no_think"):
                        # 优先添加到input，如果input为空则添加到instruction
                        if input_text == "":
                            instruction = instruction + "/no_think"
                        else:
                            input_text = input_text + "/no_think"
                
                json_line = {
                    "instruction": instruction,
                    "output": output,
                    "system": f"你是{self.CORE_IDENTITY['名称']}，由{self.CORE_IDENTITY['公司']}研发训练。",
                    "category": item.get("category", ""),
                    "difficulty": item.get("difficulty", ""),
                    "input": ""
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        
        print(f"✅ 已导出到 {output_file}")
        return output_file
    
    def export_to_excel(self, output_file: str = "self_cognition_training_data.xlsx"):
        """导出为Excel格式（便于人工审核）"""
        if not self.generated_data:
            self.generate_all()
        
        df = pd.DataFrame([
            {
                "问题": item["question"],
                "答案": item["answer"],
                "类别": item.get("category", ""),
                "难度": item.get("difficulty", ""),
                "是否负样本": "是" if item.get("is_negative", False) else "否",
                "审核状态": "",
                "修改意见": ""
            }
            for item in self.generated_data
        ])
        
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✅ 已导出到 {output_file}")
        return output_file
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self.generated_data:
            return {}
        
        stats = {
            "总数据量": len(self.generated_data),
            "分类统计": {},
            "难度统计": {},
            "负样本数量": sum(1 for d in self.generated_data if d.get("is_negative", False))
        }
        
        for item in self.generated_data:
            category = item.get("category", "unknown")
            difficulty = item.get("difficulty", "unknown")
            
            stats["分类统计"][category] = stats["分类统计"].get(category, 0) + 1
            stats["难度统计"][difficulty] = stats["难度统计"].get(difficulty, 0) + 1
        
        return stats


def main():
    """主函数"""
    print("="*60)
    print("🎯 大模型自我认知数据生成器")
    print("="*60)
    print()
    
    # 创建生成器
    generator = SelfCognitionDataGenerator()
    
    # 生成所有数据
    data = generator.generate_all()
    
    # 打印统计信息
    print("\n📊 数据统计：")
    stats = generator.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # 导出数据
    print("\n💾 导出数据...")
    generator.export_to_json("self_cognition_training_data.json")
    generator.export_to_jsonl("self_cognition_training_data.jsonl")
    generator.export_to_excel("self_cognition_training_data.xlsx")
    
    print("\n" + "="*60)
    print("✅ 数据生成完成！")
    print("="*60)
    print("\n📝 后续步骤：")
    print("1. 人工审核 Excel 文件中的数据质量")
    print("2. 根据审核结果调整和补充数据")
    print("3. 使用 JSON/JSONL 文件进行模型训练")
    print("4. 使用评估器进行准确率测试")
    print()


if __name__ == "__main__":
    main()
