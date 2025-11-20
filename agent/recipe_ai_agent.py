#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于OpenAI API的智能菜谱解析AI Agent
"""

import os
import json
import re
import time
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import csv
from datetime import datetime

@dataclass
class IngredientInfo:
    """食材信息"""
    name: str
    amount: str = ""
    unit: str = ""
    category: str = ""
    is_main: bool = True  # 是否主要食材
    
@dataclass
class CookingStep:
    """烹饪步骤"""
    step_number: int
    description: str
    methods: List[str]  # 使用的烹饪方法
    tools: List[str]    # 需要的工具
    time_estimate: str = ""  # 时间估计
    
@dataclass
class RecipeInfo:
    """菜谱信息"""
    name: str
    difficulty: int  # 1-5星
    category: str
    cuisine_type: str = ""  # 菜系
    prep_time: str = ""
    cook_time: str = ""
    servings: str = ""
    ingredients: List[IngredientInfo] = None
    steps: List[CookingStep] = None
    tags: List[str] = None
    nutrition_info: Dict = None
    
    def __post_init__(self):
        if self.ingredients is None:
            self.ingredients = []
        if self.steps is None:
            self.steps = []
        if self.tags is None:
            self.tags = []
        if self.nutrition_info is None:
            self.nutrition_info = {}

class LLMRecipeAgent:
    """OpenAI菜谱解析AI Agent"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 目录名到分类的映射
        self.directory_category_mapping = {
            "vegetable_dish": "素菜",
            "meat_dish": "荤菜", 
            "aquatic": "水产",
            "breakfast": "早餐",
            "staple": "主食",
            "soup": "汤类",
            "dessert": "甜品",
            "drink": "饮料",
            "condiment": "调料",
            "semi-finished": "半成品"
        }
        
        # 排除的目录
        self.excluded_directories = ["template", ".github", "tips", "starsystem"]
        
        # 预定义的食材分类
        self.ingredient_categories = {
            "蔬菜": ["茄子", "辣椒", "洋葱", "大葱", "西红柿", "土豆", "萝卜", "白菜", "豆腐"],
            "调料": ["盐", "酱油", "醋", "糖", "料酒", "生抽", "老抽", "蚝油", "味精"],
            "蛋白质": ["鸡蛋", "肉", "鱼", "虾", "鸡", "猪", "牛", "羊"],
            "淀粉类": ["面粉", "淀粉", "米", "面条", "面包", "土豆"]
        }
        
        # 预定义的烹饪方法
        self.cooking_methods = ["炒", "炸", "煮", "蒸", "烤", "炖", "焖", "煎", "红烧", "清炒", "爆炒"]
        
        # 预定义的工具
        self.cooking_tools = ["炒锅", "平底锅", "蒸锅", "刀", "案板", "筷子", "锅铲", "勺子"]
    
    def call_openai_api(self, messages: List[Dict], max_retries: int = 3) -> str:
        """调用OpenAI API"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="kimi-k2-0711-preview",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    stream=False
                )
                
                return response.choices[0].message.content
                    
            except Exception as e:
                print(f"API调用错误 (尝试 {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    
        raise Exception("OpenAI API调用失败")
    
    def infer_category_from_path(self, file_path: str) -> str:
        """根据文件路径推断菜谱分类"""
        path_parts = file_path.replace('\\', '/').split('/')
        
        for part in path_parts:
            if part in self.directory_category_mapping:
                return self.directory_category_mapping[part]
        
        return ""  # 如果无法推断，返回空字符串
    
    def extract_recipe_info(self, markdown_content: str, file_path: str = "") -> RecipeInfo:
        """使用AI提取菜谱信息"""
        
        # 根据路径推断分类
        inferred_category = self.infer_category_from_path(file_path)
        
        # 构建提示词
        category_hint = f"，根据文件路径推断此菜谱属于【{inferred_category}】分类" if inferred_category else ""
        
        prompt = f"""
请分析以下标准化格式的菜谱Markdown文档，提取结构化信息并以JSON格式返回。

文件路径: {file_path}
菜谱内容：
{markdown_content}

## 文档结构说明
此菜谱遵循标准格式，包含以下固定二级标题：
- ## 必备原料和工具：列出所有食材和工具
- ## 计算：包含份量计算和具体用量
- ## 操作：详细的烹饪步骤
- ## 附加内容：补充说明和技巧提示（需要过滤无关内容）

## 提取规则
1. **菜谱名称**：从一级标题（# XXX的做法）提取
2. **难度等级**：从"预估烹饪难度：★★★"中统计★的数量
3. **菜谱分类**：可以是多个分类，用逗号分隔（如"早餐,素菜"表示既是早餐又是素菜）
4. **食材信息**：从"必备原料和工具"和"计算"部分提取，合并用量信息
5. **烹饪步骤**：从"操作"部分的有序列表提取
6. **技巧补充**：从"附加内容"提取有用的烹饪技巧，忽略模板文字（如"如果您遵循本指南...Issue或Pull request"等）

请返回标准JSON格式{category_hint}：
{{
    "name": "菜谱名称（去掉'的做法'后缀）",
    "difficulty": 1-5的数字（根据★数量：★=1, ★★=2, ★★★=3, ★★★★=4, ★★★★★=5），
    "category": "{inferred_category if inferred_category else '菜谱分类（素菜/荤菜/水产/早餐/主食/汤类/甜品/饮料/调料，支持多个分类用逗号分隔，如\"早餐,素菜\"）'}",
    "cuisine_type": "菜系（川菜/粤菜/鲁菜/苏菜/闽菜/浙菜/湘菜/徽菜/东北菜/西北菜/等，如果不明确则为空）",
    "prep_time": "准备时间（从腌制、切菜等步骤推断）",
    "cook_time": "烹饪时间（从炒制、炖煮等步骤推断）", 
    "servings": "份数/人数（从'计算'部分提取，如'2个人食用'）",
    "ingredients": [
        {{
            "name": "食材名称",
            "amount": "用量数字（从计算部分提取具体数值）",
            "unit": "单位（克、个、毫升、片等）",
            "category": "食材类别（蔬菜/调料/蛋白质/淀粉类/其他）",
            "is_main": true/false（主要食材为true，调料为false）
        }}
    ],
    "steps": [
        {{
            "step_number": 1,
            "description": "步骤详细描述",
            "methods": ["使用的烹饪方法：炒、炸、煮、蒸、烤、炖、焖、煎、红烧、腌制、切等"],
            "tools": ["需要的工具：炒锅、平底锅、蒸锅、刀、案板、筷子、锅铲、盆等"],
            "time_estimate": "时间估计（如步骤中提到'15秒'、'30秒'、'10-15分钟'等）"
        }}
    ],
    "tags": ["从附加内容中提取的有用技巧标签"],
    "nutrition_info": {{
        "calories": "",
        "protein": "", 
        "carbs": "",
        "fat": ""
    }}
}}

## 重要提示：
1. 从"计算"部分精确提取食材用量和单位
2. 从"操作"部分的有序列表逐步解析烹饪步骤
3. 从"附加内容"中只提取烹饪技巧，忽略"Issue或Pull request"等模板文字
4. 食材分类要准确：蔬菜（包括各种菜类）、调料（盐、酱油、糖等）、蛋白质（鱼、肉、蛋）、淀粉类（面粉、米等）
5. 菜谱分类支持多重分类：如早餐类的蔬菜粥可以分类为"早餐,素菜,主食"（逗号分隔）
6. 当遇到"适量"、"少许"等非具体数值时，不要忘记加引号，如"amount": "适量"
7. 只返回标准JSON格式，确保语法正确
"""

        messages = [
            {"role": "system", "content": "你是一个专业的菜谱分析专家，擅长从中文菜谱中提取结构化信息。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.call_openai_api(messages)
            
            # 清理响应，确保是有效的JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # 解析JSON
            recipe_data = json.loads(response)
            
            # 转换为RecipeInfo对象
            recipe_info = RecipeInfo(
                name=recipe_data.get("name", ""),
                difficulty=recipe_data.get("difficulty", 3),
                category=recipe_data.get("category", ""),
                cuisine_type=recipe_data.get("cuisine_type", ""),
                prep_time=recipe_data.get("prep_time", ""),
                cook_time=recipe_data.get("cook_time", ""),
                servings=recipe_data.get("servings", ""),
                nutrition_info=recipe_data.get("nutrition_info", {})
            )
            
            # 转换食材信息
            for ing_data in recipe_data.get("ingredients", []):
                ingredient = IngredientInfo(
                    name=ing_data.get("name", ""),
                    amount=ing_data.get("amount", ""),
                    unit=ing_data.get("unit", ""),
                    category=ing_data.get("category", ""),
                    is_main=ing_data.get("is_main", True)
                )
                recipe_info.ingredients.append(ingredient)
            
            # 转换步骤信息
            for step_data in recipe_data.get("steps", []):
                step = CookingStep(
                    step_number=step_data.get("step_number", 0),
                    description=step_data.get("description", ""),
                    methods=step_data.get("methods", []),
                    tools=step_data.get("tools", []),
                    time_estimate=step_data.get("time_estimate", "")
                )
                recipe_info.steps.append(step)
            
            # 添加标签
            recipe_info.tags = recipe_data.get("tags", [])
            
            return recipe_info
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {response}")
            return self._fallback_parse(markdown_content)
        except Exception as e:
            print(f"AI解析错误: {e}")
            return self._fallback_parse(markdown_content)
    
    def _fallback_parse(self, content: str) -> RecipeInfo:
        """备用解析方法（基于规则）"""
        lines = content.strip().split('\n')
        
        # 提取菜谱名称
        name = ""
        for line in lines:
            if line.startswith('# '):
                name = line[2:].replace('的做法', '').strip()
                break
        
        # 提取难度
        difficulty = 3  # 默认3星
        for line in lines:
            if '★' in line:
                stars = line.count('★')
                difficulty = min(max(stars, 1), 5)
                break
        
        # 简单分类判断
        category = "其他"
        if any(keyword in name for keyword in ["蛋", "豆腐"]):
            category = "素菜"
        elif any(keyword in name for keyword in ["肉", "鸡", "鱼", "虾"]):
            category = "荤菜"
        
        return RecipeInfo(
            name=name or "未知菜谱",
            difficulty=difficulty,
            category=category
        )

class RecipeKnowledgeGraphBuilder:
    """菜谱知识图谱构建器 - 支持分批保存和断点续传"""
    
    def __init__(self, ai_agent: LLMRecipeAgent, output_dir: str = "./ai_output", batch_size: int = 20):
        self.ai_agent = ai_agent
        self.concepts = []
        self.relationships = []
        self.concept_id_counter = 201000000
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.progress_file = os.path.join(output_dir, "progress.json")
        self.processed_files = set()
        self.current_batch = 0
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化预定义概念和关系类型映射
        self._init_predefined_concepts()
        self._init_relationship_mappings()
    
    def _init_relationship_mappings(self):
        """初始化关系类型映射"""
        self.relationship_type_mapping = {
            "has_ingredient": "801000001",
            "requires_tool": "801000002", 
            "has_step": "801000003",
            "belongs_to_category": "801000004",
            "has_difficulty": "801000005",
            "uses_method": "801000006",
            "has_amount": "801000007",
            "step_follows": "801000008",
            "serves_people": "801000009",
            "cooking_time": "801000010",
            "prep_time": "801000011"
        }
    
    def _init_predefined_concepts(self):
        """初始化预定义概念"""
        self.predefined_concepts = [
            # 根概念
            {
                "concept_id": "100000000",
                "concept_type": "Root",
                "name": "烹饪概念",
                "fsn": "烹饪概念 (Culinary Concept)",
                "preferred_term": "烹饪概念"
            },
            
            # 顶级概念
            {
                "concept_id": "200000000",
                "concept_type": "Recipe", 
                "name": "菜谱",
                "fsn": "菜谱 (Recipe)",
                "preferred_term": "菜谱"
            },
            {
                "concept_id": "300000000",
                "concept_type": "Ingredient",
                "name": "食材", 
                "fsn": "食材 (Ingredient)",
                "preferred_term": "食材"
            },
            {
                "concept_id": "400000000",
                "concept_type": "CookingMethod",
                "name": "烹饪方法",
                "fsn": "烹饪方法 (Cooking Method)", 
                "preferred_term": "烹饪方法"
            },
            {
                "concept_id": "500000000",
                "concept_type": "CookingTool",
                "name": "烹饪工具",
                "fsn": "烹饪工具 (Cooking Tool)",
                "preferred_term": "烹饪工具"
            },
            
            # 难度等级
            {
                "concept_id": "610000000",
                "concept_type": "DifficultyLevel",
                "name": "一星",
                "fsn": "一星 (One Star)",
                "preferred_term": "一星"
            },
            {
                "concept_id": "620000000", 
                "concept_type": "DifficultyLevel",
                "name": "二星",
                "fsn": "二星 (Two Star)",
                "preferred_term": "二星"
            },
            {
                "concept_id": "630000000", 
                "concept_type": "DifficultyLevel",
                "name": "三星",
                "fsn": "三星 (Three Star)",
                "preferred_term": "三星"
            },
            {
                "concept_id": "640000000", 
                "concept_type": "DifficultyLevel",
                "name": "四星",
                "fsn": "四星 (Four Star)",
                "preferred_term": "四星"
            },
            {
                "concept_id": "650000000", 
                "concept_type": "DifficultyLevel",
                "name": "五星",
                "fsn": "五星 (Five Star)",
                "preferred_term": "五星"
            },
            
            # 菜谱分类
            {
                "concept_id": "710000000",
                "concept_type": "RecipeCategory",
                "name": "素菜", 
                "fsn": "素菜 (Vegetarian Dish)",
                "preferred_term": "素菜"
            },
            {
                "concept_id": "720000000",
                "concept_type": "RecipeCategory",
                "name": "荤菜", 
                "fsn": "荤菜 (Meat Dish)",
                "preferred_term": "荤菜"
            },
            {
                "concept_id": "730000000",
                "concept_type": "RecipeCategory",
                "name": "水产", 
                "fsn": "水产 (Aquatic Product)",
                "preferred_term": "水产"
            },
            {
                "concept_id": "740000000",
                "concept_type": "RecipeCategory",
                "name": "早餐", 
                "fsn": "早餐 (Breakfast)",
                "preferred_term": "早餐"
            },
            {
                "concept_id": "750000000",
                "concept_type": "RecipeCategory",
                "name": "主食", 
                "fsn": "主食 (Staple Food)",
                "preferred_term": "主食"
            },
            {
                "concept_id": "760000000",
                "concept_type": "RecipeCategory",
                "name": "汤类", 
                "fsn": "汤类 (Soup)",
                "preferred_term": "汤类"
            },
            {
                "concept_id": "770000000",
                "concept_type": "RecipeCategory",
                "name": "甜品", 
                "fsn": "甜品 (Dessert)",
                "preferred_term": "甜品"
            },
            {
                "concept_id": "780000000",
                "concept_type": "RecipeCategory",
                "name": "饮料", 
                "fsn": "饮料 (Beverage)",
                "preferred_term": "饮料"
            },
            {
                "concept_id": "790000000",
                "concept_type": "RecipeCategory",
                "name": "调料", 
                "fsn": "调料 (Condiment)",
                "preferred_term": "调料"
            }
        ]
    
    def save_progress(self, current_file: str = None, total_files: int = 0, processed_count: int = 0):
        """保存处理进度"""
        progress_data = {
            "processed_files": list(self.processed_files),
            "current_file": current_file,
            "total_files": total_files,
            "processed_count": processed_count,
            "current_batch": self.current_batch,
            "concept_id_counter": self.concept_id_counter,
            "timestamp": datetime.now().isoformat(),
            "concepts_count": len(self.concepts),
            "relationships_count": len(self.relationships)
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    def load_progress(self) -> Dict:
        """加载处理进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                
                self.processed_files = set(progress_data.get("processed_files", []))
                self.current_batch = progress_data.get("current_batch", 0)
                self.concept_id_counter = progress_data.get("concept_id_counter", 201000000)
                
                return progress_data
            except Exception as e:
                print(f"警告: 加载进度文件失败 - {str(e)}")
                return {}
        return {}
    
    def save_batch_data(self, batch_num: int = None):
        """保存当前批次数据"""
        if batch_num is None:
            batch_num = self.current_batch
            
        batch_output_dir = os.path.join(self.output_dir, f"batch_{batch_num:03d}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # 保存概念数据
        if self.concepts:
            concepts_df = pd.DataFrame(self.concepts)
            concepts_file = os.path.join(batch_output_dir, "concepts.csv")
            concepts_df.to_csv(concepts_file, index=False, encoding='utf-8')
        
        # 保存关系数据
        if self.relationships:
            relationships_df = pd.DataFrame(self.relationships)
            relationships_file = os.path.join(batch_output_dir, "relationships.csv")
            relationships_df.to_csv(relationships_file, index=False, encoding='utf-8')
        
        print(f"批次 {batch_num} 已保存")
        
        return batch_output_dir
    
    def merge_all_batches(self):
        """合并所有批次数据到最终输出文件"""
        print("合并批次数据...")
        
        all_concepts = []
        all_relationships = []
        
        # 收集所有批次数据
        batch_dirs = [d for d in os.listdir(self.output_dir) 
                     if d.startswith("batch_") and os.path.isdir(os.path.join(self.output_dir, d))]
        batch_dirs.sort()
        
        for batch_dir in batch_dirs:
            batch_path = os.path.join(self.output_dir, batch_dir)
            
            # 读取概念文件
            concepts_file = os.path.join(batch_path, "concepts.csv")
            if os.path.exists(concepts_file):
                batch_concepts = pd.read_csv(concepts_file)
                all_concepts.append(batch_concepts)
            
            # 读取关系文件
            relationships_file = os.path.join(batch_path, "relationships.csv")
            if os.path.exists(relationships_file):
                batch_relationships = pd.read_csv(relationships_file)
                all_relationships.append(batch_relationships)
        
        # 合并数据
        if all_concepts:
            final_concepts = pd.concat(all_concepts, ignore_index=True)
            final_concepts.to_csv(os.path.join(self.output_dir, "concepts.csv"), 
                                index=False, encoding='utf-8')
            print(f"合并概念: {len(final_concepts)} 个")
        
        if all_relationships:
            final_relationships = pd.concat(all_relationships, ignore_index=True)
            final_relationships.to_csv(os.path.join(self.output_dir, "relationships.csv"), 
                                     index=False, encoding='utf-8')
            print(f"合并关系: {len(final_relationships)} 个")
        
        return len(final_concepts) if all_concepts else 0, len(final_relationships) if all_relationships else 0
    
    def generate_concept_id(self) -> str:
        """生成新的概念ID"""
        self.concept_id_counter += 1
        return str(self.concept_id_counter)
    
    def process_recipe(self, markdown_content: str, file_path: str) -> Dict:
        """处理单个菜谱"""
        # 处理菜谱
        
        # 使用AI提取菜谱信息
        recipe_info = self.ai_agent.extract_recipe_info(markdown_content, file_path)
        
        # 生成概念ID
        recipe_id = self.generate_concept_id()
        
        # 创建菜谱概念
        recipe_concept = {
            "concept_id": recipe_id,
            "concept_type": "Recipe",
            "name": recipe_info.name,
            "fsn": f"{recipe_info.name} (Recipe)",
            "preferred_term": recipe_info.name,
            "synonyms": self._generate_recipe_synonyms(recipe_info.name, recipe_info.category),
            "category": recipe_info.category,
            "difficulty": recipe_info.difficulty,
            "cuisine_type": recipe_info.cuisine_type,
            "prep_time": recipe_info.prep_time,
            "cook_time": recipe_info.cook_time,
            "servings": recipe_info.servings,
            "tags": ",".join(recipe_info.tags),
            "file_path": file_path
        }
        
        self.concepts.append(recipe_concept)
        
        # 处理食材
        for ingredient in recipe_info.ingredients:
            ing_id = self.generate_concept_id()
            ing_concept = {
                "concept_id": ing_id,
                "concept_type": "Ingredient",
                "name": ingredient.name,
                "fsn": f"{ingredient.name} (Ingredient)",
                "preferred_term": ingredient.name,
                "synonyms": self._generate_ingredient_synonyms(ingredient.name),
                "category": ingredient.category,
                "amount": ingredient.amount,
                "unit": ingredient.unit,
                "is_main": ingredient.is_main
            }
            self.concepts.append(ing_concept)
            
            # 添加关系：菜谱包含食材
            self.relationships.append({
                "relationship_id": f"R_{len(self.relationships) + 1:06d}",
                "source_id": recipe_id,
                "target_id": ing_id,
                "relationship_type": self.relationship_type_mapping["has_ingredient"],
                "amount": ingredient.amount,
                "unit": ingredient.unit
            })
        
        # 处理步骤
        for step in recipe_info.steps:
            step_id = self.generate_concept_id()
            step_concept = {
                "concept_id": step_id,
                "concept_type": "CookingStep",
                "name": f"步骤{step.step_number}",
                "fsn": f"步骤{step.step_number} (Cooking Step)",
                "preferred_term": f"步骤{step.step_number}",
                "description": step.description,
                "step_number": step.step_number,
                "methods": ",".join(step.methods),
                "tools": ",".join(step.tools),
                "time_estimate": step.time_estimate
            }
            self.concepts.append(step_concept)
            
            # 添加关系：菜谱包含步骤
            self.relationships.append({
                "relationship_id": f"R_{len(self.relationships) + 1:06d}",
                "source_id": recipe_id,
                "target_id": step_id,
                "relationship_type": self.relationship_type_mapping["has_step"],
                "step_order": step.step_number
            })
        
        # 添加分类关系 - 支持多重分类
        category_mapping = {
            "素菜": "710000000",
            "荤菜": "720000000", 
            "水产": "730000000",
            "早餐": "740000000",
            "主食": "750000000",
            "汤类": "760000000",
            "甜品": "770000000",
            "饮料": "780000000",
            "调料": "790000000"
        }
        
        # 处理多重分类（支持逗号分隔的多个分类）
        categories = [cat.strip() for cat in recipe_info.category.split(',') if cat.strip()]
        
        for category in categories:
            if category in category_mapping:
                self.relationships.append({
                    "relationship_id": f"R_{len(self.relationships) + 1:06d}",
                    "source_id": recipe_id,
                    "target_id": category_mapping[category],
                    "relationship_type": self.relationship_type_mapping["belongs_to_category"]
                })
        
        # 添加难度关系
        difficulty_mapping = {
            1: "610000000",  # 一星
            2: "620000000",  # 二星
            3: "630000000",  # 三星
            4: "640000000",  # 四星
            5: "650000000"   # 五星
        }
        
        if recipe_info.difficulty in difficulty_mapping:
            self.relationships.append({
                "relationship_id": f"R_{len(self.relationships) + 1:06d}",
                "source_id": recipe_id,
                "target_id": difficulty_mapping[recipe_info.difficulty],
                "relationship_type": self.relationship_type_mapping["has_difficulty"]
            })
        
        return recipe_concept
    
    def _generate_recipe_synonyms(self, name: str, category: str) -> List[str]:
        """生成菜谱的同义词列表"""
        synonyms = []
        
        # 基于菜谱名称生成变体
        if name.endswith("的做法"):
            base_name = name.replace("的做法", "")
            synonyms.extend([
                f"{base_name}制作方法",
                f"{base_name}烹饪方法",
                base_name
            ])
        
        # 基于烹饪方法生成别名（注意：只有真正的同义词才映射）
        cooking_method_mappings = {
            "红烧": ["braised"],                    # 红烧 = 英文braised
            "糖醋": ["sweet and sour"],            # 糖醋 = 英文sweet and sour  
            "清炒": ["炒制", "stir-fried"],        # 清炒 = 炒制 = 英文stir-fried
            "蒸": ["清蒸", "steamed"],              # 蒸 = 清蒸 = 英文steamed
            "炖": ["煲", "stewed"],                # 炖 = 煲 = 英文stewed
            "烤": ["烘烤", "roasted", "baked"],    # 烤 = 烘烤 = 英文roasted/baked
            "炸": ["油炸", "deep-fried"],          # 炸 = 油炸 = 英文deep-fried
            "焖": ["闷", "braised"],               # 焖 = 闷 = 某种形式的braised
            "煎": ["pan-fried"],                   # 煎 = 英文pan-fried
            "爆炒": ["stir-fried"],               # 爆炒 = stir-fried的一种
            "白切": ["boiled"],                    # 白切 = 水煮的一种
            "油焖": ["oil-braised"]               # 油焖 = oil-braised
        }
        
        for method, variants in cooking_method_mappings.items():
            if method in name:
                for variant in variants:
                    if variant != method:  # 避免重复
                        synonym = name.replace(method, variant)
                        if synonym != name:
                            synonyms.append(synonym)
        
        # 基于食材生成别名（提取主要食材）
        ingredient_aliases = {
            "茄子": ["青茄子", "紫茄子", "eggplant"],
            "土豆": ["马铃薯", "洋芋", "potato"],
            "西红柿": ["番茄", "tomato"],
            "青椒": ["彩椒", "甜椒", "bell pepper"],
            "豆腐": ["嫩豆腐", "老豆腐", "tofu"],
            "白菜": ["大白菜", "小白菜", "cabbage"],
            "萝卜": ["白萝卜", "胡萝卜", "radish"]
        }
        
        for ingredient, aliases in ingredient_aliases.items():
            if ingredient in name:
                for alias in aliases:
                    if alias != ingredient:
                        synonym = name.replace(ingredient, alias)
                        if synonym != name:
                            synonyms.append(synonym)
        
        # 基于地域特色添加别名
        regional_mappings = {
            "川味": ["四川风味", "川菜风格"],
            "粤式": ["广东风味", "粤菜风格"],
            "京味": ["北京风味", "京菜风格"],
            "湘味": ["湖南风味", "湘菜风格"]
        }
        
        for region, variants in regional_mappings.items():
            if region in name:
                for variant in variants:
                    synonym = name.replace(region, variant)
                    if synonym != name:
                        synonyms.append(synonym)
        
        # 去重并返回，按语言分类
        unique_synonyms = list(set(synonyms))
        return self._categorize_synonyms_by_language(unique_synonyms)
    
    def _categorize_synonyms_by_language(self, synonyms: List[str]) -> List[dict]:
        """按语言分类同义词"""
        categorized = []
        
        for synonym in synonyms:
            # 检测语言
            if self._is_english(synonym):
                categorized.append({
                    "term": synonym,
                    "language": "en",
                    "language_code": "en-US"
                })
            elif self._is_chinese(synonym):
                categorized.append({
                    "term": synonym, 
                    "language": "zh",
                    "language_code": "zh-CN"
                })
            else:
                # 默认为中文
                categorized.append({
                    "term": synonym,
                    "language": "zh", 
                    "language_code": "zh-CN"
                })
        
        return categorized
    
    def _is_english(self, text: str) -> bool:
        """检测是否为英文"""
        import re
        # 检查是否主要包含英文字母和空格
        english_chars = re.findall(r'[a-zA-Z\s\-]', text)
        return len(english_chars) / len(text) > 0.7 if text else False
    
    def _is_chinese(self, text: str) -> bool:
        """检测是否为中文"""
        import re
        # 检查是否包含中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars) > 0
    
    def _format_synonyms_for_neo4j(self, synonyms) -> str:
        """格式化同义词用于Neo4j导出"""
        import pandas as pd
        import json

        # 处理NaN值和空值
        if pd.isna(synonyms) or not synonyms:
            return ""

        # 如果是字符串，尝试解析为JSON
        if isinstance(synonyms, str):
            if synonyms.strip() == "[]" or synonyms.strip() == "":
                return ""
            try:
                synonyms = json.loads(synonyms)
            except (json.JSONDecodeError, ValueError):
                # 如果不是JSON，当作单个同义词处理
                return synonyms.strip()

        # 如果不是列表，返回空字符串
        if not isinstance(synonyms, (list, tuple)):
            return ""

        formatted_terms = []
        for synonym_data in synonyms:
            if isinstance(synonym_data, dict):
                # 新格式：包含语言信息
                term = synonym_data.get('term', '')
                lang = synonym_data.get('language', 'zh')
                if term:
                    formatted_terms.append(f"{term}({lang})")
            else:
                # 兼容旧格式：纯字符串
                if synonym_data and str(synonym_data).strip():
                    formatted_terms.append(str(synonym_data).strip())

        return "|".join(formatted_terms)
    
    def _generate_ingredient_synonyms(self, name: str) -> List[str]:
        """生成食材的同义词列表"""
        ingredient_synonym_dict = {
            # 蔬菜类
            "青茄子": ["茄子", "紫茄子", "圆茄"],
            "西红柿": ["番茄", "洋柿子"],
            "土豆": ["马铃薯", "洋芋", "地蛋"],
            "红薯": ["地瓜", "甘薯", "山芋"],
            "玉米": ["苞米", "玉蜀黍"],
            "青椒": ["柿子椒", "甜椒", "彩椒"],
            "大葱": ["葱白", "韭葱"],
            "小葱": ["香葱", "细葱"],
            "香菜": ["芫荽", "胡荽"],
            "菠菜": ["赤根菜", "波斯菜"],
            
            # 调料类
            "生抽": ["淡色酱油", "鲜味酱油"],
            "老抽": ["深色酱油", "红烧酱油"],
            "料酒": ["黄酒", "绍兴酒"],
            "白糖": ["细砂糖", "绵白糖"],
            "冰糖": ["冰片糖", "块糖"],
            "八角": ["大料", "茴香"],
            
            # 蛋白质类
            "鸡蛋": ["鸡子", "土鸡蛋"],
            "豆腐": ["水豆腐", "嫩豆腐"]
        }
        
        synonyms = ingredient_synonym_dict.get(name, [])
        return self._categorize_synonyms_by_language(synonyms)
    
    def batch_process_recipes(self, recipe_dir: str, resume: bool = True) -> Tuple[int, int]:
        """批量处理菜谱目录 - 支持断点续传和分批保存"""
        import glob
        
        # 检查是否要恢复之前的进度
        progress_data = {}
        if resume:
            progress_data = self.load_progress()
            if progress_data:
                processed_count = progress_data.get("processed_count", 0)
                print(f"检测到未完成任务，已处理: {processed_count} 个菜谱")
                
                confirm = input("\n是否继续之前的处理? (Y/n): ").strip().lower()
                if confirm == 'n':
                    print("重新开始处理...")
                    self.processed_files.clear()
                    self.current_batch = 0
                    self.concept_id_counter = 201000000
                    # 清理进度文件
                    if os.path.exists(self.progress_file):
                        os.remove(self.progress_file)
                else:
                    print("继续之前的处理...")
        
        # 专门扫描dishes目录
        dishes_dir = os.path.join(recipe_dir, "dishes")
        if not os.path.exists(dishes_dir):
            # 如果没有dishes目录，则扫描整个目录（向后兼容）
            dishes_dir = recipe_dir
            print(f"未找到dishes目录，扫描整个目录: {recipe_dir}")
        else:
            print(f"扫描菜谱目录: {dishes_dir}")
        
        recipe_files = glob.glob(os.path.join(dishes_dir, "**/*.md"), recursive=True)
        
        # 过滤排除的目录
        filtered_files = []
        for recipe_file in recipe_files:
            relative_path = os.path.relpath(recipe_file, recipe_dir)
            path_parts = relative_path.replace('\\', '/').split('/')
            
            # 检查是否包含排除的目录
            if any(excluded in path_parts for excluded in self.ai_agent.excluded_directories):
                continue
                
            filtered_files.append(recipe_file)
        
        recipe_files = filtered_files
        
        # 过滤掉已处理的文件
        remaining_files = []
        for recipe_file in recipe_files:
            relative_path = os.path.relpath(recipe_file, recipe_dir)
            if relative_path not in self.processed_files:
                remaining_files.append(recipe_file)
        
        total_files = len(recipe_files)
        remaining_count = len(remaining_files)
        already_processed = len(self.processed_files)
        
        print(f"📊 文件统计:")
        print(f"   - 总文件数: {total_files}")
        print(f"   - 已处理: {already_processed}")
        print(f"   - 待处理: {remaining_count}")
        
        if remaining_count == 0:
            print("✅ 所有文件都已处理完成!")
            return already_processed, 0
        
        processed_count = already_processed
        failed_count = 0
        current_batch_count = 0
        
        try:
            for i, recipe_file in enumerate(remaining_files):
                try:
                    with open(recipe_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    relative_path = os.path.relpath(recipe_file, recipe_dir)
                    
                    # 处理菜谱
                    self.process_recipe(content, relative_path)
                    
                    # 标记为已处理
                    self.processed_files.add(relative_path)
                    processed_count += 1
                    current_batch_count += 1
                    
                    # 简化进度显示
                    if processed_count % 20 == 0:
                        progress = (processed_count / total_files) * 100
                        print(f"进度: {processed_count}/{total_files} ({progress:.1f}%)")
                    
                    # 保存进度（每处理5个文件保存一次进度）
                    if processed_count % 5 == 0:
                        self.save_progress(relative_path, total_files, processed_count)
                    
                    # 检查是否需要保存批次
                    if current_batch_count >= self.batch_size:
                        self.save_batch_data()

                        # 重置当前批次
                        self.concepts.clear()
                        self.relationships.clear()
                        self.current_batch += 1
                        current_batch_count = 0
                        
                except Exception as e:
                    failed_count += 1
                    print(f"❌ 处理失败: {recipe_file} - {str(e)}")
                    continue
            
            # 保存最后一个批次（如果有数据）
            if current_batch_count > 0:
                self.save_batch_data()
            
            # 最终保存进度
            self.save_progress("COMPLETED", total_files, processed_count)
            
            print(f"处理完成: 成功 {processed_count - already_processed} 个，失败 {failed_count} 个")
            
        except KeyboardInterrupt:
            print(f"用户中断，已保存进度: {processed_count}/{total_files}")

            # 保存当前批次数据
            if current_batch_count > 0:
                self.save_batch_data()

            # 保存进度
            self.save_progress("INTERRUPTED", total_files, processed_count)
            
        return processed_count, failed_count
    
    def export_to_csv(self, output_dir: str):
        """导出为CSV格式"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出概念
        concepts_df = pd.DataFrame(self.concepts)
        concepts_df.to_csv(os.path.join(output_dir, "concepts.csv"), 
                          index=False, encoding='utf-8')
        
        # 导出关系
        relationships_df = pd.DataFrame(self.relationships)
        relationships_df.to_csv(os.path.join(output_dir, "relationships.csv"), 
                               index=False, encoding='utf-8')
        
        print(f"CSV文件已导出到: {output_dir}")
        print(f"- 概念数量: {len(self.concepts)}")
        print(f"- 关系数量: {len(self.relationships)}")
    
    def export_to_rf2_format(self, output_dir: str):
        """导出为RF2标准格式"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 合并所有概念（包括预定义概念）
        all_concepts = list(self.predefined_concepts) + self.concepts
        
        # 1. 导出概念文件 (rf2_concept.txt)
        concept_headers = ["id", "effectiveTime", "active", "moduleId", "definitionStatusId"]
        with open(os.path.join(output_dir, "rf2_concept.txt"), 'w', encoding='utf-8') as f:
            f.write('\t'.join(concept_headers) + '\n')
            for concept in all_concepts:
                f.write(f"{concept['concept_id']}\t20241201\t1\t900000000\t900000000\n")
        
        # 2. 导出描述文件 (rf2_description.txt) - 包含别名
        desc_headers = ["id", "effectiveTime", "active", "moduleId", "conceptId", 
                       "languageCode", "typeId", "term", "caseSignificanceId"]
        
        desc_id_counter = 1
        with open(os.path.join(output_dir, "rf2_description.txt"), 'w', encoding='utf-8') as f:
            f.write('\t'.join(desc_headers) + '\n')
            
            for concept in all_concepts:
                concept_id = concept['concept_id']
                
                # 完全限定名 (FSN)
                if 'fsn' in concept and concept['fsn']:
                    f.write(f"D{desc_id_counter:06d}\t20241201\t1\t900000000\t{concept_id}\t"
                           f"zh-CN\t900000003\t{concept['fsn']}\t900000000\n")
                    desc_id_counter += 1
                
                # 首选术语 (PT)
                if 'preferred_term' in concept and concept['preferred_term']:
                    f.write(f"D{desc_id_counter:06d}\t20241201\t1\t900000000\t{concept_id}\t"
                           f"zh-CN\t900000001\t{concept['preferred_term']}\t900000000\n")
                    desc_id_counter += 1
                
                # 同义词 (Synonyms) - 支持多语言
                if 'synonyms' in concept and concept['synonyms']:
                    for synonym_data in concept['synonyms']:
                        if isinstance(synonym_data, dict):
                            # 新格式：包含语言信息
                            term = synonym_data['term']
                            lang_code = synonym_data['language_code']
                        else:
                            # 兼容旧格式：纯字符串
                            term = synonym_data
                            lang_code = "zh-CN"  # 默认中文
                        
                        f.write(f"D{desc_id_counter:06d}\t20241201\t1\t900000000\t{concept_id}\t"
                               f"{lang_code}\t900000002\t{term}\t900000000\n")
                        desc_id_counter += 1
        
        # 3. 导出关系文件 (rf2_relationship.txt)
        rel_headers = ["id", "effectiveTime", "active", "moduleId", "sourceId", 
                      "destinationId", "relationshipGroup", "typeId", "characteristicTypeId", "modifierId"]
        
        with open(os.path.join(output_dir, "rf2_relationship.txt"), 'w', encoding='utf-8') as f:
            f.write('\t'.join(rel_headers) + '\n')
            
            rel_id_counter = 1
            for relationship in self.relationships:
                f.write(f"R{rel_id_counter:06d}\t20241201\t1\t900000000\t"
                       f"{relationship['source_id']}\t{relationship['target_id']}\t0\t"
                       f"{relationship['relationship_type']}\t900000000\t900000000\n")
                rel_id_counter += 1
        
        print(f"RF2格式文件已导出到: {output_dir}")
        print(f"- rf2_concept.txt: {len(all_concepts)} 个概念")
        print(f"- rf2_description.txt: 包含首选术语和同义词")
        print(f"- rf2_relationship.txt: {len(self.relationships)} 个关系")
    
    def export_to_neo4j_csv(self, output_dir: str, merge_batches: bool = True):
        """导出为Neo4j导入格式的CSV - 支持合并批次数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果需要合并批次数据
        final_concepts = []
        final_relationships = []
        
        if merge_batches:
            # 先尝试合并所有批次数据
            total_concepts, total_relationships = self.merge_all_batches()
            
            # 如果有合并的数据，使用合并后的数据
            if total_concepts > 0:
                concepts_df = pd.read_csv(os.path.join(output_dir, "concepts.csv"))
                final_concepts = concepts_df.to_dict('records')
            else:
                final_concepts = self.concepts
                
            if total_relationships > 0:
                relationships_df = pd.read_csv(os.path.join(output_dir, "relationships.csv"))
                final_relationships = relationships_df.to_dict('records')
            else:
                final_relationships = self.relationships
        else:
            # 使用当前内存中的数据
            final_concepts = self.concepts
            final_relationships = self.relationships
        
        # 准备节点数据
        nodes_data = []
        
        # 首先添加预定义概念
        for predefined_concept in self.predefined_concepts:
            node = {
                "nodeId": predefined_concept["concept_id"],
                "labels": predefined_concept["concept_type"],
                "name": predefined_concept["name"],
                "preferredTerm": predefined_concept.get("preferred_term", ""),
                "fsn": predefined_concept.get("fsn", ""),
                "conceptType": predefined_concept["concept_type"],
                "synonyms": self._format_synonyms_for_neo4j(predefined_concept.get("synonyms", []))
            }
            nodes_data.append(node)
        
        # 然后添加动态生成的概念
        for concept in final_concepts:
            node = {
                "nodeId": concept["concept_id"],
                "labels": concept["concept_type"],
                "name": concept["name"],
                "preferredTerm": concept.get("preferred_term", ""),
                "category": concept.get("category", ""),
                "conceptType": concept["concept_type"],
                "synonyms": self._format_synonyms_for_neo4j(concept.get("synonyms", []))
            }
            
            # 添加特定类型的属性
            if concept["concept_type"] == "Recipe":
                node.update({
                    "difficulty": concept.get("difficulty", ""),
                    "cuisineType": concept.get("cuisine_type", ""),
                    "prepTime": concept.get("prep_time", ""),
                    "cookTime": concept.get("cook_time", ""),
                    "servings": concept.get("servings", ""),
                    "tags": concept.get("tags", ""),
                    "filePath": concept.get("file_path", "")
                })
            elif concept["concept_type"] == "Ingredient":
                node.update({
                    "amount": concept.get("amount", ""),
                    "unit": concept.get("unit", ""),
                    "isMain": concept.get("is_main", "")
                })
            elif concept["concept_type"] == "CookingStep":
                node.update({
                    "description": concept.get("description", ""),
                    "stepNumber": concept.get("step_number", ""),
                    "methods": concept.get("methods", ""),
                    "tools": concept.get("tools", ""),
                    "timeEstimate": concept.get("time_estimate", "")
                })
            
            nodes_data.append(node)
        
        # 导出节点
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(os.path.join(output_dir, "nodes.csv"), 
                       index=False, encoding='utf-8')
        
        # 准备关系数据
        relationships_data = []
        for rel in final_relationships:
            relationship = {
                "startNodeId": rel["source_id"],
                "endNodeId": rel["target_id"],
                "relationshipType": rel["relationship_type"],
                "relationshipId": rel["relationship_id"]
            }
            
            # 添加额外属性
            for key, value in rel.items():
                if key not in ["source_id", "target_id", "relationship_type", "relationship_id"]:
                    relationship[key] = value
            
            relationships_data.append(relationship)
        
        # 导出关系
        relationships_df = pd.DataFrame(relationships_data)
        relationships_df.to_csv(os.path.join(output_dir, "relationships.csv"), 
                               index=False, encoding='utf-8')
        
        # 生成Neo4j导入脚本
        import_script = f"""
// Neo4j 数据导入脚本

// 导入节点
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CREATE (n:Concept)
SET n.nodeId = row.nodeId,
    n.name = row.name,
    n.preferredTerm = row.preferredTerm,
    n.category = row.category,
    n.conceptType = row.conceptType,
    n.difficulty = toInteger(row.difficulty),
    n.cuisineType = row.cuisineType,
    n.prepTime = row.prepTime,
    n.cookTime = row.cookTime,
    n.servings = row.servings,
    n.tags = row.tags,
    n.filePath = row.filePath,
    n.amount = row.amount,
    n.unit = row.unit,
    n.isMain = toBoolean(row.isMain),
    n.description = row.description,
    n.stepNumber = toInteger(row.stepNumber),
    n.methods = row.methods,
    n.tools = row.tools,
    n.timeEstimate = row.timeEstimate;

// 创建索引
CREATE INDEX concept_id_index IF NOT EXISTS FOR (c:Concept) ON (c.nodeId);
CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name);
CREATE INDEX concept_category_index IF NOT EXISTS FOR (c:Concept) ON (c.category);

// 导入关系
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (start:Concept {{nodeId: row.startNodeId}})
MATCH (end:Concept {{nodeId: row.endNodeId}})
CALL apoc.create.relationship(start, row.relationshipType, {{
    relationshipId: row.relationshipId,
    amount: row.amount,
    unit: row.unit,
    stepOrder: toInteger(row.step_order)
}}, end) YIELD rel
RETURN count(rel);
"""
        
        with open(os.path.join(output_dir, "neo4j_import.cypher"), 'w', encoding='utf-8') as f:
            f.write(import_script)
        
        print(f"Neo4j CSV文件已导出到: {output_dir}")
        print(f"- 节点数量: {len(nodes_data)}")
        print(f"- 关系数量: {len(relationships_data)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用AI智能解析菜谱生成知识图谱')
    parser.add_argument('recipe_dir', help='菜谱目录路径')
    parser.add_argument('-k', '--api-key', required=True, help='OpenAI API密钥')
    parser.add_argument('-o', '--output', default='./ai_output', help='输出目录路径')
    parser.add_argument('--format', choices=['csv', 'neo4j'], default='neo4j', 
                       help='输出格式 (csv 或 neo4j)')
    parser.add_argument('--base-url', default='https://api.openai.com/v1', 
                       help='OpenAI API基础URL')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.recipe_dir):
        print(f"错误: 菜谱目录不存在 - {args.recipe_dir}")
        return
    
    # 创建AI agent
    print("初始化OpenAI AI Agent...")
    ai_agent = LLMRecipeAgent(args.api_key, args.base_url)
    
    # 创建知识图谱构建器
    builder = RecipeKnowledgeGraphBuilder(ai_agent, args.output)
    
    # 批量处理菜谱
    print(f"开始处理菜谱目录: {args.recipe_dir}")
    processed, failed = builder.batch_process_recipes(args.recipe_dir)
    
    print(f"\n处理完成:")
    print(f"- 成功处理: {processed} 个菜谱")
    print(f"- 处理失败: {failed} 个菜谱")
    
    # 导出数据
    if args.format == 'neo4j':
        builder.export_to_neo4j_csv(args.output)
    else:
        builder.export_to_csv(args.output)

if __name__ == "__main__":
    # 测试用例
    if len(os.sys.argv) == 1:
        print("AI菜谱解析器测试模式")
        print("请提供OpenAI API密钥和菜谱目录路径")
        print("使用方法:")
        print("python recipe_ai_agent.py /path/to/recipes -k YOUR_API_KEY")
    else:
        main() 
