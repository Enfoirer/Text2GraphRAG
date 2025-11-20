"""
图索引模块
实现实体和关系的键值对结构 (K,V)
K: 索引键（简短词汇或短语）
V: 详细描述段落（包含相关文本片段）
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class EntityKeyValue:
    """实体键值对"""
    entity_name: str
    index_keys: List[str]  # 索引键列表
    value_content: str     # 详细描述内容
    entity_type: str       # 实体类型 (Disease, Symptom, Treatment, 等)
    metadata: Dict[str, Any]

@dataclass 
class RelationKeyValue:
    """关系键值对"""
    relation_id: str
    index_keys: List[str]  # 多个索引键（可包含全局主题）
    value_content: str     # 关系描述内容
    relation_type: str     # 关系类型
    source_entity: str     # 源实体
    target_entity: str     # 目标实体
    metadata: Dict[str, Any]

class GraphIndexingModule:
    """
    图索引模块
    核心功能：
    1. 为实体创建键值对（名称作为唯一索引键）
    2. 为关系创建键值对（多个索引键，包含全局主题）
    3. 去重和优化图操作
    4. 支持增量更新
    """
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        
        # 键值对存储
        self.entity_kv_store: Dict[str, EntityKeyValue] = {}
        self.relation_kv_store: Dict[str, RelationKeyValue] = {}
        
        # 索引映射：key -> entity/relation IDs
        self.key_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] = defaultdict(list)
        
    def create_entity_key_values(self, diseases: List[Any], symptoms: List[Any],
                                treatments: List[Any], medications: List[Any],
                                risk_factors: List[Any], care_tips: List[Any]) -> Dict[str, EntityKeyValue]:
        """
        为实体创建键值对结构
        每个实体使用其名称作为唯一索引键
        """
        logger.info("开始创建实体键值对...")
        
        label_display = {
            "Disease": "疾病",
            "Symptom": "症状",
            "Treatment": "治疗方案",
            "Medication": "药物",
            "RiskFactor": "风险因素",
            "CareTip": "护理建议"
        }

        def _store_entity(entity_list, entity_label, aliases=None):
            for entity in entity_list:
                entity_id = entity.node_id
                entity_name = entity.name or f"{entity_label}_{entity_id}"
                
                props = getattr(entity, "properties", {})
                display_label = label_display.get(entity_label, entity_label)
                content_parts = [f"{display_label}名称: {entity_name}"]
                if props.get("description"):
                    content_parts.append(f"描述: {props['description']}")
                if props.get("category"):
                    content_parts.append(f"分类: {props['category']}")
                if props.get("severity"):
                    content_parts.append(f"严重程度: {props['severity']}")
                if props.get("tags"):
                    content_parts.append(f"标签: {props['tags']}")
                if props.get("methods"):
                    content_parts.append(f"方法: {props['methods']}")
                if props.get("dosage"):
                    content_parts.append(f"剂量: {props['dosage']}")
                
                entity_kv = EntityKeyValue(
                    entity_name=entity_name,
                    index_keys=[entity_name],
                    value_content="\n".join(content_parts),
                    entity_type=entity_label,
                    metadata={
                        "node_id": entity_id,
                        "properties": props
                    }
                )
                
                self.entity_kv_store[entity_id] = entity_kv
                self.key_to_entities[entity_name].append(entity_id)
                if aliases:
                    alias_values = aliases(entity) or []
                    if isinstance(alias_values, str):
                        alias_values = [alias_values]
                    for alias in alias_values:
                        if alias and alias != entity_name:
                            self.key_to_entities[alias].append(entity_id)
        
        _store_entity(diseases, "Disease", aliases=lambda e: e.properties.get("aliases", []))
        _store_entity(symptoms, "Symptom")
        _store_entity(treatments, "Treatment")
        _store_entity(medications, "Medication")
        _store_entity(risk_factors, "RiskFactor")
        _store_entity(care_tips, "CareTip")
        
        logger.info(f"实体键值对创建完成，共 {len(self.entity_kv_store)} 个实体")
        return self.entity_kv_store
    
    def create_relation_key_values(self, relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        """
        为关系创建键值对结构
        关系可能有多个索引键，包含从LLM增强的全局主题
        """
        logger.info("开始创建关系键值对...")
        
        for i, (source_id, relation_type, target_id) in enumerate(relationships):
            relation_id = f"rel_{i}_{source_id}_{target_id}"
            
            # 获取源实体和目标实体信息
            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)
            
            if not source_entity or not target_entity:
                continue
            
            # 构建关系描述
            content_parts = [
                f"关系类型: {relation_type}",
                f"源实体: {source_entity.entity_name} ({source_entity.entity_type})",
                f"目标实体: {target_entity.entity_name} ({target_entity.entity_type})"
            ]
            
            # 生成多个索引键（包含全局主题）
            index_keys = self._generate_relation_index_keys(
                source_entity, target_entity, relation_type
            )
            
            # 创建关系键值对
            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content='\n'.join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={
                    "source_name": source_entity.entity_name,
                    "target_name": target_entity.entity_name,
                    "created_from_graph": True
                }
            )
            
            self.relation_kv_store[relation_id] = relation_kv
            
            # 为每个索引键建立映射
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)
        
        logger.info(f"关系键值对创建完成，共 {len(self.relation_kv_store)} 个关系")
        return self.relation_kv_store
    
    def _generate_relation_index_keys(self, source_entity: EntityKeyValue, 
                                    target_entity: EntityKeyValue, 
                                    relation_type: str) -> List[str]:
        """
        为关系生成多个索引键，包含全局主题
        """
        keys = [relation_type]  # 基础关系类型键
        
        # 根据关系类型和实体类型生成主题键
        if relation_type == "HAS_SYMPTOM":
            keys.extend([
                "临床症状",
                "表现",
                f"{source_entity.entity_name}_症状",
                target_entity.entity_name
            ])
        elif relation_type == "RECOMMENDS_TREATMENT":
            keys.extend([
                "治疗方案",
                "干预措施",
                f"{source_entity.entity_name}_治疗",
                target_entity.entity_name
            ])
        elif relation_type == "USES_MEDICATION":
            keys.extend([
                "药物治疗",
                "用药方案",
                f"{source_entity.entity_name}_药物",
                target_entity.entity_name
            ])
        elif relation_type == "HAS_RISK_FACTOR":
            keys.extend([
                "危险因素",
                "发病风险",
                f"{source_entity.entity_name}_风险",
                target_entity.entity_name
            ])
        elif relation_type == "HAS_CARE_TIP":
            keys.extend([
                "护理建议",
                "生活方式",
                target_entity.entity_name
            ])
        elif relation_type == "BELONGS_TO_CATEGORY":
            keys.extend([
                "疾病分类",
                "医学领域",
                target_entity.entity_name
            ])
        elif relation_type == "RELIEVED_BY":
            keys.extend([
                "症状缓解",
                "对症治疗",
                target_entity.entity_name
            ])
        
        # 使用LLM增强关系索引键（可选）
        if getattr(self.config, 'enable_llm_relation_keys', False):
            enhanced_keys = self._llm_enhance_relation_keys(source_entity, target_entity, relation_type)
            keys.extend(enhanced_keys)
        
        # 去重并返回
        return list(set(keys))
    
    def _llm_enhance_relation_keys(self, source_entity: EntityKeyValue, 
                                 target_entity: EntityKeyValue, 
                                 relation_type: str) -> List[str]:
        """
        使用LLM增强关系索引键，生成全局主题
        """
        prompt = f"""
        分析以下实体关系，生成相关的主题关键词：
        
        源实体: {source_entity.entity_name} ({source_entity.entity_type})
        目标实体: {target_entity.entity_name} ({target_entity.entity_type})
        关系类型: {relation_type}
        
        请生成3-5个相关的主题关键词，用于索引和检索。
        返回JSON格式：{{"keywords": ["关键词1", "关键词2", "关键词3"]}}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result.get("keywords", [])
            
        except Exception as e:
            logger.error(f"LLM增强关系索引键失败: {e}")
            return []
    
    def deduplicate_entities_and_relations(self):
        """
        去重相同的实体和关系，优化图操作
        """
        logger.info("开始去重实体和关系...")
        
        # 实体去重：基于名称
        name_to_entities = defaultdict(list)
        for entity_id, entity_kv in self.entity_kv_store.items():
            name_to_entities[entity_kv.entity_name].append(entity_id)
        
        # 合并重复实体
        entities_to_remove = []
        for name, entity_ids in name_to_entities.items():
            if len(entity_ids) > 1:
                # 保留第一个，合并其他的内容
                primary_id = entity_ids[0]
                primary_entity = self.entity_kv_store[primary_id]
                
                for entity_id in entity_ids[1:]:
                    duplicate_entity = self.entity_kv_store[entity_id]
                    # 合并内容
                    primary_entity.value_content += f"\n\n补充信息: {duplicate_entity.value_content}"
                    # 标记删除
                    entities_to_remove.append(entity_id)
        
        # 删除重复实体
        for entity_id in entities_to_remove:
            del self.entity_kv_store[entity_id]
        
        # 关系去重：基于源-目标-类型
        relation_signature_to_ids = defaultdict(list)
        for relation_id, relation_kv in self.relation_kv_store.items():
            signature = f"{relation_kv.source_entity}_{relation_kv.target_entity}_{relation_kv.relation_type}"
            relation_signature_to_ids[signature].append(relation_id)
        
        # 合并重复关系
        relations_to_remove = []
        for signature, relation_ids in relation_signature_to_ids.items():
            if len(relation_ids) > 1:
                # 保留第一个，删除其他
                for relation_id in relation_ids[1:]:
                    relations_to_remove.append(relation_id)
        
        # 删除重复关系
        for relation_id in relations_to_remove:
            del self.relation_kv_store[relation_id]
        
        # 重建索引映射
        self._rebuild_key_mappings()
        
        logger.info(f"去重完成 - 删除了 {len(entities_to_remove)} 个重复实体，{len(relations_to_remove)} 个重复关系")
    
    def _rebuild_key_mappings(self):
        """重建键到实体/关系的映射"""
        self.key_to_entities.clear()
        self.key_to_relations.clear()
        
        # 重建实体映射
        for entity_id, entity_kv in self.entity_kv_store.items():
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)
        
        # 重建关系映射
        for relation_id, relation_kv in self.relation_kv_store.items():
            for key in relation_kv.index_keys:
                self.key_to_relations[key].append(relation_id)
    
    def get_entities_by_key(self, key: str) -> List[EntityKeyValue]:
        """根据索引键获取实体"""
        entity_ids = self.key_to_entities.get(key, [])
        return [self.entity_kv_store[eid] for eid in entity_ids if eid in self.entity_kv_store]
    
    def get_relations_by_key(self, key: str) -> List[RelationKeyValue]:
        """根据索引键获取关系"""
        relation_ids = self.key_to_relations.get(key, [])
        return [self.relation_kv_store[rid] for rid in relation_ids if rid in self.relation_kv_store]
    

    
    def get_statistics(self) -> Dict[str, Any]:
        """获取键值对存储统计信息"""
        return {
            "total_entities": len(self.entity_kv_store),
            "total_relations": len(self.relation_kv_store),
            "total_entity_keys": sum(len(kv.index_keys) for kv in self.entity_kv_store.values()),
            "total_relation_keys": sum(len(kv.index_keys) for kv in self.relation_kv_store.values()),
            "entity_types": {
                "Disease": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Disease"]),
                "Symptom": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Symptom"]),
                "Treatment": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Treatment"]),
                "Medication": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Medication"]),
                "RiskFactor": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "RiskFactor"]),
                "CareTip": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "CareTip"])
            }
        }
