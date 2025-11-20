"""
图数据库数据准备模块
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]

@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]

class GraphDataPreparationModule:
    """图数据库数据准备模块 - 从Neo4j读取数据并转换为文档"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化图数据库连接
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.diseases: List[GraphNode] = []
        self.symptoms: List[GraphNode] = []
        self.treatments: List[GraphNode] = []
        self.medications: List[GraphNode] = []
        self.risk_factors: List[GraphNode] = []
        self.care_tips: List[GraphNode] = []
        
        self._connect()
    
    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                database=self.database
            )
            logger.info(f"已连接到Neo4j数据库: {self.uri}")
            
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    logger.info("Neo4j连接测试成功")
                    
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def load_graph_data(self) -> Dict[str, Any]:
        """
        从Neo4j加载图数据
        
        Returns:
            包含节点和关系的数据字典
        """
        logger.info("正在从Neo4j加载图数据...")
        
        with self.driver.session() as session:
            # 加载疾病节点，并整合分类信息
            disease_query = """
            MATCH (d:Disease)
            WHERE d.nodeId >= '200000000'
            OPTIONAL MATCH (d)-[:BELONGS_TO_CATEGORY]->(c:Category)
            WITH d, collect(c.name) as categories
            RETURN d.nodeId as nodeId, labels(d) as labels, d.name as name,
                   properties(d) as originalProperties,
                   CASE WHEN size(categories) > 0
                        THEN categories[0]
                        ELSE COALESCE(d.category, '未分类') END as mainCategory,
                   CASE WHEN size(categories) > 0
                        THEN categories
                        ELSE [COALESCE(d.category, '未分类')] END as allCategories
            ORDER BY d.nodeId
            """
            
            result = session.run(disease_query)
            self.diseases = []
            for record in result:
                properties = dict(record["originalProperties"])
                properties["category"] = record["mainCategory"]
                properties["all_categories"] = record["allCategories"]
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=properties
                )
                self.diseases.append(node)
            
            logger.info(f"加载了 {len(self.diseases)} 个疾病节点")
            
            def _load_simple_nodes(label: str) -> List[GraphNode]:
                query = f"""
                MATCH (n:{label})
                WHERE n.nodeId >= '200000000'
                RETURN n.nodeId as nodeId, labels(n) as labels, n.name as name,
                       properties(n) as properties
                ORDER BY n.nodeId
                """
                records = session.run(query)
                nodes = []
                for rec in records:
                    nodes.append(GraphNode(
                        node_id=rec["nodeId"],
                        labels=rec["labels"],
                        name=rec["name"],
                        properties=rec["properties"]
                    ))
                logger.info(f"加载了 {len(nodes)} 个 {label} 节点")
                return nodes
            
            self.symptoms = _load_simple_nodes("Symptom")
            self.treatments = _load_simple_nodes("Treatment")
            self.medications = _load_simple_nodes("Medication")
            self.risk_factors = _load_simple_nodes("RiskFactor")
            self.care_tips = _load_simple_nodes("CareTip")
        
        # 向后兼容的属性命名
        self.recipes = self.diseases
        self.ingredients = self.symptoms
        self.cooking_steps = self.treatments
        
        return {
            'diseases': len(self.diseases),
            'symptoms': len(self.symptoms),
            'treatments': len(self.treatments),
            'medications': len(self.medications),
            'risk_factors': len(self.risk_factors),
            'care_tips': len(self.care_tips)
        }
    
    def build_recipe_documents(self) -> List[Document]:
        """
        构建疾病知识文档，汇总症状、治疗、药物等信息
        """
        logger.info("正在构建疾病文档...")
        
        documents = []
        
        with self.driver.session() as session:
            for disease in self.diseases:
                try:
                    disease_id = disease.node_id
                    disease_name = disease.name
                    
                    symptoms_query = """
                    MATCH (d:Disease {nodeId: $disease_id})-[:HAS_SYMPTOM]->(s:Symptom)
                    RETURN s.name as name, s.description as description, s.severity as severity
                    ORDER BY s.name
                    """
                    treatments_query = """
                    MATCH (d:Disease {nodeId: $disease_id})-[:RECOMMENDS_TREATMENT]->(t:Treatment)
                    RETURN t.name as name, t.description as description, t.methods as methods
                    ORDER BY t.name
                    """
                    medications_query = """
                    MATCH (d:Disease {nodeId: $disease_id})-[:USES_MEDICATION]->(m:Medication)
                    RETURN m.name as name, m.description as description, m.dosage as dosage
                    ORDER BY m.name
                    """
                    risks_query = """
                    MATCH (d:Disease {nodeId: $disease_id})-[:HAS_RISK_FACTOR]->(r:RiskFactor)
                    RETURN r.name as name, r.description as description
                    ORDER BY r.name
                    """
                    care_query = """
                    MATCH (d:Disease {nodeId: $disease_id})-[:HAS_CARE_TIP]->(c:CareTip)
                    RETURN c.name as name, c.description as description
                    ORDER BY c.name
                    """
                    
                    def _format_result(result):
                        formatted = []
                        for record in result:
                            parts = [record["name"]]
                            if record.get("description"):
                                parts.append(record["description"])
                            if record.get("severity"):
                                parts.append(f"严重程度: {record['severity']}")
                            if record.get("methods"):
                                parts.append(f"方式: {record['methods']}")
                            if record.get("dosage"):
                                parts.append(f"剂量: {record['dosage']}")
                            formatted.append(" - ".join(parts))
                        return formatted
                    
                    symptoms_info = _format_result(session.run(symptoms_query, {"disease_id": disease_id}))
                    treatments_info = _format_result(session.run(treatments_query, {"disease_id": disease_id}))
                    medications_info = _format_result(session.run(medications_query, {"disease_id": disease_id}))
                    risks_info = _format_result(session.run(risks_query, {"disease_id": disease_id}))
                    care_info = _format_result(session.run(care_query, {"disease_id": disease_id}))
                    
                    content_parts = [f"# {disease_name}"]
                    
                    if disease.properties.get("description"):
                        content_parts.append(f"\n## 疾病概述\n{disease.properties['description']}")
                    
                    if disease.properties.get("severity"):
                        content_parts.append(f"严重程度: {disease.properties['severity']}")
                    
                    if disease.properties.get("category"):
                        content_parts.append(f"所属分类: {disease.properties['category']}")
                    
                    if disease.properties.get("aliases"):
                        aliases = disease.properties.get("aliases", [])
                        if isinstance(aliases, str):
                            alias_text = aliases
                        else:
                            alias_text = "、".join(aliases)
                        content_parts.append(f"别名: {alias_text}")
                    
                    if disease.properties.get("tags"):
                        content_parts.append(f"相关标签: {disease.properties['tags']}")
                    
                    if symptoms_info:
                        content_parts.append("\n## 典型症状")
                        for i, symptom in enumerate(symptoms_info, 1):
                            content_parts.append(f"{i}. {symptom}")
                    
                    if treatments_info:
                        content_parts.append("\n## 常用治疗方案")
                        for i, treatment in enumerate(treatments_info, 1):
                            content_parts.append(f"{i}. {treatment}")
                    
                    if medications_info:
                        content_parts.append("\n## 常用药物")
                        for i, medication in enumerate(medications_info, 1):
                            content_parts.append(f"{i}. {medication}")
                    
                    if risks_info:
                        content_parts.append("\n## 常见风险因素")
                        for i, risk in enumerate(risks_info, 1):
                            content_parts.append(f"{i}. {risk}")
                    
                    if care_info:
                        content_parts.append("\n## 护理与生活方式建议")
                        for i, tip in enumerate(care_info, 1):
                            content_parts.append(f"{i}. {tip}")
                    
                    full_content = "\n".join(content_parts)
                    
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "node_id": disease_id,
                            "entity_name": disease_name,
                            "node_type": "Disease",
                            "category": disease.properties.get("category", "未分类"),
                            "all_categories": disease.properties.get("all_categories", []),
                            "severity": disease.properties.get("severity", ""),
                            "aliases": disease.properties.get("aliases", []),
                            "tags": disease.properties.get("tags", ""),
                            "symptom_count": len(symptoms_info),
                            "treatment_count": len(treatments_info),
                            "medication_count": len(medications_info),
                            "risk_factor_count": len(risks_info),
                            "care_tip_count": len(care_info),
                            "doc_type": "disease",
                            "content_length": len(full_content)
                        }
                    )
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.warning(f"构建疾病文档失败 {disease_name} (ID: {disease_id}): {e}")
                    continue
        
        self.documents = documents
        logger.info(f"成功构建 {len(documents)} 个疾病文档")
        return documents
    
    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        对文档进行分块处理
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分块后的文档列表
        """
        logger.info(f"正在进行文档分块，块大小: {chunk_size}, 重叠: {chunk_overlap}")
        
        if not self.documents:
            raise ValueError("请先构建文档")
        
        chunks = []
        chunk_id = 0
        
        for doc in self.documents:
            content = doc.page_content
            
            # 简单的按长度分块
            if len(content) <= chunk_size:
                # 内容较短，不需要分块
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                        "parent_id": doc.metadata["node_id"],
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(content),
                        "doc_type": "chunk"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 按章节分块（基于标题）
                sections = content.split('\n## ')
                if len(sections) <= 1:
                    # 没有二级标题，按长度强制分块
                    total_chunks = (len(content) - 1) // (chunk_size - chunk_overlap) + 1
                    
                    for i in range(total_chunks):
                        start = i * (chunk_size - chunk_overlap)
                        end = min(start + chunk_size, len(content))
                        
                        chunk_content = content[start:end]
                        
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                else:
                    # 按章节分块
                    total_chunks = len(sections)
                    for i, section in enumerate(sections):
                        if i == 0:
                            # 第一个部分包含标题
                            chunk_content = section
                        else:
                            # 其他部分添加章节标题
                            chunk_content = f"## {section}"
                        
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk",
                                "section_title": section.split('\n')[0] if i > 0 else "主标题"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
        
        self.chunks = chunks
        logger.info(f"文档分块完成，共生成 {len(chunks)} 个块")
        return chunks
    

    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_diseases': len(self.diseases),
            'total_symptoms': len(self.symptoms),
            'total_treatments': len(self.treatments),
            'total_medications': len(self.medications),
            'total_risk_factors': len(self.risk_factors),
            'total_care_tips': len(self.care_tips),
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks)
        }
        
        if self.documents:
            categories = {}
            severities = {}
            
            for doc in self.documents:
                category = doc.metadata.get('category', '未分类')
                categories[category] = categories.get(category, 0) + 1
                
                severity = doc.metadata.get('severity', '未知')
                severities[severity] = severities.get(severity, 0) + 1
            
            stats.update({
                'categories': categories,
                'severity_distribution': severities,
                'avg_content_length': sum(doc.metadata.get('content_length', 0) for doc in self.documents) / len(self.documents),
                'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
            })
        
        return stats
    

    
    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close() 
