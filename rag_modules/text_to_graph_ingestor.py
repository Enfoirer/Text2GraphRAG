"""
Automatically convert recipe markdown into the Neo4j graph that C9 expects.

This module reuses the light-weight chunking / entity extraction ideas from
``nano_graphrag`` but simplifies the workflow so that we can ingest arbitrary
markdown files without maintaining a hand-written ``nodes.csv`` beforehand.

Typical usage::

    from rag_modules.text_to_graph_ingestor import RecipeTextToGraphBuilder
    builder = RecipeTextToGraphBuilder(
        data_path=\"../../data/C8/cook\",
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password,
    )
    builder.build()

After `build()` finishes, the Neo4j instance already contains `Recipe`,
`Ingredient`, `CookingStep` and `Category` nodes together with the relationships
(`REQUIRES`, `CONTAINS_STEP`, `BELONGS_TO_CATEGORY`) that `GraphDataPreparation`
expects. The original ``GraphDataPreparationModule`` can therefore load the data
without touching the legacy CSV importer.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict

from neo4j import GraphDatabase

from nano_graphrag import prompt as nano_prompt
from nano_graphrag.graphrag import GraphRAG
from nano_graphrag._storage import NetworkXStorage
from nano_graphrag._op import get_chunks, chunking_by_token_size
from nano_graphrag._utils import TokenizerWrapper


COOKING_ENTITY_TYPES = [
    "recipe",
    "dish",
    "ingredient",
    "cooking_step",
    "procedure",
    "tool",
    "utensil",
    "cuisine",
    "category",
]

MEDICAL_ENTITY_TYPES = [
    "disease",
    "condition",
    "diagnosis",
    "symptom",
    "sign",
    "treatment",
    "therapy",
    "medication",
    "drug",
    "procedure",
    "risk_factor",
    "contraindication",
    "care_tip",
    "lifestyle",
    "specialty",
]

SUPPORTED_DOMAINS = {"cooking", "medical"}

MEDICAL_NAME_ALIASES = {
    "cataract": "白内障",
    "age-related macular degeneration": "老年性黄斑变性",
    "amd": "老年性黄斑变性",
    "acute angle-closure glaucoma": "急性闭角型青光眼",
    "glaucoma": "青光眼",
    "dry eye disease": "干眼症",
    "dry eye": "干眼症",
    "keratitis": "角膜炎",
    "conjunctivitis": "结膜炎",
    "uveitis": "葡萄膜炎",
    "anti-vegf": "抗VEGF注药",
    "anti-vegf therapy": "抗VEGF注药",
    "artificial tears": "人工泪液",
    "cyclosporine eye drops": "环孢素A滴眼液",
    "sodium hyaluronate": "透明质酸钠",
    "photophobia": "畏光",
    "blurry vision": "视物模糊",
    "blurred vision": "视物模糊",
    "eye pain": "眼痛",
    "halos": "看灯有光晕",
    "floaters": "飞蚊症",
    "dryness": "干涩",
    "tearing": "流泪",
    "itching": "瘙痒",
    "diabetes": "糖尿病",
    "smoking": "吸烟",
    "age": "高龄",
}


def _sanitize_description(text: str | None) -> str:
    if not text:
        return ""
    return text.replace("\\n", " ").strip()


@dataclass
class RecipeTextToGraphBuilder:
    """
    Build a recipe knowledge graph directly from raw markdown files.

    The builder first relies on :class:`nano_graphrag.graphrag.GraphRAG` to
    chunk the documents and run LLM-based entity / relationship extraction.
    Afterwards the intermediate NetworkX graph is projected into Neo4j using
    the schema that ``GraphDataPreparationModule`` consumes.
    """

    data_path: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    working_dir: str = "./.nano_graphrag_cache"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    llm_concurrency: int = 1
    domain: str = "cooking"

    def __post_init__(self):
        self.domain = (self.domain or "cooking").lower()
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError(f"Unsupported domain '{self.domain}'. Supported domains: {sorted(SUPPORTED_DOMAINS)}")
        self._tokenizer = TokenizerWrapper(tokenizer_type="tiktoken", model_name="gpt-4o")
        self._graph_writer = _Neo4jRecipeGraphWriter(
            uri=self.neo4j_uri, user=self.neo4j_user, password=self.neo4j_password, domain=self.domain
        )
        self._domain_entity_types = (
            COOKING_ENTITY_TYPES if self.domain == "cooking" else MEDICAL_ENTITY_TYPES
        )

    def build(self) -> None:
        """Entrypoint that loads markdown, runs extraction and persists to Neo4j."""
        documents = self._load_markdown_documents()
        if not documents:
            raise ValueError(f"No markdown files found under {self.data_path}")

        # Temporarily steer nano_graphrag to the cooking domain.
        original_entity_types = nano_prompt.PROMPTS.get("DEFAULT_ENTITY_TYPES", [])
        nano_prompt.PROMPTS["DEFAULT_ENTITY_TYPES"] = self._domain_entity_types
        try:
            graph = self._build_semantic_graph(documents)
        finally:
            nano_prompt.PROMPTS["DEFAULT_ENTITY_TYPES"] = original_entity_types

        self._graph_writer.persist_graph(graph)

    def _load_markdown_documents(self) -> List[str]:
        data_dir = Path(self.data_path)
        docs: List[str] = []
        alias_map: Dict[str, str] = {}
        for file in data_dir.rglob("*.md"):
            try:
                content = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file.read_text(encoding="utf-8", errors="ignore")
            docs.append(content)
            if self.domain == "medical":
                alias_map.update(self._extract_aliases_from_document(content))
        if self.domain == "medical" and alias_map:
            self._graph_writer.update_aliases(alias_map)
        return docs

    def _extract_aliases_from_document(self, text: str) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        current_canonical = None
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith("# "):
                title = stripped.lstrip("#").strip()
                current_canonical = self._canonical_name_from_title(title)
                if current_canonical:
                    alias_map.setdefault(current_canonical, current_canonical)
                i += 1
                continue
            if stripped == "## 别名" and current_canonical:
                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        i += 1
                        continue
                    if line.startswith("# ") or line.startswith("## "):
                        break
                    alias = line.lstrip("-").strip()
                    if alias:
                        alias_map[alias] = current_canonical
                    i += 1
                continue
            i += 1
        return alias_map

    @staticmethod
    def _canonical_name_from_title(title: str) -> str:
        if "（" in title:
            return title.split("（", 1)[0].strip()
        if "(" in title:
            return title.split("(", 1)[0].strip()
        return title.strip()

    def _build_semantic_graph(self, documents: Iterable[str]):
        """
        Run nano-graphrag over the given documents and return the NetworkX graph.
        """
        graph_rag = GraphRAG(
            working_dir=self.working_dir,
            enable_local=False,
            enable_naive_rag=False,
            graph_storage_cls=NetworkXStorage,
            best_model_max_async=self.llm_concurrency,
            cheap_model_max_async=self.llm_concurrency,
            embedding_func_max_async=self.llm_concurrency,
        )

        # ``insert`` will chunk, embed and call the LLM asynchronously.
        graph_rag.insert(list(documents))
        storage = graph_rag.chunk_entity_relation_graph
        graph = storage._graph  # type: ignore[attr-defined]
        return graph


class _Neo4jRecipeGraphWriter:
    """
    Helper that converts the extracted NetworkX graph into Neo4j nodes/edges.
    """

    _COOKING_ID_BASE = {
        "Recipe": 201000000,
        "Ingredient": 301000000,
        "CookingStep": 401000000,
        "Category": 501000000,
    }

    _MEDICAL_ID_BASE = {
        "Disease": 211000000,
        "Symptom": 311000000,
        "Treatment": 321000000,
        "Medication": 331000000,
        "RiskFactor": 341000000,
        "CareTip": 351000000,
        "Category": 361000000,
    }

    def __init__(self, uri: str, user: str, password: str, domain: str = "cooking") -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.domain = domain
        self._id_counters = dict(
            self._COOKING_ID_BASE if domain == "cooking" else self._MEDICAL_ID_BASE
        )
        if self.domain == "medical":
            self.name_aliases = {k.lower(): v for k, v in MEDICAL_NAME_ALIASES.items()}
        else:
            self.name_aliases = {}
        self.alias_groups = defaultdict(set)
        if self.domain == "medical":
            for alias, canonical in MEDICAL_NAME_ALIASES.items():
                alias_clean = alias.strip()
                canonical_clean = canonical.strip()
                if alias_clean and canonical_clean and alias_clean.lower() != canonical_clean.lower():
                    self.alias_groups[canonical_clean].add(alias_clean)
        self._ensure_indexes()

    def close(self) -> None:
        self._driver.close()

    def persist_graph(self, graph) -> None:
        """
        Iterate over nano-graphrag nodes/edges and map them to the C9 schema.
        """
        node_cache: Dict[str, Tuple[str, str]] = {}
        with self._driver.session() as session:
            for node_name, node_data in graph.nodes(data=True):
                label = self._map_label(node_data.get("entity_type", ""))
                if not label:
                    continue
                normalized_name = self._normalize_name(node_name)
                node_id = self._ensure_node(session, label, normalized_name, node_data)
                node_cache[node_name] = (label, normalized_name)

            for source, target, edge_data in graph.edges(data=True):
                if source not in node_cache or target not in node_cache:
                    continue
                self._ensure_relationship(
                    session, node_cache[source], node_cache[target], edge_data
                )

    def update_aliases(self, alias_map: Dict[str, str]) -> None:
        if not alias_map:
            return
        for alias, canonical in alias_map.items():
            if not alias or not canonical:
                continue
            canonical_clean = canonical.strip()
            alias_clean = alias.strip()
            self.name_aliases[alias_clean.lower()] = canonical_clean
            if self.domain == "medical" and alias_clean and alias_clean != canonical_clean:
                self.alias_groups[canonical_clean].add(alias_clean)

    def _ensure_indexes(self):
        if self.domain == "medical":
            constraint_specs = [
                ("Disease", "name", "disease_name_unique"),
                ("Symptom", "name", "symptom_name_unique"),
                ("Treatment", "name", "treatment_name_unique"),
                ("Medication", "name", "medication_name_unique"),
                ("RiskFactor", "name", "riskfactor_name_unique"),
                ("CareTip", "name", "caretip_name_unique"),
                ("Category", "name", "medical_category_name_unique"),
            ]
        else:
            constraint_specs = [
                ("Recipe", "name", "recipe_name_unique"),
                ("Ingredient", "name", "ingredient_name_unique"),
                ("Category", "name", "category_name_unique"),
                ("CookingStep", "name", "step_name_unique"),
            ]

        with self._driver.session() as session:
            for label, prop, constraint_name in constraint_specs:
                statement = (
                    f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
                try:
                    session.run(statement)
                except Exception:
                    self._drop_conflicting_indexes(session, label, prop)
                    session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                    session.run(statement)

    def _drop_conflicting_indexes(self, session, label: str, prop: str):
        """Drop legacy indexes (without names) that block constraint creation."""
        result = session.run(
            """
            SHOW INDEXES
            YIELD name, labelsOrTypes, properties
            WHERE $label IN labelsOrTypes AND $prop IN properties
            RETURN name
            """,
            label=label,
            prop=prop,
        )
        names = [record["name"] for record in result]

        for name in names:
            if not name:
                continue
            session.run(f"DROP INDEX {name} IF EXISTS")

    def _map_label(self, entity_type: str) -> str | None:
        et = (entity_type or "").lower()
        if self.domain == "medical":
            if any(key in et for key in ("disease", "condition", "diagnosis")):
                return "Disease"
            if any(key in et for key in ("symptom", "sign")):
                return "Symptom"
            if any(key in et for key in ("treatment", "therapy", "surgery", "procedure")):
                return "Treatment"
            if any(key in et for key in ("medication", "drug", "medicine", "pharmaceutical")):
                return "Medication"
            if any(key in et for key in ("risk", "contraindication", "factor", "comorbidity")):
                return "RiskFactor"
            if any(key in et for key in ("tip", "advice", "lifestyle", "care")):
                return "CareTip"
            if "category" in et or "specialty" in et:
                return "Category"
            # Default to Disease so关键信息不会丢.
            return "Disease"

        if any(key in et for key in ("recipe", "dish")):
            return "Recipe"
        if "ingredient" in et or "spice" in et:
            return "Ingredient"
        if "step" in et or "procedure" in et or "process" in et:
            return "CookingStep"
        if "cuisine" in et or "category" in et:
            return "Category"
        return None

    def _normalize_name(self, name: str) -> str:
        cleaned = (name or "").strip().strip('"').strip("'")
        if self.domain == "medical":
            key = cleaned.lower()
            return self.name_aliases.get(key, cleaned)
        return cleaned

    def _ensure_node(self, session, label: str, name: str, data: dict) -> str:
        node_id = self._next_id(label)
        description = _sanitize_description(data.get("description"))
        params = {
            "name": name.strip(),
            "node_id": node_id,
            "description": description,
            "source_id": data.get("source_id", ""),
            "category": data.get("entity_type", label).title(),
            "aliases": list(self.alias_groups.get(name.strip(), []))
        }
        query = (
            f"MERGE (n:{label} {{name:$name}}) "
            "ON CREATE SET n.nodeId = $node_id, "
            "              n.description = $description, "
            "              n.source_id = $source_id, "
            "              n.category = $category, "
            "              n.aliases = $aliases "
            "SET n.description = CASE "
            "        WHEN n.description IS NULL OR n.description = '' "
            "        THEN $description ELSE n.description END, "
            "    n.aliases = CASE "
            "        WHEN size($aliases) > 0 "
            "        THEN $aliases "
            "        ELSE COALESCE(n.aliases, []) END"
        )
        session.run(query, **params)
        result = session.run(
            f"MATCH (n:{label} {{name:$name}}) RETURN n.nodeId AS nodeId", name=name
        )
        return result.single()["nodeId"]

    def _ensure_relationship(
        self,
        session,
        source_info: Tuple[str, str],
        target_info: Tuple[str, str],
        edge_data: dict,
    ) -> None:
        source_label, source_name = source_info
        target_label, target_name = target_info
        rel_type, direction = self._map_relationship(source_label, target_label)
        if rel_type is None:
            return

        description = _sanitize_description(edge_data.get("description"))
        weight = float(edge_data.get("weight", 1))

        if direction == "reverse":
            source_label, target_label = target_label, source_label
            source_name, target_name = target_name, source_name

        query = (
            f"MATCH (a:{source_label} {{name:$source_name}}), "
            f"      (b:{target_label} {{name:$target_name}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r.description = COALESCE(r.description, $description), "
            "    r.weight = COALESCE(r.weight, $weight)"
        )
        session.run(
            query,
            source_name=source_name,
            target_name=target_name,
            description=description,
            weight=weight,
        )

    def _map_relationship(self, label_a: str, label_b: str) -> Tuple[str | None, str]:
        pair = {label_a, label_b}
        if self.domain == "medical":
            if pair == {"Disease", "Symptom"}:
                return ("HAS_SYMPTOM", "forward" if label_a == "Disease" else "reverse")
            if pair == {"Disease", "Treatment"}:
                return ("RECOMMENDS_TREATMENT", "forward" if label_a == "Disease" else "reverse")
            if pair == {"Disease", "Medication"}:
                return ("USES_MEDICATION", "forward" if label_a == "Disease" else "reverse")
            if pair == {"Disease", "RiskFactor"}:
                return ("HAS_RISK_FACTOR", "forward" if label_a == "Disease" else "reverse")
            if pair == {"Disease", "CareTip"}:
                return ("HAS_CARE_TIP", "forward" if label_a == "Disease" else "reverse")
            if pair == {"Disease", "Category"}:
                return ("BELONGS_TO_CATEGORY", "forward" if label_a == "Disease" else "reverse")
            if pair == {"Symptom", "Treatment"}:
                return ("RELIEVED_BY", "forward" if label_a == "Symptom" else "reverse")
            return ("RELATED_TO", "forward")

        if pair == {"Recipe", "Ingredient"}:
            return ("REQUIRES", "forward" if label_a == "Recipe" else "reverse")
        if pair == {"Recipe", "CookingStep"}:
            return ("CONTAINS_STEP", "forward" if label_a == "Recipe" else "reverse")
        if pair == {"Recipe", "Category"}:
            return ("BELONGS_TO_CATEGORY", "forward" if label_a == "Recipe" else "reverse")
        return ("RELATED_TO", "forward")

    def _next_id(self, label: str) -> str:
        current = self._id_counters[label]
        self._id_counters[label] += 1
        return str(current)


__all__ = ["RecipeTextToGraphBuilder"]
