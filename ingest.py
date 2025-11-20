"""
Utility script to convert raw markdown documents into the Neo4j graph that the
GraphRAG system consumes. It wraps RecipeTextToGraphBuilder and exposes common
parameters via CLI flags.
"""

import argparse
import sys
from pathlib import Path

from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules.text_to_graph_ingestor import RecipeTextToGraphBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert markdown/text documents into the Neo4j knowledge graph."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../data/C8/cook",
        help="Directory containing markdown documents to ingest (default: %(default)s)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default="./.nano_cache",
        help="Temporary cache directory for nano-graphrag (default: %(default)s)",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help="Override Neo4j URI; defaults to value from config.py/.env",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=None,
        help="Override Neo4j username; defaults to value from config.py/.env",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Override Neo4j password; defaults to value from config.py/.env",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=1,
        help="Max concurrent LLM/embedding calls during ingestion (default: %(default)s)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["cooking", "medical"],
        default="cooking",
        help="Domain schema to apply when ingesting documents (default: %(default)s)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GraphRAGConfig:
    cfg_dict = DEFAULT_CONFIG.to_dict()
    if args.neo4j_uri is not None:
        cfg_dict["neo4j_uri"] = args.neo4j_uri
    if args.neo4j_user is not None:
        cfg_dict["neo4j_user"] = args.neo4j_user
    if args.neo4j_password is not None:
        cfg_dict["neo4j_password"] = args.neo4j_password
    return GraphRAGConfig.from_dict(cfg_dict)


def main():
    args = parse_args()
    cfg = build_config(args)

    data_dir = Path(args.data_path)
    if not data_dir.exists():
        print(f"[Error] Data path does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    builder = RecipeTextToGraphBuilder(
        data_path=str(data_dir),
        neo4j_uri=cfg.neo4j_uri,
        neo4j_user=cfg.neo4j_user,
        neo4j_password=cfg.neo4j_password,
        working_dir=args.working_dir,
        llm_concurrency=max(1, args.llm_concurrency),
        domain=args.domain,
    )
    builder.build()
    print("✅ Document ingestion completed.")


if __name__ == "__main__":
    main()
