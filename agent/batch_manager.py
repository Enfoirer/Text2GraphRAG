#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分批处理管理器 - 管理菜谱处理的分批保存和断点续传功能
"""

import os
import json
import sys
import argparse
from datetime import datetime
from recipe_ai_agent import LLMRecipeAgent, RecipeKnowledgeGraphBuilder

def load_config():
    """加载配置文件"""
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("❌ 未找到config.json配置文件")
        sys.exit(1)

def show_progress_status(output_dir: str):
    """显示处理进度状态"""
    progress_file = os.path.join(output_dir, "progress.json")
    
    if not os.path.exists(progress_file):
        print("📋 未找到进度文件，没有正在进行的任务")
        return
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        print("处理进度状态:")
        print(f"总文件: {progress_data.get('total_files', 'N/A')}, 已处理: {progress_data.get('processed_count', 0)}")

        status = progress_data.get('current_file', 'N/A')
        if status == 'COMPLETED':
            print("状态: 已完成")
        elif status == 'INTERRUPTED':
            print("状态: 被中断")
        else:
            print("状态: 进行中")
            
        # 检查批次文件
        batch_dirs = [d for d in os.listdir(output_dir) 
                     if d.startswith("batch_") and os.path.isdir(os.path.join(output_dir, d))]
        
        if batch_dirs:
            print(f"已保存批次: {len(batch_dirs)} 个")
        
    except Exception as e:
        print(f"❌ 读取进度文件失败: {str(e)}")

def clean_progress(output_dir: str):
    """清理进度文件，重新开始处理"""
    progress_file = os.path.join(output_dir, "progress.json")
    
    if os.path.exists(progress_file):
        confirm = input("⚠️  确认要清理进度文件吗？这将删除所有处理进度 (y/N): ").strip().lower()
        if confirm == 'y':
            os.remove(progress_file)
            print("✅ 进度文件已清理")
        else:
            print("取消操作")
    else:
        print("📋 未找到进度文件")

def clean_batches(output_dir: str):
    """清理所有批次数据"""
    batch_dirs = [d for d in os.listdir(output_dir) 
                 if d.startswith("batch_") and os.path.isdir(os.path.join(output_dir, d))]
    
    if not batch_dirs:
        print("📁 未找到批次数据")
        return
    
    print(f"找到 {len(batch_dirs)} 个批次目录:")
    for batch_dir in sorted(batch_dirs):
        print(f"   - {batch_dir}")
    
    confirm = input("\n⚠️  确认要删除所有批次数据吗？ (y/N): ").strip().lower()
    if confirm == 'y':
        import shutil
        for batch_dir in batch_dirs:
            batch_path = os.path.join(output_dir, batch_dir)
            shutil.rmtree(batch_path)
            print(f"🗑️  已删除: {batch_dir}")
        print("✅ 所有批次数据已清理")
    else:
        print("取消操作")

def merge_batches(output_dir: str):
    """手动合并所有批次数据"""
    config = load_config()
    api_key = config["openai"].get("api_key")
    
    if not api_key:
        print("❌ 未找到API密钥配置")
        return
    
    try:
        ai_agent = LLMRecipeAgent(api_key)
        builder = RecipeKnowledgeGraphBuilder(ai_agent, output_dir)
        
        print("合并批次数据...")
        total_concepts, total_relationships = builder.merge_all_batches()

        if total_concepts > 0 or total_relationships > 0:
            print(f"合并完成: {total_concepts} 概念, {total_relationships} 关系")

            # 生成Neo4j格式
            format_type = config["output"].get("format", "neo4j")
            if format_type == "neo4j":
                builder.export_to_neo4j_csv(output_dir, merge_batches=False)
        else:
            print("未找到有效的批次数据")
            
    except Exception as e:
        print(f"❌ 合并失败: {str(e)}")

def continue_processing(recipe_dir: str, output_dir: str):
    """继续处理中断的任务"""
    config = load_config()
    api_key = config["openai"].get("api_key")
    
    if not api_key:
        print("❌ 未找到API密钥配置")
        return
    
    try:
        ai_agent = LLMRecipeAgent(api_key)
        batch_size = config.get("processing", {}).get("batch_size", 20)
        builder = RecipeKnowledgeGraphBuilder(ai_agent, output_dir, batch_size)
        
        print("继续处理中断的任务...")
        processed, failed = builder.batch_process_recipes(recipe_dir, resume=True)

        print(f"处理完成: 总数 {processed}, 失败 {failed}")

        # 自动合并数据
        print("自动合并批次数据...")
        merge_batches(output_dir)
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")

def show_batch_details(output_dir: str, batch_num: int = None):
    """显示批次详细信息"""
    batch_dirs = [d for d in os.listdir(output_dir) 
                 if d.startswith("batch_") and os.path.isdir(os.path.join(output_dir, d))]
    
    if not batch_dirs:
        print("📁 未找到批次数据")
        return
    
    batch_dirs.sort()
    
    if batch_num is not None:
        target_batch = f"batch_{batch_num:03d}"
        if target_batch not in batch_dirs:
            print(f"❌ 未找到批次 {batch_num}")
            return
        batch_dirs = [target_batch]
    
    import pandas as pd
    
    for batch_dir in batch_dirs:
        print(f"\n📁 {batch_dir}:")
        batch_path = os.path.join(output_dir, batch_dir)
        
        # 概念文件
        concepts_file = os.path.join(batch_path, "concepts.csv")
        if os.path.exists(concepts_file):
            df = pd.read_csv(concepts_file)
            print(f"   概念数量: {len(df)}")
            
            # 按类型统计
            if 'concept_type' in df.columns:
                type_counts = df['concept_type'].value_counts()
                for concept_type, count in type_counts.items():
                    print(f"     - {concept_type}: {count}")
        
        # 关系文件
        relationships_file = os.path.join(batch_path, "relationships.csv")
        if os.path.exists(relationships_file):
            df = pd.read_csv(relationships_file)
            print(f"   关系数量: {len(df)}")
            
            # 按类型统计
            if 'relationship_type' in df.columns:
                type_counts = df['relationship_type'].value_counts()
                for rel_type, count in type_counts.items():
                    print(f"     - {rel_type}: {count}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='分批处理管理器 - 管理菜谱处理的分批保存和断点续传',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python batch_manager.py status                    # 查看处理状态
  python batch_manager.py continue ./HowToCook-master  # 继续中断的处理
  python batch_manager.py merge                     # 合并批次数据
  python batch_manager.py clean-progress            # 清理进度文件
  python batch_manager.py clean-batches             # 清理批次数据
  python batch_manager.py details                   # 显示所有批次详情
  python batch_manager.py details -b 1              # 显示指定批次详情
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # status命令
    subparsers.add_parser('status', help='查看处理进度状态')
    
    # continue命令
    continue_parser = subparsers.add_parser('continue', help='继续中断的处理')
    continue_parser.add_argument('recipe_dir', help='菜谱目录路径')
    
    # merge命令
    subparsers.add_parser('merge', help='合并所有批次数据')
    
    # clean命令
    subparsers.add_parser('clean-progress', help='清理进度文件')
    subparsers.add_parser('clean-batches', help='清理所有批次数据')
    
    # details命令
    details_parser = subparsers.add_parser('details', help='显示批次详细信息')
    details_parser.add_argument('-b', '--batch', type=int, help='指定批次编号')
    
    # 全局参数
    parser.add_argument('-o', '--output', default='./ai_output', help='输出目录路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    print("🛠️  分批处理管理器")
    print("=" * 50)
    
    try:
        if args.command == 'status':
            show_progress_status(args.output)
        elif args.command == 'continue':
            continue_processing(args.recipe_dir, args.output)
        elif args.command == 'merge':
            merge_batches(args.output)
        elif args.command == 'clean-progress':
            clean_progress(args.output)
        elif args.command == 'clean-batches':
            clean_batches(args.output)
        elif args.command == 'details':
            show_batch_details(args.output, args.batch)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  操作被用户中断")
    except Exception as e:
        print(f"\n❌ 操作失败: {str(e)}")

if __name__ == "__main__":
    main() 
