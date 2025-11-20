#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的AI菜谱解析运行脚本
"""

import os
import json
import sys
from recipe_ai_agent import LLMRecipeAgent, RecipeKnowledgeGraphBuilder

def load_config():
    """加载配置文件"""
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("警告: 未找到config.json配置文件，将使用默认配置")
        return {
            "openai": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1"
            },
            "output": {
                "format": "neo4j",
                "directory": "./ai_output"
            }
        }

def setup_api_key():
    """设置API密钥"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("请输入OpenAI API密钥: ").strip()
        if not api_key:
            print("错误: 必须提供API密钥")
            sys.exit(1)
    return api_key

def get_recipe_directory():
    """获取菜谱目录"""
    if len(sys.argv) > 1:
        recipe_dir = sys.argv[1]
    else:
        recipe_dir = input("请输入菜谱目录路径: ").strip()
    
    if not os.path.exists(recipe_dir):
        print(f"错误: 目录不存在 - {recipe_dir}")
        sys.exit(1)
    
    return recipe_dir

def test_single_recipe():
    """测试单个菜谱解析"""
    test_recipe = """# 红烧茄子的做法
预估烹饪难度：★★★★

## 必备原料和工具
- 大蒜
- 大葱
- 青辣椒
- 洋葱
- 西红柿
- 青茄子
- 盐
- 酱油
- 鸡蛋
- 面粉
- 淀粉

## 计算
每次制作前需要确定计划做几份。一份正好够 2 个人食用

## 操作
1. 青茄子、青辣椒、西红柿、洋葱、大葱洗净。
2. 大葱切 5 毫米宽的葱花，大蒜扒皮并拍碎，西红柿切 6 立方厘米的块。
3. 茄子切菱形块。
4. 将面粉倒入盆中，依次加入少量水，搅拌均匀，呈粘稠糊状。
5. 热锅，放入茄块翻炒至金黄色。
"""
    
    print("=== 测试单个菜谱解析 ===")
    
    # 加载配置
    config = load_config()
    api_key = config["openai"].get("api_key")
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        api_key = setup_api_key()
    
    try:
        agent = LLMRecipeAgent(api_key)
        recipe_info = agent.extract_recipe_info(test_recipe, "dishes/vegetable_dish/红烧茄子.md")
        
        print(f"测试成功: {recipe_info.name} ({len(recipe_info.ingredients)}个食材, {len(recipe_info.steps)}个步骤)")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("🍳 AI菜谱知识图谱生成器")
    print("=" * 50)
    
    # 检查是否为测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        success = test_single_recipe()
        sys.exit(0 if success else 1)
    
    # 加载配置
    config = load_config()
    
    # 设置API密钥
    api_key = config["openai"].get("api_key")
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        api_key = setup_api_key()
    
    # 获取菜谱目录
    recipe_dir = get_recipe_directory()
    
    # 确认参数
    print(f"\n配置信息:")
    print(f"- API密钥: {api_key[:8]}...")
    print(f"- 菜谱目录: {recipe_dir}")
    print(f"- 输出格式: {config['output'].get('format', 'neo4j')}")
    print(f"- 输出目录: {config['output'].get('directory', './ai_output')}")
    
    confirm = input("\n确认开始处理? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消处理")
        return
    
    try:
        # 创建AI agent
        print("\n🤖 初始化AI Agent...")
        ai_agent = LLMRecipeAgent(api_key, config["openai"].get("base_url"))
        
        # 创建知识图谱构建器
        output_dir = config["output"].get("directory", "./ai_output")
        batch_size = config.get("processing", {}).get("batch_size", 20)  # 默认批次大小为20
        builder = RecipeKnowledgeGraphBuilder(ai_agent, output_dir, batch_size)
        
        # 批量处理菜谱
        print(f"\n📚 开始处理菜谱目录...")
        processed, failed = builder.batch_process_recipes(recipe_dir)
        
        print(f"处理结果: 成功 {processed} 个，失败 {failed} 个")
        
        # 导出数据
        output_dir = config["output"].get("directory", "./ai_output")
        output_format = config["output"].get("format", "neo4j")
        
        print(f"导出数据 (格式: {output_format})...")

        if output_format == "neo4j":
            builder.export_to_neo4j_csv(output_dir)
            print(f"Neo4j文件已生成: {output_dir}")
        elif output_format == "rf2":
            builder.export_to_rf2_format(output_dir)
            print(f"RF2文件已生成: {output_dir}")
        else:
            builder.export_to_csv(output_dir)
            print(f"CSV文件已生成: {output_dir}")
        
        print("处理完成!")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {str(e)}")
        print(f"请检查API密钥、网络连接和菜谱文件格式")

def show_help():
    """显示帮助信息"""
    help_text = """
🍳 AI菜谱知识图谱生成器 - 使用指南

基本用法:
  python run_ai_agent.py [菜谱目录路径]
  
测试模式:
  python run_ai_agent.py test
  
环境变量:
          OPENAI_API_KEY - OpenAI API密钥
  
配置文件:
  config.json - 详细配置选项
  
示例:
  python run_ai_agent.py ./HowToCook-master
  python run_ai_agent.py test
  
输出格式:
  - neo4j: 生成Neo4j导入格式的CSV文件
  - csv: 生成标准CSV文件
  
更多信息请查看README.md
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
    else:
        main() 
