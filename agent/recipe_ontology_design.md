# 菜谱知识图谱设计方案
## 烹饪本体设计

### 概述
本设计方案将菜谱数据结构化为本体图数据库，支持语义查询、推理和知识发现。

### 核心概念体系 (Concept Hierarchy)

#### 1. 根概念 (Root Concept)
```
100000000 | Culinary Concept | (烹饪概念)
```

#### 2. 顶级概念 (Top Level Concepts)
```
200000000 | Recipe | (菜谱)
300000000 | Ingredient | (食材)
400000000 | Tool | (工具)
500000000 | Cooking Method | (烹饪方法)
600000000 | Difficulty Level | (难度等级)
700000000 | Recipe Category | (菜谱分类)
800000000 | Measurement Unit | (计量单位)
900000000 | Cooking Step | (烹饪步骤)
```

### 概念分类体系

#### 菜谱分类 (Recipe Categories)
```
710000000 | Vegetable Dish | (素菜)
720000000 | Meat Dish | (荤菜)
730000000 | Aquatic Product | (水产)
740000000 | Breakfast | (早餐)
750000000 | Staple Food | (主食)
760000000 | Soup | (汤类)
770000000 | Dessert | (甜品)
780000000 | Beverage | (饮料)
790000000 | Condiment | (调料)
```

#### 食材分类 (Ingredient Categories)
```
310000000 | Vegetable | (蔬菜)
320000000 | Seasoning | (调料)
330000000 | Protein | (蛋白质)
340000000 | Starch | (淀粉类)
350000000 | Dairy | (乳制品)
360000000 | Fruit | (水果)
370000000 | Herb | (香草香料)
380000000 | Oil Fat | (油脂类)
```

#### 难度等级 (Difficulty Levels)
```
610000000 | One Star | (一星) ★
620000000 | Two Star | (二星) ★★
630000000 | Three Star | (三星) ★★★
640000000 | Four Star | (四星) ★★★★
650000000 | Five Star | (五星) ★★★★★
```

#### 烹饪方法 (Cooking Methods)
```
501000000 | Stir Fry | (炒)
502000000 | Deep Fry | (炸)
503000000 | Braise | (红烧)
504000000 | Steam | (蒸)
505000000 | Boil | (煮)
506000000 | Roast | (烤)
507000000 | Stew | (炖)
508000000 | Mix | (拌)
509000000 | Marinate | (腌)
510000000 | Blanch | (焯)
```

### 关系类型定义 (Relationship Types)

#### 核心关系 (Core Relationships)
```
116680003 | is_a | (是一个) - 层次分类关系
```

#### 属性关系 (Attribute Relationships)
```
801000001 | has_ingredient | (包含食材)
801000002 | requires_tool | (需要工具)
801000003 | has_step | (包含步骤)
801000004 | belongs_to_category | (属于分类)
801000005 | has_difficulty | (具有难度)
801000006 | uses_method | (使用方法)
801000007 | has_amount | (具有用量)
801000008 | step_follows | (步骤顺序)
801000009 | serves_people | (供应人数)
801000010 | cooking_time | (烹饪时间)
801000011 | prep_time | (准备时间)
801000012 | ingredient_substitute | (食材替代)
801000013 | recipe_variant | (菜谱变体)
801000014 | nutritional_info | (营养信息)
```

### 具体实例设计

#### 红烧茄子菜谱实例
```
概念ID: 201000001
完全限定名: 201000001 | 红烧茄子 (Braised Eggplant) |
首选术语: 红烧茄子
同义词: 茄子烧制, 红烧青茄子

属性关系:
- belongs_to_category = 710000000 | 素菜
- has_difficulty = 640000000 | 四星
- serves_people = 2人份
- cooking_time = 30分钟
- prep_time = 15分钟

食材关系:
- has_ingredient = 311000001 | 青茄子 | : has_amount = "0.7个/份"
- has_ingredient = 311000002 | 大蒜 | : has_amount = "3瓣"
- has_ingredient = 321000001 | 酱油 | : has_amount = "茄子数量*7克"
- has_ingredient = 331000001 | 鸡蛋 | : has_amount = "1个"
- has_ingredient = 341000001 | 面粉 | : has_amount = "青茄子数量*150克"

工具关系:
- requires_tool = 401000001 | 炒锅
- requires_tool = 401000002 | 菜刀
- requires_tool = 401000003 | 筷子

方法关系:
- uses_method = 502000000 | 炸
- uses_method = 503000000 | 红烧
- uses_method = 501000000 | 炒

步骤关系:
- has_step = S001 | 清洗食材
- has_step = S002 | 切配处理  
- has_step = S003 | 调制面糊
- has_step = S004 | 油炸茄块
- has_step = S005 | 炒制调味
```

### 表达式系统

#### 预协调表达式 (Precoordinated)
```
201000001 | 红烧茄子 |
```

#### 后协调表达式 (Postcoordinated)
```
# 四星难度的素菜
710000000 | 素菜 | : has_difficulty = 640000000 | 四星 |

# 包含茄子的红烧菜谱
200000000 | 菜谱 | : {
    has_ingredient = 311000001 | 青茄子 |,
    uses_method = 503000000 | 红烧 |
}

# 30分钟内完成的四星菜谱
200000000 | 菜谱 | : {
    has_difficulty = 640000000 | 四星 |,
    cooking_time <= 30分钟
}
```

### 数据文件结构

#### 概念文件 (rf2_concept.txt)
```
id	effectiveTime	active	moduleId	definitionStatusId
100000000	20241201	1	900000000	900000000
200000000	20241201	1	900000000	900000000
201000001	20241201	1	900000000	900000000
```

#### 描述文件 (rf2_description.txt)
```
id	effectiveTime	active	moduleId	conceptId	languageCode	typeId	term	caseSignificanceId
D001	20241201	1	900000000	201000001	zh-CN	900000001	红烧茄子	900000000
D002	20241201	1	900000000	201000001	zh-CN	900000002	茄子烧制	900000000
D003	20241201	1	900000000	201000001	en	900000001	Braised Eggplant	900000000
```

#### 关系文件 (rf2_relationship.txt)
```
id	effectiveTime	active	moduleId	sourceId	destinationId	relationshipGroup	typeId	characteristicTypeId	modifierId
R001	20241201	1	900000000	201000001	710000000	0	801000004	900000000	900000000
R002	20241201	1	900000000	201000001	640000000	0	801000005	900000000	900000000
R003	20241201	1	900000000	201000001	311000001	1	801000001	900000000	900000000
```

### 查询示例

#### 1. 基础查询 - 所有素菜
```cypher
MATCH (recipe:Concept)-[:IS_A*]->(category:Concept {conceptId: "710000000"})
RETURN recipe.preferredTerm
```

#### 2. 复杂查询 - 包含特定食材的四星菜谱
```cypher
MATCH (recipe:Concept)-[:HAS_INGREDIENT]->(ingredient:Concept)
WHERE ingredient.conceptId = "311000001" 
AND (recipe)-[:HAS_DIFFICULTY]->(:Concept {conceptId: "640000000"})
RETURN recipe.preferredTerm, recipe.cookingTime
```

#### 3. 语义查询 - 所有炒制类菜谱
```cypher
MATCH (recipe:Concept)-[:USES_METHOD]->(method:Concept)
WHERE (method)-[:IS_A*]->(:Concept {conceptId: "501000000"})
RETURN DISTINCT recipe.preferredTerm
```

### 实现技术栈

#### 图数据库选择
- **Neo4j**: 成熟的图数据库，适合复杂查询
- **ArangoDB**: 多模型数据库，支持图和文档
- **Amazon Neptune**: 云原生图数据库

#### API设计
```python
class RecipeOntologyAPI:
    def search_recipes_by_ingredient(self, ingredient_id: str) -> List[Recipe]:
        """根据食材搜索菜谱"""
        pass
    
    def get_recipe_variants(self, recipe_id: str) -> List[Recipe]:
        """获取菜谱变体"""
        pass
    
    def suggest_substitutes(self, ingredient_id: str) -> List[Ingredient]:
        """建议食材替代"""
        pass
    
    def analyze_nutrition(self, recipe_id: str) -> NutritionInfo:
        """分析营养成分"""
        pass
```

### 应用场景

#### 1. 智能菜谱推荐
- 基于现有食材推荐菜谱
- 根据难度等级筛选
- 营养搭配建议

#### 2. 食材替代建议
- 过敏原替代
- 地域性食材替换
- 营养等价替代

#### 3. 烹饪知识推理
- 步骤优化建议
- 工具使用指导
- 时间管理优化

#### 4. 营养分析
- 卡路里计算
- 营养成分分析
- 膳食搭配建议

### 扩展性设计

#### 多语言支持
```
zh-CN: 红烧茄子
en-US: Braised Eggplant
ja-JP: 茄子の煮物
ko-KR: 가지조림
```

#### 地域化扩展
```
中式烹饪: 701000000
法式烹饪: 702000000
意式烹饪: 703000000
日式烹饪: 704000000
```

#### 个性化标签
```
vegetarian: 素食主义
vegan: 纯素
halal: 清真
kosher: 犹太教食物
gluten_free: 无麸质
```

### 质量控制

#### 概念一致性检查
- 循环依赖检测
- 概念完整性验证
- 关系合理性校验

#### 数据质量保证
- 重复概念检测
- 缺失关系补充
- 术语标准化

这个设计方案提供了一个完整的菜谱知识图谱框架，可以支持复杂的语义查询、推理和知识发现功能。 