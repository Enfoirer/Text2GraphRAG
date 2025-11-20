#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用量标准化工具 - 处理中文菜谱中的非数值用量表达
"""

import re
from typing import Tuple, Optional

class AmountNormalizer:
    """用量标准化器"""
    
    def __init__(self):
        # 标准化映射表
        self.amount_mappings = {
            # 适量类
            "适量": "适量",
            "适当": "适量", 
            "酌量": "适量",
            "随意": "适量",
            "看个人喜好": "适量",
            
            # 少量类
            "少许": "少许",
            "少量": "少许",
            "一点点": "少许",
            "微量": "少许",
            "稍许": "少许",
            "稍微": "少许",
            
            # 中等量类
            "中量": "中量",
            "适中": "中量",
            "正常": "中量",
            
            # 大量类
            "大量": "大量",
            "足量": "大量",
            "充足": "大量",
            
            # 数量词类
            "一把": "1把",
            "小把": "小把",
            "大把": "大把",
            "一勺": "1勺",
            "一小勺": "1小勺",
            "一大勺": "1大勺",
            "一茶匙": "1茶匙",
            "一汤匙": "1汤匙",
            "一撮": "1撮",
            
            # 滴类
            "几滴": "几滴",
            "数滴": "几滴",
            "2-3滴": "几滴",
            
            # 片/个/根类
            "几片": "几片",
            "数片": "几片", 
            "几个": "几个",
            "数个": "几个",
            "几根": "几根",
            "数根": "几根",
            "几颗": "几颗",
            "数颗": "几颗"
        }
        
        # 估算数值映射（用于营养计算等）
        self.estimated_values = {
            "适量": None,        # 无法估算
            "少许": 2,          # 约2g/ml
            "中量": 10,         # 约10g/ml
            "大量": 30,         # 约30g/ml
            "1把": 15,          # 约15g
            "小把": 10,         # 约10g
            "大把": 20,         # 约20g
            "1勺": 5,           # 约5ml
            "1小勺": 3,         # 约3ml
            "1大勺": 15,        # 约15ml
            "1茶匙": 5,         # 约5ml
            "1汤匙": 15,        # 约15ml
            "1撮": 1,           # 约1g
            "几滴": 1,          # 约1ml
            "几片": 3,          # 约3片
            "几个": 3,          # 约3个
            "几根": 3,          # 约3根
            "几颗": 3           # 约3颗
        }
    
    def normalize_amount(self, amount: str, unit: str = "") -> Tuple[str, Optional[float]]:
        """
        标准化用量表达
        
        Args:
            amount: 原始用量
            unit: 单位
            
        Returns:
            Tuple[标准化后的用量, 估算数值]
        """
        if not amount:
            return "", None
            
        # 清理输入
        amount = amount.strip()
        
        # 尝试提取数字
        number_match = re.match(r'^(\d+(?:\.\d+)?)', amount)
        if number_match:
            # 如果是纯数字，直接返回
            try:
                numeric_value = float(number_match.group(1))
                return amount, numeric_value
            except ValueError:
                pass
        
        # 标准化非数值表达
        normalized = self.amount_mappings.get(amount, amount)
        estimated = self.estimated_values.get(normalized, None)
        
        return normalized, estimated
    
    def parse_amount_with_unit(self, amount_str: str) -> Tuple[str, str, Optional[float]]:
        """
        解析包含单位的用量字符串
        
        Args:
            amount_str: 如"300毫升"、"适量盐"等
            
        Returns:
            Tuple[数量, 单位, 估算数值]
        """
        if not amount_str:
            return "", "", None
        
        # 常见单位模式
        unit_patterns = [
            r'(\d+(?:\.\d+)?)\s*(克|千克|斤|两|钱)',      # 重量
            r'(\d+(?:\.\d+)?)\s*(毫升|升|杯|勺|茶匙|汤匙)', # 体积  
            r'(\d+(?:\.\d+)?)\s*(个|只|条|片|根|瓣|颗)',   # 数量
            r'(\d+(?:\.\d+)?)\s*(把|撮|滴)',             # 特殊
        ]
        
        # 尝试匹配数字+单位
        for pattern in unit_patterns:
            match = re.search(pattern, amount_str)
            if match:
                amount = match.group(1)
                unit = match.group(2)
                try:
                    numeric_value = float(amount)
                    return amount, unit, numeric_value
                except ValueError:
                    pass
        
        # 处理纯文字表达
        normalized, estimated = self.normalize_amount(amount_str)
        return normalized, "", estimated
    
    def get_comparable_value(self, amount: str, unit: str = "") -> Optional[float]:
        """
        获取可比较的数值（用于排序、筛选等）
        
        Args:
            amount: 用量
            unit: 单位
            
        Returns:
            可比较的数值，None表示无法比较
        """
        # 尝试解析为数字
        try:
            return float(amount)
        except ValueError:
            pass
        
        # 使用估算值
        normalized, estimated = self.normalize_amount(amount, unit)
        return estimated
    
    def format_for_display(self, amount: str, unit: str = "") -> str:
        """
        格式化用于显示的用量字符串
        
        Args:
            amount: 用量
            unit: 单位
            
        Returns:
            格式化后的字符串
        """
        normalized, _ = self.normalize_amount(amount, unit)
        
        if unit and normalized:
            # 如果有单位且不是特殊表达（如"适量"已经很完整）
            if normalized not in ["适量", "少许", "中量", "大量"]:
                return f"{normalized} {unit}"
            else:
                return normalized
        else:
            return normalized or amount

# 使用示例
def demo_normalization():
    """演示标准化功能"""
    normalizer = AmountNormalizer()
    
    test_cases = [
        ("适量", "毫升"),
        ("少许", "克"),
        ("一把", ""),
        ("300", "毫升"),
        ("几滴", ""),
        ("酌量", ""),
        ("2-3滴", ""),
        ("一小勺", ""),
    ]

    pass

if __name__ == "__main__":
    demo_normalization() 