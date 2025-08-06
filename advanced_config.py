#!/usr/bin/env python3
"""
Advanced PDF Enhancement Configuration
高级PDF增强配置选项
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

class AdvancedConfig:
    """高级配置管理器"""
    
    def __init__(self, config_file: str = "pdf_enhancement_config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
            "processing": {
                "dpi": 300,
                "quality": 90,
                "device": "auto",
                "batch_size": 1,
                "max_memory_usage": 0.8  # 最大显存使用比例
            },
            "enhancement": {
                "enable_super_resolution": True,
                "enable_denoising": True,
                "enable_demoireing": True,
                "enable_sharpening": True,
                "noise_reduction_strength": 0.1,
                "sharpening_strength": 1.0,
                "contrast_enhancement": 1.1
            },
            "output": {
                "preserve_metadata": True,
                "optimize_size": True,
                "progressive_jpeg": True,
                "max_output_size_ratio": 1.5
            },
            "logging": {
                "log_level": "INFO",
                "save_processing_stats": True,
                "show_progress_bar": True
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # 合并默认配置和加载的配置
                config = self.default_config.copy()
                self._deep_update(config, loaded_config)
                return config
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
        return self.default_config.copy()
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default=None):
        """获取配置值（支持点分隔的路径）"""
        keys = key_path.split('.')
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def set(self, key_path: str, value):
        """设置配置值（支持点分隔的路径）"""
        keys = key_path.split('.')
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def create_sample_config(self):
        """创建示例配置文件"""
        sample_config = {
            "processing": {
                "dpi": 300,
                "quality": 90,
                "device": "auto",  # auto, cuda, cpu
                "batch_size": 1,
                "max_memory_usage": 0.8
            },
            "enhancement": {
                "enable_super_resolution": True,
                "enable_denoising": True,
                "enable_demoireing": True,
                "enable_sharpening": True,
                "noise_reduction_strength": 0.1,  # 0.0-1.0
                "sharpening_strength": 1.0,       # 0.0-2.0
                "contrast_enhancement": 1.1        # 0.5-2.0
            },
            "output": {
                "preserve_metadata": True,
                "optimize_size": True,
                "progressive_jpeg": True,
                "max_output_size_ratio": 1.5
            },
            "logging": {
                "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
                "save_processing_stats": True,
                "show_progress_bar": True
            }
        }
        
        sample_file = Path("pdf_enhancement_config_sample.json")
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        print(f"示例配置文件已创建: {sample_file}")


def main():
    """演示配置使用"""
    print("PDF增强高级配置管理器")
    
    config_manager = AdvancedConfig()
    
    print(f"当前DPI设置: {config_manager.get('processing.dpi')}")
    print(f"当前质量设置: {config_manager.get('processing.quality')}")
    print(f"启用超分辨率: {config_manager.get('enhancement.enable_super_resolution')}")
    
    # 创建示例配置
    config_manager.create_sample_config()
    
    # 修改配置示例
    config_manager.set('processing.dpi', 600)
    config_manager.set('processing.quality', 95)
    
    print("配置已修改:")
    print(f"新DPI设置: {config_manager.get('processing.dpi')}")
    print(f"新质量设置: {config_manager.get('processing.quality')}")
    
    # 保存配置
    config_manager.save_config()


if __name__ == '__main__':
    main()