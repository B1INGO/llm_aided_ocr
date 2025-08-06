#!/usr/bin/env python3
"""
PDF Enhancement Examples
演示不同场景下的PDF增强用法
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"示例: {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 命令执行成功")
            if result.stdout:
                print(f"输出:\n{result.stdout}")
        else:
            print("❌ 命令执行失败")
            if result.stderr:
                print(f"错误:\n{result.stderr}")
    except Exception as e:
        print(f"❌ 执行异常: {e}")

def main():
    """主函数"""
    print("PDF视觉增强系统 - 使用示例")
    print("确保已安装所有依赖: pip install -r requirements.txt")
    
    # 检查是否有测试文件
    input_dir = Path("input_pdf")
    if not input_dir.exists():
        input_dir.mkdir()
    
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n⚠️  在 {input_dir} 目录中未找到PDF文件")
        print("请先将测试PDF文件放入 input_pdf/ 目录")
        return
    
    print(f"\n📁 发现 {len(pdf_files)} 个PDF文件:")
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        print(f"  - {pdf_file.name} ({size_mb:.1f}MB)")
    
    examples = [
        {
            "cmd": "python enhance_pdf_cli.py --help",
            "desc": "显示帮助信息"
        },
        {
            "cmd": "python enhance_pdf_cli.py --device cpu --dpi 200 --quality 80",
            "desc": "快速预览模式（低质量，快速处理）"
        },
        {
            "cmd": "python enhance_pdf_cli.py --device auto --dpi 300 --quality 90",
            "desc": "标准质量模式（推荐设置）"
        },
        {
            "cmd": "python enhance_pdf_cli.py --device auto --dpi 600 --quality 95",
            "desc": "高质量模式（最佳效果，较慢）"
        }
    ]
    
    print(f"\n🚀 准备运行示例命令...")
    
    for i, example in enumerate(examples, 1):
        user_input = input(f"\n是否运行示例 {i}: {example['desc']}? (y/n/q): ").lower().strip()
        
        if user_input == 'q':
            print("退出示例演示")
            break
        elif user_input == 'y':
            run_command(example["cmd"], example["desc"])
        else:
            print("跳过此示例")
    
    print(f"\n✨ 示例演示完成！")
    print(f"📂 查看输出文件: ls output_pdf/")

if __name__ == '__main__':
    main()