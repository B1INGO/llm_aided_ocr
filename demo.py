"""
PDF增强系统演示脚本
创建示例文件并展示处理效果
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_pdf(output_path: str):
    """创建一个示例PDF文件用于测试"""
    doc = fitz.open()
    
    # 创建几个不同类型的页面
    pages_data = [
        {
            "type": "黑白文字",
            "content": "这是一个黑白文字页面示例\n包含中文文字内容\n用于测试文字增强效果"
        },
        {
            "type": "灰度图像", 
            "content": "这是一个灰度页面示例\n包含灰度图像和文字\n用于测试灰度增强"
        },
        {
            "type": "彩色内容",
            "content": "这是一个彩色页面示例\n包含彩色内容和图像\n用于测试彩色增强"
        }
    ]
    
    for i, page_data in enumerate(pages_data):
        # 创建页面
        page = doc.new_page(width=595, height=842)  # A4 尺寸
        
        # 添加文字内容
        text_rect = fitz.Rect(50, 50, 545, 792)
        page.insert_text(
            (60, 100),
            f"页面 {i+1}: {page_data['type']}\n\n{page_data['content']}",
            fontsize=16,
            color=(0, 0, 0)
        )
        
        # 添加一些噪点模拟扫描效果
        for _ in range(50):
            x = np.random.randint(50, 545)
            y = np.random.randint(200, 700)
            size = np.random.randint(1, 3)
            page.draw_circle(fitz.Point(x, y), size, color=(0.8, 0.8, 0.8))
    
    # 保存PDF
    doc.save(output_path)
    doc.close()
    print(f"示例PDF已创建: {output_path}")

def main():
    """主演示函数"""
    print("PDF增强系统演示")
    print("=" * 50)
    
    # 确保目录存在
    input_dir = Path("提高扫描PDF清晰度/input_pdf")
    output_dir = Path("提高扫描PDF清晰度/output_pdf")
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例文件
    sample_file = input_dir / "sample_ebook.pdf"
    if not sample_file.exists():
        create_sample_pdf(str(sample_file))
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"示例文件: {sample_file}")
    
    # 检查现有文件
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 个PDF文件:")
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        print(f"  - {pdf_file.name} ({size_mb:.2f} MB)")
    
    print("\n运行命令示例:")
    print("python pdf_enhancer.py                    # 使用默认设置")
    print("python pdf_enhancer.py --quality 70      # 设置质量为70")
    print("python pdf_enhancer.py --max-size 2.0    # 允许文件大小为原来的2倍")
    
    print("\n开始处理...")
    
    # 导入并运行处理器
    try:
        from pdf_enhancer import process_directory
        process_directory(str(input_dir), str(output_dir), jpeg_quality=75)
        
        print("\n处理完成！检查输出文件:")
        output_files = list(output_dir.glob("*.pdf"))
        for output_file in output_files:
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"  - {output_file.name} ({size_mb:.2f} MB)")
            
    except Exception as e:
        print(f"处理出错: {e}")
        print("请确保已安装所有依赖：pip install -r pdf_enhancer_requirements.txt")

if __name__ == "__main__":
    main()