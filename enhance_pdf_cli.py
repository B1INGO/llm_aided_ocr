#!/usr/bin/env python3
"""
PDF视觉增强命令行工具
提高扫描PDF清晰度 - 专业电子书修复系统

使用方法:
    python enhance_pdf_cli.py
    python enhance_pdf_cli.py --input_dir input_pdf --output_dir output_pdf
    python enhance_pdf_cli.py --dpi 600 --quality 95 --device cuda
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from pdf_enhancer import PDFEnhancer
except ImportError as e:
    logging.error(f"导入pdf_enhancer模块失败: {e}")
    sys.exit(1)


def setup_logging():
    """设置日志格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_directories(input_dir: str, output_dir: str):
    """检查和创建目录"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 检查输入目录
    if not input_path.exists():
        input_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"创建输入目录: {input_path}")
    
    # 创建输出目录
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"创建输出目录: {output_path}")
    
    return input_path, output_path


def main():
    """主函数"""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description='PDF视觉增强工具 - 提升扫描PDF清晰度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s                                    # 使用默认设置
  %(prog)s --dpi 600 --quality 95           # 高质量处理
  %(prog)s --input_dir my_pdfs --device cpu # 指定输入目录和CPU处理
  
支持的文件格式: PDF
输出格式: PDF (保持原始格式)
        """
    )
    
    parser.add_argument(
        '--input_dir', 
        default='input_pdf',
        help='输入PDF文件目录 (默认: input_pdf)'
    )
    
    parser.add_argument(
        '--output_dir',
        default='output_pdf', 
        help='输出PDF文件目录 (默认: output_pdf)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='处理DPI分辨率 (默认: 300, 推荐: 300-600)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=90,
        choices=range(50, 101),
        help='JPEG压缩质量 (50-100, 默认: 90)'
    )
    
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='计算设备 (默认: auto - 自动检测GPU)'
    )
    
    args = parser.parse_args()
    
    # 显示配置信息
    logging.info("=" * 60)
    logging.info("PDF视觉增强工具启动")
    logging.info("=" * 60)
    logging.info(f"输入目录: {args.input_dir}")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"处理DPI: {args.dpi}")
    logging.info(f"图像质量: {args.quality}")
    logging.info(f"计算设备: {args.device}")
    logging.info("=" * 60)
    
    try:
        # 检查和创建目录
        input_path, output_path = check_directories(args.input_dir, args.output_dir)
        
        # 检查输入文件
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            logging.warning(f"在目录 '{args.input_dir}' 中未找到PDF文件")
            logging.info("请将需要处理的PDF文件放入输入目录中")
            return
        
        logging.info(f"发现 {len(pdf_files)} 个PDF文件:")
        for pdf_file in pdf_files:
            file_size = pdf_file.stat().st_size / 1024 / 1024
            logging.info(f"  - {pdf_file.name} ({file_size:.1f}MB)")
        
        # 创建PDF增强器
        logging.info("初始化PDF增强器...")
        enhancer = PDFEnhancer(device=args.device, quality=args.quality)
        
        # 开始批量处理
        logging.info("开始批量处理...")
        enhancer.batch_enhance_pdfs(str(input_path), str(output_path), args.dpi)
        
        logging.info("=" * 60)
        logging.info("处理完成！增强后的PDF文件保存在: " + str(output_path))
        logging.info("=" * 60)
        
    except KeyboardInterrupt:
        logging.info("用户中断处理")
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()