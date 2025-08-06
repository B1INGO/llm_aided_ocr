"""
批量PDF增强脚本
支持多目录、多任务并行处理
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
import concurrent.futures
from datetime import datetime

def process_single_file(input_file: str, output_file: str, quality: int = 75) -> Dict:
    """处理单个文件"""
    from pdf_enhancer import PDFEnhancer
    
    start_time = datetime.now()
    
    try:
        enhancer = PDFEnhancer(jpeg_quality=quality)
        success = enhancer.enhance_pdf(input_file, output_file)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success and os.path.exists(output_file):
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            size_ratio = output_size / input_size
            
            return {
                "status": "success",
                "input_file": input_file,
                "output_file": output_file,
                "input_size_mb": input_size / 1024 / 1024,
                "output_size_mb": output_size / 1024 / 1024,
                "size_ratio": size_ratio,
                "duration_seconds": duration,
                "quality_used": quality
            }
        else:
            return {
                "status": "failed",
                "input_file": input_file,
                "error": "Enhancement failed",
                "duration_seconds": duration
            }
            
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "status": "error",
            "input_file": input_file,
            "error": str(e),
            "duration_seconds": duration
        }

def batch_process(input_dirs: List[str], output_dir: str, 
                 max_workers: int = 2, quality: int = 75) -> Dict:
    """批量处理多个目录"""
    
    # 收集所有PDF文件
    all_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if input_path.exists():
            pdf_files = list(input_path.glob("*.pdf"))
            for pdf_file in pdf_files:
                output_file = Path(output_dir) / pdf_file.name
                all_files.append((str(pdf_file), str(output_file)))
    
    print(f"找到 {len(all_files)} 个PDF文件待处理")
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 并行处理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file, input_file, output_file, quality): 
            (input_file, output_file)
            for input_file, output_file in all_files
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            results.append(result)
            
            # 显示进度
            if result["status"] == "success":
                print(f"✓ {Path(result['input_file']).name} -> {result['output_size_mb']:.2f}MB "
                      f"({result['size_ratio']:.2f}x, {result['duration_seconds']:.1f}s)")
            else:
                print(f"✗ {Path(result['input_file']).name} - {result.get('error', 'Failed')}")
    
    # 统计结果
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    summary = {
        "total_files": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0,
        "total_duration": sum(r["duration_seconds"] for r in results),
        "results": results
    }
    
    if successful:
        avg_size_ratio = sum(r["size_ratio"] for r in successful) / len(successful)
        total_input_size = sum(r["input_size_mb"] for r in successful)
        total_output_size = sum(r["output_size_mb"] for r in successful)
        
        summary.update({
            "average_size_ratio": avg_size_ratio,
            "total_input_size_mb": total_input_size,
            "total_output_size_mb": total_output_size
        })
    
    return summary

def create_quality_comparison(input_file: str, output_dir: str):
    """创建不同质量设置的对比"""
    qualities = [95, 85, 75, 65, 55, 45]
    
    print(f"创建质量对比: {input_file}")
    results = []
    
    for quality in qualities:
        output_file = f"{output_dir}/quality_{quality}_{Path(input_file).name}"
        result = process_single_file(input_file, output_file, quality)
        results.append(result)
        
        if result["status"] == "success":
            print(f"质量 {quality}: {result['output_size_mb']:.2f}MB ({result['size_ratio']:.2f}x)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="PDF批量增强处理")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 批量处理命令
    batch_parser = subparsers.add_parser("batch", help="批量处理多个目录")
    batch_parser.add_argument("input_dirs", nargs="+", help="输入目录列表")
    batch_parser.add_argument("--output", "-o", required=True, help="输出目录")
    batch_parser.add_argument("--workers", "-w", type=int, default=2, help="并行工作数")
    batch_parser.add_argument("--quality", "-q", type=int, default=75, help="JPEG质量")
    batch_parser.add_argument("--report", help="生成JSON报告文件")
    
    # 质量对比命令
    compare_parser = subparsers.add_parser("compare", help="创建质量对比")
    compare_parser.add_argument("input_file", help="输入PDF文件")
    compare_parser.add_argument("--output", "-o", required=True, help="输出目录")
    compare_parser.add_argument("--report", help="生成JSON报告文件")
    
    args = parser.parse_args()
    
    if args.command == "batch":
        print("PDF批量增强处理")
        print("=" * 50)
        
        summary = batch_process(
            args.input_dirs, 
            args.output, 
            args.workers, 
            args.quality
        )
        
        print("\n处理完成！")
        print(f"总文件数: {summary['total_files']}")
        print(f"成功: {summary['successful']}")
        print(f"失败: {summary['failed']}")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"总耗时: {summary['total_duration']:.1f}秒")
        
        if "average_size_ratio" in summary:
            print(f"平均大小比例: {summary['average_size_ratio']:.2f}x")
            print(f"总输入大小: {summary['total_input_size_mb']:.2f}MB")
            print(f"总输出大小: {summary['total_output_size_mb']:.2f}MB")
        
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"报告已保存: {args.report}")
    
    elif args.command == "compare":
        print("质量对比测试")
        print("=" * 50)
        
        Path(args.output).mkdir(parents=True, exist_ok=True)
        results = create_quality_comparison(args.input_file, args.output)
        
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"对比报告已保存: {args.report}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()