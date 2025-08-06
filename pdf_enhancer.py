"""
PDF清晰度增强系统 - 专为中文电子书扫描文档优化
高质量深度学习超分辨率、去噪、去网纹处理

作者: 电子书修复师
版本: 1.0.0
支持: Windows 命令行，PyTorch CUDA 12.x，RTX 2080Ti
"""

import os
import sys
import time
import glob
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import argparse

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import psutil
import GPUtil
from deep_enhancement import DeepEnhancer, TraditionalEnhancer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pdf_enhance.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class GPUMonitor:
    """GPU显存和使用率监控"""
    
    def __init__(self):
        self.gpus = GPUtil.getGPUs()
        
    def get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        if not self.gpus:
            return {"available": False}
            
        gpu = self.gpus[0]
        return {
            "available": True,
            "name": gpu.name,
            "memory_used": gpu.memoryUsed,
            "memory_total": gpu.memoryTotal,
            "memory_percent": gpu.memoryUtil * 100,
            "load": gpu.load * 100
        }
    
    def log_gpu_status(self):
        """记录GPU状态"""
        info = self.get_gpu_info()
        if info["available"]:
            logger.info(f"GPU: {info['name']} | "
                       f"显存: {info['memory_used']:.0f}/{info['memory_total']:.0f}MB "
                       f"({info['memory_percent']:.1f}%) | "
                       f"使用率: {info['load']:.1f}%")
        else:
            logger.warning("未检测到可用GPU")

class PDFPageProcessor:
    """PDF页面处理器"""
    
    def __init__(self, output_dpi: int = 300):
        self.output_dpi = output_dpi
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
    def extract_pages_as_images(self, pdf_path: str) -> List[Tuple[np.ndarray, Dict]]:
        """提取PDF页面为图像"""
        pages = []
        doc = fitz.open(pdf_path)
        
        logger.info(f"开始提取PDF页面: {pdf_path}")
        logger.info(f"总页数: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 获取页面信息
            page_info = {
                'page_num': page_num + 1,
                'mediabox': page.mediabox,
                'rotation': page.rotation,
                'width': page.mediabox.width,
                'height': page.mediabox.height
            }
            
            # 转换为图像
            mat = fitz.Matrix(self.output_dpi / 72, self.output_dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            
            # 转换为numpy数组
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            pages.append((img_rgb, page_info))
            
        doc.close()
        return pages
    
    def detect_page_type(self, image: np.ndarray) -> str:
        """检测页面类型：彩色/灰度/黑白"""
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 计算饱和度统计
        saturation = hsv[:, :, 1]
        mean_sat = np.mean(saturation)
        
        if mean_sat < 20:
            # 进一步检测是灰度还是黑白
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            unique_values = len(np.unique(gray))
            
            if unique_values < 50:  # 主要是黑白
                return "黑白"
            else:
                return "灰度"
        else:
            return "彩色"

class ImageEnhancer:
    """图像增强处理器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.use_deep_learning = device.type == 'cuda'  # 只在GPU上使用深度学习
        
        if self.use_deep_learning:
            logger.info("使用深度学习增强模型")
            self.deep_enhancer = DeepEnhancer(device)
        else:
            logger.info("使用传统图像处理方法")
            self.traditional_enhancer = TraditionalEnhancer()
        
        logger.info("图像增强器初始化完成")
    
    def enhance_image(self, image: np.ndarray, page_type: str) -> np.ndarray:
        """根据页面类型增强图像"""
        enhanced = image.copy()
        
        # 预处理
        enhanced = self._preprocess_image(enhanced, page_type)
        
        # 主要增强
        if self.use_deep_learning:
            enhanced = self.deep_enhancer.enhance_combined(enhanced, page_type)
        else:
            enhanced = self._traditional_enhance(enhanced, page_type)
        
        # 后处理
        enhanced = self._postprocess_image(enhanced, page_type)
        
        return enhanced
    
    def _preprocess_image(self, image: np.ndarray, page_type: str) -> np.ndarray:
        """预处理图像"""
        # 白平衡校正
        if page_type == "彩色":
            image = self._white_balance_correction(image)
        
        # 轻微降噪
        image = cv2.bilateralFilter(image, 5, 50, 50)
        
        return image
    
    def _white_balance_correction(self, image: np.ndarray) -> np.ndarray:
        """白平衡校正"""
        result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        return result
    
    def _traditional_enhance(self, image: np.ndarray, page_type: str) -> np.ndarray:
        """传统方法增强"""
        if page_type == "黑白":
            # 对黑白文档使用锐化
            enhanced = self.traditional_enhancer.enhance_sharpness(image, strength=0.5)
            enhanced = self.traditional_enhancer.enhance_contrast(enhanced, alpha=1.1)
        else:
            # 对彩色和灰度使用CLAHE和降噪
            enhanced = self.traditional_enhancer.reduce_noise(image)
            enhanced = self.traditional_enhancer.enhance_clahe(enhanced)
            enhanced = self.traditional_enhancer.enhance_contrast(enhanced, alpha=1.05)
        
        return enhanced
    
    def _postprocess_image(self, image: np.ndarray, page_type: str) -> np.ndarray:
        """后处理图像"""
        # 去摩尔纹（轻微高斯模糊）
        if page_type != "黑白":
            image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # 确保值范围正确
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image

class PDFEnhancer:
    """PDF增强主处理器"""
    
    def __init__(self, jpeg_quality: int = 90, max_size_multiplier: float = 1.5):
        self.jpeg_quality = jpeg_quality
        self.max_size_multiplier = max_size_multiplier
        self.gpu_monitor = GPUMonitor()
        self.page_processor = PDFPageProcessor()
        self.image_enhancer = ImageEnhancer(self.page_processor.device)
        
    def enhance_pdf(self, input_path: str, output_path: str) -> bool:
        """增强单个PDF文件"""
        try:
            start_time = time.time()
            original_size = os.path.getsize(input_path)
            
            logger.info(f"开始处理: {input_path}")
            logger.info(f"原始文件大小: {original_size / 1024 / 1024:.2f} MB")
            
            # 提取页面
            pages = self.page_processor.extract_pages_as_images(input_path)
            total_pages = len(pages)
            
            # 处理每一页
            enhanced_pages = []
            
            with tqdm(total=total_pages, desc="处理页面", unit="页") as pbar:
                for i, (image, page_info) in enumerate(pages):
                    page_start = time.time()
                    
                    # 检测页面类型
                    page_type = self.page_processor.detect_page_type(image)
                    
                    # 增强图像
                    enhanced_image = self.image_enhancer.enhance_image(image, page_type)
                    
                    enhanced_pages.append((enhanced_image, page_info, page_type))
                    
                    # 更新进度
                    page_time = time.time() - page_start
                    pbar.set_postfix({
                        '页面': f"{i+1}/{total_pages}",
                        '类型': page_type,
                        '耗时': f"{page_time:.1f}s"
                    })
                    pbar.update(1)
                    
                    # 定期显示GPU状态
                    if i % 5 == 0:
                        self.gpu_monitor.log_gpu_status()
            
            # 自适应重建PDF
            success = self._adaptive_rebuild_pdf(enhanced_pages, output_path, original_size)
            
            if success:
                output_size = os.path.getsize(output_path)
                size_ratio = output_size / original_size
                total_time = time.time() - start_time
                
                logger.info(f"处理完成: {output_path}")
                logger.info(f"输出文件大小: {output_size / 1024 / 1024:.2f} MB")
                logger.info(f"大小比例: {size_ratio:.2f}x")
                logger.info(f"总耗时: {total_time:.1f}秒")
                
                if size_ratio > self.max_size_multiplier:
                    logger.warning(f"输出文件超过大小限制 ({self.max_size_multiplier}x)")
                else:
                    logger.info("文件大小在限制范围内")
            
            return success
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _adaptive_rebuild_pdf(self, enhanced_pages: List[Tuple[np.ndarray, Dict, str]], 
                            output_path: str, original_size: int) -> bool:
        """自适应重建PDF文件，自动调整质量以控制文件大小"""
        quality_levels = [self.jpeg_quality, self.jpeg_quality - 10, self.jpeg_quality - 20, 50]
        
        for attempt, quality in enumerate(quality_levels):
            temp_path = output_path + f".tmp{attempt}"
            
            try:
                self._rebuild_pdf_with_quality(enhanced_pages, temp_path, quality)
                
                # 检查文件大小
                output_size = os.path.getsize(temp_path)
                size_ratio = output_size / original_size
                
                if size_ratio <= self.max_size_multiplier or attempt == len(quality_levels) - 1:
                    # 达到大小要求或最后一次尝试
                    os.rename(temp_path, output_path)
                    if attempt > 0:
                        logger.info(f"自适应调整质量为 {quality}，文件大小比例: {size_ratio:.2f}x")
                    return True
                else:
                    # 删除临时文件，尝试更低质量
                    os.remove(temp_path)
                    logger.info(f"质量 {quality} 输出过大 ({size_ratio:.2f}x)，尝试降低质量")
                    
            except Exception as e:
                logger.error(f"重建失败，质量={quality}: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                continue
        
        return False
    
    def _rebuild_pdf_with_quality(self, enhanced_pages: List[Tuple[np.ndarray, Dict, str]], 
                                output_path: str, quality: int):
        """使用指定质量重建PDF文件"""
        doc = fitz.open()
        
        for enhanced_image, page_info, page_type in enhanced_pages:
            # 转换图像格式
            pil_image = Image.fromarray(enhanced_image)
            
            # 智能压缩策略
            import io
            img_bytes = io.BytesIO()
            
            if page_type == "黑白":
                # 黑白页面使用PNG压缩，转为1位图像
                pil_image = pil_image.convert('1')
                pil_image.save(img_bytes, format='PNG', optimize=True)
            elif page_type == "灰度":
                # 灰度页面使用JPEG，较低质量
                pil_image = pil_image.convert('L')
                gray_quality = max(60, quality - 15)
                pil_image.save(img_bytes, format='JPEG', quality=gray_quality, optimize=True)
            else:
                # 彩色页面使用JPEG压缩
                pil_image = pil_image.convert('RGB')
                pil_image.save(img_bytes, format='JPEG', quality=quality, optimize=True)
            
            img_bytes.seek(0)
            
            # 创建新页面
            page_width = page_info['width']
            page_height = page_info['height']
            page = doc.new_page(width=page_width, height=page_height)
            
            # 插入图像
            rect = fitz.Rect(0, 0, page_width, page_height)
            page.insert_image(rect, stream=img_bytes.getvalue())
        
        # 保存PDF，使用压缩
        doc.save(output_path, garbage=4, deflate=True, clean=True)
        doc.close()
    


def process_directory(input_dir: str, output_dir: str, jpeg_quality: int = 90):
    """处理整个目录"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有PDF文件
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"未在 {input_dir} 中找到PDF文件")
        return
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 创建增强器
    enhancer = PDFEnhancer(jpeg_quality=jpeg_quality)
    
    success_count = 0
    total_count = len(pdf_files)
    
    for pdf_file in pdf_files:
        input_file = str(pdf_file)
        output_file = str(output_path / pdf_file.name)
        
        logger.info(f"处理 {success_count + 1}/{total_count}: {pdf_file.name}")
        
        if enhancer.enhance_pdf(input_file, output_file):
            success_count += 1
        else:
            logger.error(f"处理失败: {pdf_file.name}")
    
    logger.info(f"处理完成: {success_count}/{total_count} 个文件成功")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PDF清晰度增强系统 - 专为中文电子书优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python pdf_enhancer.py                          # 处理默认目录
  python pdf_enhancer.py --quality 85             # 设置JPEG质量
  python pdf_enhancer.py --input custom_input     # 自定义输入目录
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        default="提高扫描PDF清晰度/input_pdf",
        help="输入PDF目录 (默认: 提高扫描PDF清晰度/input_pdf)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="提高扫描PDF清晰度/output_pdf",
        help="输出PDF目录 (默认: 提高扫描PDF清晰度/output_pdf)"
    )
    
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=90,
        choices=range(50, 101),
        help="JPEG压缩质量 (50-100, 默认: 90)"
    )
    
    parser.add_argument(
        "--max-size",
        type=float,
        default=1.5,
        help="最大文件大小倍数 (默认: 1.5)"
    )
    
    args = parser.parse_args()
    
    # 显示系统信息
    logger.info("=" * 60)
    logger.info("PDF清晰度增强系统 v1.0.0")
    logger.info("专为中文电子书扫描文档优化")
    logger.info("=" * 60)
    
    # GPU信息
    gpu_monitor = GPUMonitor()
    gpu_monitor.log_gpu_status()
    
    # 系统信息
    logger.info(f"系统: {os.name}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU设备: {torch.cuda.get_device_name()}")
    
    logger.info("-" * 60)
    logger.info(f"输入目录: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"JPEG质量: {args.quality}")
    logger.info(f"最大文件大小: {args.max_size}x")
    logger.info("-" * 60)
    
    # 检查输入目录
    if not os.path.exists(args.input):
        logger.error(f"输入目录不存在: {args.input}")
        logger.info("正在创建目录...")
        os.makedirs(args.input, exist_ok=True)
        logger.info(f"请将PDF文件放入 {args.input} 目录后重新运行")
        return
    
    # 开始处理
    start_time = time.time()
    process_directory(args.input, args.output, args.quality)
    total_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info(f"全部处理完成，总耗时: {total_time:.1f}秒")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()