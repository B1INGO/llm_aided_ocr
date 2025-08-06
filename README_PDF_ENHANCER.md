# PDF视觉增强系统

## 概述

这是一个专业的PDF视觉质量增强工具，旨在将扫描PDF转换为"视觉上几乎等于原版印刷"的清晰PDF。该系统专门针对电子书修复场景设计，追求完美的排版、字体、留白和灰阶过渡效果。

## 主要特性

### 🎯 核心目标
- **首要目标**: 清晰度（≠ 高锐度），还原原版纸张质感、笔画细节、灰阶层次
- **次级目标**: 文件体积尽量小（≤ 1.5倍原始大小）

### 🔧 技术特性
- **深度学习增强**: 基于PyTorch的超分辨率、去噪、去网纹算法
- **传统图像处理**: 轻量级白平衡、纠偏、锐化算法作为辅助
- **GPU加速**: 支持CUDA 12.x（2080Ti等GPU）
- **CPU回退**: 自动检测并回退到CPU处理
- **实时监控**: 显示当前页/总页、耗时、显存占用

### 📁 输入输出
- **输入**: `input_pdf/*` 目录中的PDF文件（彩色/黑白混排）
- **输出**: `output_pdf/*` 目录中的增强PDF文件（同名保存）
- **支持格式**: 300+ DPI扫描的彩页、灰度页、黑白页
- **内容兼容**: 文字、公式、曲线图、半色调插图

## 安装要求

### 系统要求
- 操作系统: Windows/Linux/macOS
- Python: 3.8+
- 内存: 4GB+ 推荐
- 硬盘: 足够存储临时图像文件

### 软件依赖
```bash
# 必须安装 poppler-utils (用于PDF转换)
# Ubuntu/Debian:
sudo apt-get install poppler-utils

# macOS:
brew install poppler

# Windows: 下载poppler并添加到PATH
```

### Python依赖
```bash
pip install -r requirements.txt
```

主要包括：
- `torch` - PyTorch深度学习框架
- `torchvision` - 图像处理工具
- `opencv-python` - 计算机视觉库
- `pdf2image` - PDF转图像转换
- `pillow` - 图像处理库
- `scikit-image` - 科学图像处理
- `tqdm` - 进度条显示
- `psutil` - 系统资源监控

## 使用方法

### 基本用法

1. **准备文件**
   ```bash
   # 将需要处理的PDF文件放入input_pdf目录
   cp your_document.pdf input_pdf/
   ```

2. **运行处理**
   ```bash
   # 使用默认设置
   python enhance_pdf_cli.py
   
   # 或者指定参数
   python enhance_pdf_cli.py --dpi 300 --quality 90 --device auto
   ```

3. **获取结果**
   ```bash
   # 处理完成后，增强的PDF保存在output_pdf目录
   ls output_pdf/
   ```

### 命令行参数

```bash
python enhance_pdf_cli.py [选项]

选项:
  --input_dir INPUT_DIR    输入PDF目录 (默认: input_pdf)
  --output_dir OUTPUT_DIR  输出PDF目录 (默认: output_pdf)
  --dpi DPI               处理DPI (默认: 300, 推荐: 300-600)
  --quality QUALITY       JPEG质量 (50-100, 默认: 90)
  --device DEVICE         计算设备 (auto/cuda/cpu, 默认: auto)
```

### 使用示例

```bash
# 默认设置处理
python enhance_pdf_cli.py

# 高质量处理（更慢但效果更好）
python enhance_pdf_cli.py --dpi 600 --quality 95

# 快速处理（适合批量预览）
python enhance_pdf_cli.py --dpi 200 --quality 80

# 强制使用CPU（兼容性最好）
python enhance_pdf_cli.py --device cpu

# 自定义目录
python enhance_pdf_cli.py --input_dir my_pdfs --output_dir enhanced_pdfs
```

## 技术原理

### 图像增强流程

1. **PDF分解**: 将PDF按页转换为高DPI图像
2. **去摩尔纹**: 使用高斯滤波和双边滤波去除扫描摩尔纹
3. **深度学习增强**: 
   - 超分辨率CNN模型提升图像分辨率
   - 文档增强模型优化文字清晰度
   - 去噪网络降低图像噪声
4. **传统增强**: 自适应直方图均衡化、锐化滤波
5. **PDF重建**: 将处理后的图像重新组合为PDF

### 深度学习模型

#### SRCNN超分辨率模型
- 9层卷积神经网络
- 特征提取 → 非线性映射 → 图像重建
- 专门优化文档图像的边缘清晰度

#### 文档增强模型
- 双分支结构：去噪分支 + 细节增强分支
- 特征融合层智能合并处理结果
- 保持文字边缘锐利的同时去除背景噪声

### 性能优化

- **GPU加速**: 自动检测CUDA并使用GPU处理
- **内存管理**: 分页处理避免内存溢出
- **批量优化**: 针对批量文件处理的缓存策略
- **渐进式处理**: 支持中断和续传

## 输出质量

### 视觉效果
- ✅ 文字边缘干净、无锯齿、无墨迹飞白
- ✅ 公式线条连续、无断笔
- ✅ 插图无网纹、无摩尔纹、噪点可控
- ✅ 保持原始页面布局和比例

### 文件特性
- 📏 页面顺序、页码、尺寸、DPI与原件一致
- 📦 文件大小≤ 1.5倍原始大小（通常0.5-1.2倍）
- 🎨 支持彩色、灰度、黑白混合页面
- 📖 保持PDF格式兼容性

## 故障排除

### 常见问题

1. **"pdftoppm not found"错误**
   ```bash
   # 安装poppler-utils
   sudo apt-get install poppler-utils  # Ubuntu
   brew install poppler                 # macOS
   ```

2. **CUDA内存不足**
   ```bash
   # 降低DPI或使用CPU
   python enhance_pdf_cli.py --device cpu --dpi 200
   ```

3. **处理速度慢**
   ```bash
   # 降低DPI和质量设置
   python enhance_pdf_cli.py --dpi 200 --quality 80
   ```

4. **输出文件太大**
   ```bash
   # 降低质量设置
   python enhance_pdf_cli.py --quality 70
   ```

### 性能调优

- **DPI设置**: 300DPI适合大多数场景，600DPI用于高质量需求
- **质量设置**: 90适合大多数场景，95用于追求极致质量
- **设备选择**: GPU > CPU，但CPU兼容性更好

## 开发信息

### 代码结构
```
pdf_enhancer.py         # 核心增强算法
enhance_pdf_cli.py      # 命令行接口
requirements.txt        # 依赖包列表
README_PDF_ENHANCER.md  # 本文档
```

### 扩展开发
- 模型文件可以替换为更先进的预训练模型
- 支持添加自定义图像处理算法
- 可以集成更多深度学习框架（TensorFlow等）

## 许可证

本项目遵循MIT许可证，详见LICENSE文件。

---

**注意**: 本工具专为文档图像优化设计，不适用于照片或艺术作品的处理。处理时间取决于文档页数、DPI设置和硬件性能。