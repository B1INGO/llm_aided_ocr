# PDF清晰度增强系统

专为中文电子书扫描文档优化的高质量PDF增强工具，使用深度学习超分辨率、去噪、去网纹技术。

## 功能特点

### 🚀 核心功能
- **深度学习增强**: 超分辨率、去噪、去网纹等AI技术
- **智能页面检测**: 自动识别彩色/灰度/黑白页面类型
- **自适应压缩**: 智能调整质量以控制文件大小
- **批量处理**: 支持整个目录的PDF文件批量增强
- **实时监控**: 显示处理进度、GPU使用情况、内存占用

### 📊 处理效果
- **文字边缘**: 干净、无锯齿、无墨迹飞白
- **公式线条**: 连续、无断笔
- **插图质量**: 无网纹、无摩尔纹、噪点可控
- **文件大小**: ≤ 1.5倍原始大小（可调节）

### 🎯 技术特色
- **GPU加速**: 支持CUDA 12.x，优化RTX 2080Ti
- **传统备选**: CPU环境下使用优化的传统算法
- **中文优化**: 专为中文文档设计的增强策略
- **Windows兼容**: 完全命令行操作，无GUI依赖

## 安装说明

### 系统要求
- Python 3.12+
- Windows/Linux操作系统
- 可选：NVIDIA GPU (CUDA 12.x)
- 内存：建议8GB+

### 依赖安装

1. **基础依赖**
```bash
pip install -r requirements.txt
```

2. **PDF增强专用依赖**
```bash
pip install -r pdf_enhancer_requirements.txt
```

3. **GPU支持** (可选)
```bash
# 确保已安装CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 使用方法

### 📁 目录结构
```
提高扫描PDF清晰度/
├── input_pdf/          # 输入PDF文件目录
│   ├── 文档1.pdf
│   ├── 文档2.pdf
│   └── ...
└── output_pdf/         # 输出增强PDF目录
    ├── 文档1.pdf
    ├── 文档2.pdf
    └── ...
```

### 🚀 快速开始

1. **放置文件**
```bash
# 将要处理的PDF文件放入input_pdf目录
cp your_ebook.pdf 提高扫描PDF清晰度/input_pdf/
```

2. **运行处理**
```bash
# 使用默认设置
python pdf_enhancer.py

# 自定义质量
python pdf_enhancer.py --quality 85

# 自定义输入输出目录
python pdf_enhancer.py --input custom_input --output custom_output
```

3. **检查结果**
```bash
# 增强后的PDF会出现在output_pdf目录中
ls 提高扫描PDF清晰度/output_pdf/
```

### ⚙️ 命令行参数

```bash
python pdf_enhancer.py [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input, -i` | 输入PDF目录 | `提高扫描PDF清晰度/input_pdf` |
| `--output, -o` | 输出PDF目录 | `提高扫描PDF清晰度/output_pdf` |
| `--quality, -q` | JPEG压缩质量 (50-100) | `90` |
| `--max-size` | 最大文件大小倍数 | `1.5` |
| `--help, -h` | 显示帮助信息 | - |

### 💡 使用示例

```bash
# 高质量处理，允许文件变大
python pdf_enhancer.py --quality 95 --max-size 2.0

# 快速处理，控制文件大小
python pdf_enhancer.py --quality 75 --max-size 1.2

# 处理特定目录
python pdf_enhancer.py --input ~/Documents/scans --output ~/Documents/enhanced

# 查看详细帮助
python pdf_enhancer.py --help
```

## 处理流程

### 🔄 工作原理

1. **页面提取**: 将PDF转换为高分辨率图像
2. **类型识别**: 自动检测页面类型（彩色/灰度/黑白）
3. **预处理**: 白平衡校正、基础降噪
4. **AI增强**: 
   - 彩色页面: 去摩尔纹 → 去噪 → 轻微超分
   - 灰度页面: 去噪 → 超分辨率
   - 黑白页面: 锐化 → 对比度增强
5. **后处理**: 最终优化和格式转换
6. **重建PDF**: 智能压缩，保持原始结构

### 📈 增强策略

| 页面类型 | 主要处理 | 压缩方式 |
|----------|----------|----------|
| **彩色** | 去摩尔纹、去噪、色彩优化 | JPEG压缩 |
| **灰度** | 去噪、超分辨率、对比度增强 | JPEG低质量 |
| **黑白** | 锐化、边缘优化 | PNG 1位压缩 |

## 性能优化

### 🎛️ 质量设置建议

| 用途 | 质量设置 | 预期效果 |
|------|----------|----------|
| **高质量归档** | `--quality 95 --max-size 2.0` | 最佳视觉效果 |
| **平衡模式** | `--quality 85 --max-size 1.5` | 质量与大小平衡 |
| **快速处理** | `--quality 70 --max-size 1.2` | 快速，文件较小 |

### 🔧 硬件优化

**GPU加速** (推荐)
- RTX 2080Ti: 最佳性能
- GTX 1060+: 良好性能
- 无GPU: 自动使用CPU优化算法

**内存管理**
- 大文件: 自动分块处理
- 并行处理: 根据硬件自适应

## 输出日志解读

### 📊 进度显示
```
处理页面: 50%|████████████▌ | 10/20 [00:15<00:15, 页面=10/20, 类型=灰度, 耗时=0.6s]
```
- 处理进度百分比
- 已处理/总页数
- 预计剩余时间
- 当前页面类型
- 单页处理时间

### 💾 文件大小监控
```
2025-08-06 04:47:40,771 - INFO - 自适应调整质量为 70，文件大小比例: 1.2x
```
- 自动质量调整
- 文件大小比例
- 是否符合限制

### 🖥️ GPU状态
```
2025-08-06 04:47:15,315 - INFO - GPU: RTX 2080Ti | 显存: 2048/11264MB (18.2%) | 使用率: 45.3%
```
- GPU型号
- 显存使用情况
- GPU使用率

## 故障排除

### ❗ 常见问题

**1. 文件大小过大**
```bash
# 降低质量设置
python pdf_enhancer.py --quality 60 --max-size 1.2
```

**2. 处理速度慢**
```bash
# 检查GPU状态，确保CUDA可用
python -c "import torch; print(torch.cuda.is_available())"
```

**3. 内存不足**
```bash
# 关闭其他程序，或使用更小的输入文件
```

**4. 依赖缺失**
```bash
# 重新安装依赖
pip install -r pdf_enhancer_requirements.txt --force-reinstall
```

### 🔍 调试模式

查看详细日志:
```bash
# 日志文件：pdf_enhance.log
tail -f pdf_enhance.log
```

### 📋 系统检查

运行演示脚本检查环境:
```bash
python demo.py
```

## 技术原理

### 🧠 深度学习模型

**超分辨率网络 (EnhancementNet)**
- 基于RRDB块设计
- 2x/4x分辨率提升
- 保持细节不失真

**去噪网络 (DenoiseNet)**
- 编码器-解码器结构
- 多尺度特征提取
- 保留文字清晰度

**去摩尔纹网络 (DemoireNet)**
- 多尺度卷积核
- 频域特征分析
- 残差连接保持原始信息

### 🔬 传统算法备选

当无GPU时，使用优化的传统方法:
- CLAHE对比度增强
- 双边滤波降噪
- Unsharp Mask锐化
- 形态学操作

## 开发说明

### 📁 代码结构
```
pdf_enhancer.py         # 主处理程序
deep_enhancement.py     # 深度学习模型
demo.py                 # 演示脚本
pdf_enhancer_requirements.txt  # 依赖列表
```

### 🔧 自定义增强

可以通过修改 `deep_enhancement.py` 中的模型参数来调整增强效果:

```python
# 调整超分辨率强度
class EnhancementNet(nn.Module):
    def __init__(self, scale=2):  # 修改倍数
        # ...
```

### 🚀 性能调优

1. **批处理大小**: 根据GPU内存调整
2. **模型精度**: 可选FP16以提升速度
3. **并行处理**: 多GPU支持 (待实现)

## 许可证

本项目采用 MIT 许可证，详见 `LICENSE` 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

**PDF清晰度增强系统 v1.0.0**  
专为中文电子书修复师打造 | 追求完美的扫描文档增强工具