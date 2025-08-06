# PDF清晰度增强系统配置文件

# 基本设置
DEFAULT_JPEG_QUALITY = 75          # 默认JPEG质量 (50-100)
MAX_SIZE_MULTIPLIER = 1.5          # 最大文件大小倍数
OUTPUT_DPI = 300                   # 输出DPI

# 图像增强设置
ENABLE_SUPER_RESOLUTION = True     # 启用超分辨率
ENABLE_DENOISING = True            # 启用去噪
ENABLE_DEMOIRE = True              # 启用去摩尔纹
ENABLE_WHITE_BALANCE = True        # 启用白平衡校正

# 页面类型检测阈值
COLOR_SATURATION_THRESHOLD = 20    # 彩色/灰度判断阈值
BW_UNIQUE_VALUES_THRESHOLD = 50    # 黑白/灰度判断阈值

# 压缩策略
BW_USE_PNG = True                  # 黑白页面使用PNG
GRAYSCALE_QUALITY_REDUCTION = 15   # 灰度页面质量降低值
ADAPTIVE_QUALITY_STEPS = [10, 20, 0]  # 自适应质量调整步长

# GPU设置
USE_GPU_IF_AVAILABLE = True        # 有GPU时自动使用
GPU_MEMORY_FRACTION = 0.8          # GPU内存使用比例

# 日志设置
LOG_LEVEL = "INFO"                 # 日志级别
PROGRESS_UPDATE_INTERVAL = 5       # 进度更新间隔(页数)
ENABLE_DETAILED_TIMING = True      # 启用详细计时

# 处理优化
MAX_IMAGE_SIZE = 4096              # 最大图像尺寸限制
ENABLE_BATCH_PROCESSING = False    # 批处理模式(暂未实现)
PARALLEL_WORKERS = 1               # 并行工作进程数

# 质量预设
QUALITY_PRESETS = {
    "高质量": {"quality": 95, "max_size": 2.0},
    "平衡": {"quality": 85, "max_size": 1.5},
    "快速": {"quality": 70, "max_size": 1.2},
    "压缩": {"quality": 60, "max_size": 1.0}
}

# 支持的文件格式
SUPPORTED_FORMATS = [".pdf"]
OUTPUT_FORMAT = "pdf"

# 临时文件设置
CLEANUP_TEMP_FILES = True          # 自动清理临时文件
TEMP_FILE_PREFIX = ".tmp"          # 临时文件前缀