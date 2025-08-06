"""
深度学习图像增强模型
实现超分辨率、去噪、去网纹等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional

class RRDBBlock(nn.Module):
    """RRDB块，用于超分辨率网络"""
    
    def __init__(self, num_features=64, num_blocks=3, num_grow_ch=32):
        super(RRDBBlock, self).__init__()
        self.blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(num_features + i * num_grow_ch, num_grow_ch, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_grow_ch, num_grow_ch, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # 最后的1x1卷积
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        out = x
        for block in self.blocks:
            out = torch.cat([out, block(out)], 1)
        
        return out * 0.2 + x

class EnhancementNet(nn.Module):
    """图像增强网络"""
    
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=16):
        super(EnhancementNet, self).__init__()
        self.scale = scale
        
        # 第一层卷积
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # RRDB块
        self.body = nn.ModuleList()
        for i in range(num_block):
            self.body.append(RRDBBlock(num_feat))
        
        # 特征融合
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # 上采样
        if scale == 2:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
        elif scale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)
        
        # 输出层
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        body_feat = feat
        
        for block in self.body:
            body_feat = block(body_feat)
        
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        # 上采样
        if self.scale == 2:
            feat = self.lrelu(self.pixel_shuffle1(self.upconv1(feat)))
        elif self.scale == 4:
            feat = self.lrelu(self.pixel_shuffle1(self.upconv1(feat)))
            feat = self.lrelu(self.pixel_shuffle2(self.upconv2(feat)))
        
        out = self.conv_last(feat)
        return out

class DenoiseNet(nn.Module):
    """去噪网络"""
    
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64):
        super(DenoiseNet, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat * 2, num_feat * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_feat * 4, num_feat * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(num_feat * 2, num_feat, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1),
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.middle(feat)
        out = self.decoder(feat)
        return out

class DemoireNet(nn.Module):
    """去摩尔纹网络"""
    
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=32):
        super(DemoireNet, self).__init__()
        
        # 多尺度卷积
        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_in_ch, num_feat, 5, 1, 2)
        self.conv3 = nn.Conv2d(num_in_ch, num_feat, 7, 1, 3)
        
        # 特征融合
        self.fusion = nn.Conv2d(num_feat * 3, num_feat, 1, 1, 0)
        
        # 处理层
        self.process = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1),
        )
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        feat1 = self.lrelu(self.conv1(x))
        feat2 = self.lrelu(self.conv2(x))
        feat3 = self.lrelu(self.conv3(x))
        
        feat = torch.cat([feat1, feat2, feat3], dim=1)
        feat = self.fusion(feat)
        
        out = self.process(feat)
        return out + x  # 残差连接

class DeepEnhancer:
    """深度学习增强器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """加载预训练模型"""
        # 创建模型实例
        self.models['super_resolution'] = EnhancementNet(scale=2).to(self.device)
        self.models['denoise'] = DenoiseNet().to(self.device)
        self.models['demoire'] = DemoireNet().to(self.device)
        
        # 设置为评估模式
        for model in self.models.values():
            model.eval()
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 归一化到[0,1]
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)
    
    def postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """后处理图像"""
        # 转换回numpy
        image = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # 限制到[0,1]范围
        image = np.clip(image, 0, 1)
        
        # 转换到[0,255]
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def enhance_super_resolution(self, image: np.ndarray) -> np.ndarray:
        """超分辨率增强"""
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output_tensor = self.models['super_resolution'](input_tensor)
            return self.postprocess_image(output_tensor)
    
    def enhance_denoise(self, image: np.ndarray) -> np.ndarray:
        """去噪增强"""
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output_tensor = self.models['denoise'](input_tensor)
            return self.postprocess_image(output_tensor)
    
    def enhance_demoire(self, image: np.ndarray) -> np.ndarray:
        """去摩尔纹"""
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output_tensor = self.models['demoire'](input_tensor)
            return self.postprocess_image(output_tensor)
    
    def enhance_combined(self, image: np.ndarray, page_type: str) -> np.ndarray:
        """组合增强"""
        enhanced = image.copy()
        
        # 根据页面类型选择增强策略
        if page_type == "彩色":
            # 彩色页面：去摩尔纹 -> 去噪 -> 轻微超分
            enhanced = self.enhance_demoire(enhanced)
            enhanced = self.enhance_denoise(enhanced)
            
        elif page_type == "灰度":
            # 灰度页面：去噪 -> 超分辨率
            enhanced = self.enhance_denoise(enhanced)
            if enhanced.shape[0] < 2000 and enhanced.shape[1] < 2000:  # 避免内存问题
                enhanced = self.enhance_super_resolution(enhanced)
                
        elif page_type == "黑白":
            # 黑白页面：主要是锐化和去噪
            enhanced = self.enhance_denoise(enhanced)
        
        return enhanced

# 传统图像处理增强（作为备选方案）
class TraditionalEnhancer:
    """传统图像处理增强器"""
    
    @staticmethod
    def enhance_sharpness(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """锐化增强"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
        kernel[1,1] = 8 + strength
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.2) -> np.ndarray:
        """对比度增强"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    @staticmethod
    def reduce_noise(image: np.ndarray) -> np.ndarray:
        """降噪"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    @staticmethod
    def enhance_clahe(image: np.ndarray) -> np.ndarray:
        """CLAHE增强"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image)