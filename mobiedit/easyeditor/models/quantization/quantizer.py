import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict
from transformers import PreTrainedModel
from torch.profiler import profile, ProfilerActivity
import time
# def profile_section(name, enable_cuda=True):
#     """用于上下文管理的 profile（按需包裹你要分析的部分）"""
#     activities = [ProfilerActivity.CPU]
#     if enable_cuda and torch.cuda.is_available():
#         activities.append(ProfilerActivity.CUDA)
#     return profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True, label=name)
def format_time(seconds):
    """将秒数转换为ms或μs为单位的字符串"""
    if seconds < 1e-6:
        return f"{seconds*1e9:.2f}ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.2f}μs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f}ms"
    else:
        return f"{seconds:.2f}s"

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # return grad_output  # straight-through
        # print(f"grad_output.norm:{grad_output.norm()}")
        # x, = ctx.saved_tensors
        # 用sigmoid拉平
        # grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))
        # print(f"grad.norm:{grad.norm()}")
        return grad_output

round_ste = RoundSTE.apply

class StaticQuantizer:
    """静态量化器，用于将权重和激活值静态量化"""
    
    def __init__(self, bits: int = 8, symmetric: bool = True, per_channel: bool = False):
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.zero_point = None
        self.dtype = torch.qint8 if bits == 8 else torch.qint16
        self.qrange = 2**(bits-1) - 1 if symmetric else 2**bits - 1
        
    def calibrate(self, x: torch.Tensor):
        """计算量化参数
        Args:
            x: 输入tensor
        """
        with torch.no_grad():
            if self.per_channel and len(x.shape) > 1:
                # 通道级量化
                if self.min_val is None:
                    self.min_val = x.min(dim=1)[0]  # 沿通道维度取最小值
                if self.max_val is None:
                    self.max_val = x.max(dim=1)[0]  # 沿通道维度取最大值
                
                # 计算每个通道的量化参数
                if self.symmetric:
                    abs_max = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
                    self.scale = abs_max / self.qrange
                    self.zero_point = torch.zeros_like(self.scale)
                else:
                    self.scale = (self.max_val - self.min_val) / self.qrange
                    self.zero_point = -torch.round(self.min_val / self.scale)
            else:
                # 全局量化
                if self.min_val is None:
                    self.min_val = x.min()
                if self.max_val is None:
                    self.max_val = x.max()
                    
                if self.symmetric:
                    abs_max = max(abs(self.min_val), abs(self.max_val))
                    self.scale = abs_max / self.qrange
                    self.zero_point = 0
                else:
                    self.scale = (self.max_val - self.min_val) / self.qrange
                    self.zero_point = -torch.round(self.min_val / self.scale)

    def quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """静态量化一个tensor
        Args:
            x: 输入tensor
        Returns:
            量化后的tensor
        """
        # with torch.no_grad():
        if self.scale is None:
            self.calibrate(x)
        
        if self.per_channel and len(x.shape) > 1:
            # 通道级量化
            scale = self.scale.view(-1, 1)  # 调整形状以匹配输入
            zero_point = self.zero_point.view(-1, 1)
            q_x = torch.round(x / scale + zero_point)
            
            if self.symmetric:
                q_min, q_max = -self.qrange, self.qrange
            else:
                q_min, q_max = 0, self.qrange
                
            q_x = torch.clamp(q_x, q_min, q_max)
        else:
            # 全局量化
            q_x = torch.round(x / self.scale + self.zero_point)
            
            if self.symmetric:
                q_min, q_max = -self.qrange, self.qrange
            else:
                q_min, q_max = 0, self.qrange
                
            q_x = torch.clamp(q_x, q_min, q_max)
        
        return q_x.to(self.dtype)

    def dequantize_tensor(self, q_x: torch.Tensor) -> torch.Tensor:
        """将量化tensor反量化为浮点数
        Args:
            q_x: 量化的tensor
        Returns:
            反量化后的浮点tensor
        """
        # with torch.no_grad():
        if self.per_channel and len(q_x.shape) > 1:
            scale = self.scale.view(-1, 1)
            zero_point = self.zero_point.view(-1, 1)
            return (q_x.to(torch.float32) - zero_point) * scale
        else:
            return (q_x.to(torch.float32) - self.zero_point) * self.scale

class StaticQuantizedLinear(nn.Module):
    """静态量化的线性层"""
    
    def __init__(self, in_features, out_features, original_module=None, weight_quantizer=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化量化器
        self.weight_quantizer = weight_quantizer or StaticQuantizer(bits=8, symmetric=True)
        
        # 注册缓冲区以存储量化和反量化参数
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        
        # 如果有原始模块，从中初始化
        if original_module is not None:
            # 量化权重
            self.register_buffer('weight_int', self.weight_quantizer.quantize_tensor(original_module.weight))
            self.register_buffer('weight_scale', self.weight_quantizer.scale)
            self.register_buffer('weight_zero_point', self.weight_quantizer.zero_point)
            
            # 处理偏置
            if hasattr(original_module, 'bias') and original_module.bias is not None:
                self.register_buffer('bias', original_module.bias.clone())
            else:
                self.register_buffer('bias', None)
        else:
            # 从头初始化
            self.register_buffer('weight_int', torch.zeros((out_features, in_features), dtype=torch.qint8))
            self.register_buffer('weight_scale', torch.tensor(1.0))
            self.register_buffer('weight_zero_point', torch.tensor(0.0))
            
            if bias:
                self.register_buffer('bias', torch.zeros(out_features))
            else:
                self.register_buffer('bias', None)
    
    def forward(self, x):
        """前向传播，使用量化权重执行线性变换
        
        Args:
            x: 输入tensor
            
        Returns:
            输出tensor
        """
        # 确保输入为float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        # 反量化权重
        weight_float = (self.weight_int.to(torch.float32) - self.weight_zero_point) * self.weight_scale
        
        # 线性变换
        return nn.functional.linear(x, weight_float, self.bias)
    
    @classmethod
    def from_float(cls, module, weight_quantizer=None):
        """从浮点Linear模块创建量化模块
        
        Args:
            module: 原始浮点Linear模块
            weight_quantizer: 权重量化器
            
        Returns:
            量化的LinearModule
        """
        assert isinstance(module, nn.Linear), "模块必须是nn.Linear类型"
        
        quantized_module = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            original_module=module,
            weight_quantizer=weight_quantizer,
            bias=module.bias is not None
        )
        
        return quantized_module

class ActivationStaticQuantizer:
    """激活值静态量化器，用于将激活值静态量化为int16类型"""
    
    def __init__(self, bits: int = 16, symmetric: bool = False, per_channel: bool = False):
        self.bits = bits
        self.symmetric = None
        self.per_channel = per_channel
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.zero_point = None
        self.dtype = torch.qint8 if bits == 8 else torch.int16
        self.qrange = 2**(bits-1) - 1 if symmetric else 2**bits - 1
        self.percentile_99 = None  # 存储第99百分位数
        self.use_special_strategy = False  # 是否使用特殊量化策略
        self.is_calibrated = False  # 标记是否已经完成校准
        self.stats = {}  # 存储统计信息
        
    def calibrate(self, x: torch.Tensor, module_name: str = None):
        """计算量化参数并收集统计信息
        Args:
            x: 输入tensor
            module_name: 模块名称，用于存储统计信息
        """
        with torch.no_grad():
            # 计算基本统计量
            min_val = x.min()
            max_val = x.max()
            
            # 计算第99百分位数
            flat_tensor = x.reshape(-1)
            # sorted_tensor, _ = torch.sort(flat_tensor)
            # percentile_99_idx = int(0.99 * sorted_tensor.numel())
            # percentile_99 = sorted_tensor[percentile_99_idx].item()
            percentile_99 = torch.quantile(flat_tensor, 0.99).item()

            
            # 存储统计信息
            if module_name:
                self.stats[module_name] = {
                    'min_val': min_val.item(),
                    'max_val': max_val.item(),
                    'percentile_99': percentile_99,
                    'shape': x.shape,
                    'dtype': x.dtype
                }
            
            # 如果是第一次校准或强制刷新，更新量化参数
            # if not self.is_calibrated:
            self.min_val = min_val
            self.max_val = max_val
            self.percentile_99 = percentile_99
            
            # print(f" before max_val:{self.max_val}, percentile_99:{self.percentile_99}, min_val:{self.min_val}")
            
            # print(f"10000*percentile_99:{10000*self.percentile_99}")
            # 检查是否需要使用特殊量化策略
            if self.max_val > 9999 * self.percentile_99:
                # print(f"max_val:{self.max_val}")
                # print(f"4*percentile_99:{4*self.percentile_99}")
                # print(f"检测到异常值: max={self.max_val.item():.6f}, top1%={self.percentile_99:.6f}, 比例={self.max_val.item()/self.percentile_99:.2f}")
                # print("启用特殊量化策略: 将top1%设为最大值，不量化超过最大值的部分")
                self.use_special_strategy = True
                self.max_val = self.percentile_99
            
            # 计算量化和反量化的参数
            if self.symmetric:
                # 对称量化，找到绝对值最大值
                # print("symmetric")
                abs_max = max(abs(self.min_val), abs(self.max_val))
                self.scale = abs_max / self.qrange
                self.zero_point = 0
            else:
                # 非对称量化
                # print("non-symmetric")
                self.scale = (self.max_val - self.min_val) / self.qrange
                self.scale = self.scale * 1.5
                # self.zero_point = -torch.round(self.min_val / self.scale)
                self.zero_point = int(round(-self.min_val.item() / self.scale.item()))
            # print(f"after min_val:{self.min_val:.6f}, max_val:{self.max_val:.6f}, scale:{self.scale:.6f}, zero_point:{self.zero_point:.6f}")
            
            self.is_calibrated = True
            # print(f"量化器校准完成: min={self.min_val.item():.6f}, max={self.max_val.item():.6f}, scale={self.scale.item():.6f}, zero_point={self.zero_point.item():.6f}")

    def quantize_tensor(self, x: torch.Tensor, force_refresh=False) -> torch.Tensor:
        """使用校准阶段设置的固定参数进行量化
        Args:
            x: 输入tensor
            force_refresh: 是否强制刷新量化参数（通常不使用）
        Returns:
            量化后的tensor
        """
        
        # with torch.no_grad():
        # 如果强制刷新，重新校准
        # with profile_section("quantize_tensor"):
        if not self.is_calibrated:
            self.is_calibrated = True
            self.calibrate(x)
        # 如果使用特殊量化策略，先处理超过最大值的部分
        if self.use_special_strategy:
            # print(f"use_special_strategy")
            # 创建一个掩码，标记超过最大值的部分
            mask = x > self.max_val
            # 创建一个结果tensor，初始值为量化后的值
            result = torch.zeros_like(x)
            # 对不超过最大值的部分进行常规量化
            q_x_normal = round_ste(x / self.scale + self.zero_point)
            if self.symmetric:
                q_min, q_max = -self.qrange, self.qrange
            else:
                q_min, q_max = 0, self.qrange
            q_x_normal = torch.clamp(q_x_normal, q_min, q_max)
            result[~mask] = q_x_normal[~mask]
            # 对超过最大值的部分进行量化但不裁剪
            q_x_overflow = round_ste(x / self.scale + self.zero_point)
            # print(f"q_x_overflow:{q_x_overflow}")
            result[mask] = q_x_overflow[mask]
            return result
        else:
            # 常规量化流程
            q_x = round_ste(x / self.scale + self.zero_point)
            if self.symmetric:
                q_min, q_max = -self.qrange, self.qrange
            else:
                q_min, q_max = 0, self.qrange
            q_x = torch.clamp(q_x, q_min, q_max)
            return q_x

    def dequantize_tensor(self, q_x: torch.Tensor) -> torch.Tensor:
        """使用校准阶段设置的固定参数进行反量化
        Args:
            q_x: 量化的tensor
        Returns:
            反量化后的浮点tensor
        """
        # with profile_section("dequantize_tensor"):
        if not self.is_calibrated:
            raise RuntimeError("量化器尚未校准，请先调用calibrate方法")
        # with torch.no_grad():
        # 获取输入tensor的原始数据类型
        original_dtype = q_x.dtype
        # 如果使用特殊量化策略，需要区分处理
        if self.use_special_strategy:
            # 创建一个掩码，标记超过量化范围的部分
            if self.symmetric:
                q_min, q_max = -self.qrange, self.qrange
            else:
                q_min, q_max = 0, self.qrange
            mask = (q_x > q_max) | (q_x < q_min)
            # 创建一个结果tensor
            result = torch.zeros_like(q_x, dtype=torch.float32)
            # 对在量化范围内的部分进行常规反量化
            result[~mask] = (q_x[~mask].to(torch.float32) - self.zero_point) * self.scale
            # 对超出量化范围的部分进行反量化，并检查是否会导致溢出
            overflow_values = (q_x[mask].to(torch.float32) - self.zero_point) * self.scale
            # 检查反量化后的值是否会导致溢出
            if torch.any(torch.isinf(overflow_values)) or torch.any(torch.isnan(overflow_values)):
                print("警告：检测到反量化后出现溢出，将使用最大值进行截断")
                # 使用最大值进行截断
                overflow_values = torch.clamp(overflow_values, -1e38, 1e38)
            result[mask] = overflow_values
            # 转换回原始数据类型
            return result.to(original_dtype)
        else:
            # 常规反量化流程
            result = (q_x.to(torch.float32) - self.zero_point) * self.scale
            return result.to(original_dtype)
            
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """量化然后立即反量化一个tensor（用于模拟量化效果）
        Args:
            x: 输入tensor
        Returns:
            经过量化和反量化的tensor，保持原始数据类型
        """
        # start_time = time.time()
        # with torch.no_grad():
        # with profile_section("quantize_dequantize"):
        original_dtype = x.dtype
        # print(f"original_dtype:{original_dtype}")
        q_x = self.quantize_tensor(x)
        # print(f"[quantize_dequantize] q_x id: {id(q_x)}")
        q_x_dequant = self.dequantize_tensor(q_x)
        # print(f"[quantize_dequantize] q_x_dequant id: {id(q_x_dequant)}")
        # total_time = time.time() - start_time
        # print(f"总耗时: {format_time(total_time)}")
        return q_x_dequant.to(original_dtype)
            
    def save_calibration(self, path: str):
        """保存校准参数到文件
        Args:
            path: 保存路径
        """
        if not self.is_calibrated:
            raise RuntimeError("量化器尚未校准，无法保存参数")
            
        calibration_data = {
            'min_val': self.min_val.item(),
            'max_val': self.max_val.item(),
            'scale': self.scale.item(),
            'zero_point': self.zero_point.item(),
            'percentile_99': self.percentile_99,
            'use_special_strategy': self.use_special_strategy,
            'symmetric': self.symmetric,
            'bits': self.bits
        }
        
        import json
        with open(path, 'w') as f:
            json.dump(calibration_data, f)
        print(f"校准参数已保存到 {path}")
        
    def load_calibration(self, path: str):
        """从文件加载校准参数
        Args:
            path: 加载路径
        """
        import json
        with open(path, 'r') as f:
            calibration_data = json.load(f)
            
        self.min_val = torch.tensor(calibration_data['min_val'])
        self.max_val = torch.tensor(calibration_data['max_val'])
        self.scale = torch.tensor(calibration_data['scale'])
        self.zero_point = torch.tensor(calibration_data['zero_point'])
        self.percentile_99 = calibration_data['percentile_99']
        self.use_special_strategy = calibration_data['use_special_strategy']
        self.symmetric = calibration_data['symmetric']
        self.bits = calibration_data['bits']
        
        # 更新相关参数
        self.dtype = torch.qint8 if self.bits == 8 else torch.int16
        self.qrange = 2**(self.bits-1) - 1 if self.symmetric else 2**self.bits - 1
        
        self.is_calibrated = True
        print(f"已从 {path} 加载校准参数")

    def get_stats(self):
        """获取统计信息"""
        return self.stats

class W8A16Model(nn.Module):
    """W8A16量化模型包装器
    
    使用BitsAndBytes的8-bit权重量化和16-bit激活值的静态量化策略
    """
    
    def __init__(self, model: PreTrainedModel, device: str, skip_modules: Optional[List[str]] = None, 
                 weight_per_channel: bool = True,  # 默认启用权重的per-channel量化
                 activation_per_channel: bool = False,  # 默认不启用激活值的per-channel量化
                 mixed_precision: bool = False):
        super().__init__()
        self.model = model
        self.device = device
        self.skip_modules = skip_modules or []
        self.weight_per_channel = weight_per_channel
        self.activation_per_channel = activation_per_channel
        self.mixed_precision = mixed_precision
        self.debug_mode = False  # 添加调试模式标志
        self.gradient_stats = {}  # 存储梯度统计信息
        
        # 初始化量化器
        if mixed_precision:
            self.weight_quantizer = MixedPrecisionQuantizer(default_bits=8)
            self.activation_quantizer = MixedPrecisionQuantizer(default_bits=16)
        else:
            # 权重量化器总是使用per-channel量化
            # self.weight_quantizer = StaticQuantizer(bits=8, symmetric=True, per_channel=True)
            # 激活值量化器根据参数决定是否使用per-channel量化
            self.activation_quantizer = ActivationStaticQuantizer(
                bits=16, 
                symmetric=False,
                per_channel=activation_per_channel
            )
        
        self._original_weights = {}
        self._store_original_weights()
        
        # 存储每个模块的量化参数
        self.module_quantization_params = {}
        
        print(f"初始化W8A16Model完成:")
        print(f"- 权重量化: {'per-channel' if weight_per_channel else 'per-tensor'}")
        print(f"- 激活值量化: {'per-channel' if activation_per_channel else 'per-tensor'}")
        print(f"- 混合精度: {'启用' if mixed_precision else '禁用'}")

    def enable_debug(self):
        """启用调试模式，将注册梯度钩子"""
        self.debug_mode = True
        self.gradient_stats.clear()
        
        def hook_fn(name):
            def hook(grad):
                if grad is not None:
                    if name not in self.gradient_stats:
                        self.gradient_stats[name] = []
                    stats = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'shape': list(grad.shape),
                        'has_nan': torch.isnan(grad).any().item(),
                        'has_inf': torch.isinf(grad).any().item()
                    }
                    self.gradient_stats[name].append(stats)
                    print(f"\n层级: {name}")
                    print(f"梯度统计: {stats}")
            return hook

        # 为所有参数注册梯度钩子
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(hook_fn(name))
        print("已启用梯度调试模式")

    def disable_debug(self):
        """禁用调试模式"""
        self.debug_mode = False
        self.gradient_stats.clear()
        print("已禁用梯度调试模式")

    def print_gradient_stats(self):
        """打印累积的梯度统计信息"""
        if not self.gradient_stats:
            print("没有可用的梯度统计信息")
            return

        print("\n=== 梯度统计信息汇总 ===")
        for name, stats_list in self.gradient_stats.items():
            if stats_list:
                last_stats = stats_list[-1]  # 获取最新的统计信息
                print(f"\n层级: {name}")
                print(f"形状: {last_stats['shape']}")
                print(f"均值: {last_stats['mean']:.6f}")
                print(f"标准差: {last_stats['std']:.6f}")
                print(f"最大值: {last_stats['max']:.6f}")
                print(f"最小值: {last_stats['min']:.6f}")
                print(f"存在NaN: {last_stats['has_nan']}")
                print(f"存在Inf: {last_stats['has_inf']}")
        print("\n=== 统计信息结束 ===")

    def get_gradient_stats(self):
        """获取梯度统计信息"""
        return self.gradient_stats

    def set_layer_bits(self, layer_name: str, weight_bits: int = None, activation_bits: int = None):
        """设置特定层的量化位数
        Args:
            layer_name: 层名称
            weight_bits: 权重量化位数
            activation_bits: 激活值量化位数
        """
        if self.mixed_precision:
            if weight_bits is not None:
                self.weight_quantizer.set_layer_bits(layer_name, weight_bits)
            if activation_bits is not None:
                self.activation_quantizer.set_layer_bits(layer_name, activation_bits)
                
    def _setup_hooks(self):
        """设置钩子来静态量化激活值"""
        print("设置激活值静态量化钩子...")
        
        def create_quantize_hook(module_name):
            def hook(module, inputs, output):
                try:
                    if isinstance(output, torch.Tensor):
                        # 使用模块特定的量化器
                        # if self.mixed_precision:
                            # return self.activation_quantizer.quantize_dequantize(output, module_name)
                        if module_name in self.module_quantization_params:
                            quant_output = self.module_quantization_params[module_name].quantize_dequantize(output)
                            # print(f"[hook] {module_name} quant_output id: {id(quant_output)}")
                            return quant_output
                        # else:
                        #     print(f"警告：模块 {module_name} 没有对应的量化器，使用全局量化器")
                        #     return self.activation_quantizer.quantize_dequantize(output)
                    elif isinstance(output, tuple):
                        # if self.mixed_precision:
                            # return tuple(self.activation_quantizer.quantize_dequantize(o, module_name) 
                            #           if isinstance(o, torch.Tensor) else o 
                            #           for o in output)
                        if module_name in self.module_quantization_params:
                            quant_output = tuple(self.module_quantization_params[module_name].quantize_dequantize(o) 
                                      if isinstance(o, torch.Tensor) else o 
                                      for o in output)
                            # print(f"[hook] {module_name} quant_output id: {id(quant_output)}")
                            return quant_output
                        else:
                            print(f"警告：模块 {module_name} 没有对应的量化器")
                            # return tuple(self.activation_quantizer.quantize_dequantize(o) 
                                    #   if isinstance(o, torch.Tensor) else o 
                            #           for o in output)
                    return output
                except Exception as e:
                    print(f"警告：模块 {module_name} 量化时发生错误: {str(e)}")
                    return output
            return hook

        hook_count = 0
        for name, module in self.model.named_modules():
            if self._should_quantize_module(name, module):
                hook = create_quantize_hook(name)
                module.register_forward_hook(hook)
                hook_count += 1
        
        print(f"已为 {hook_count} 个模块设置激活值量化钩子")

    def _store_original_weights(self):
        """存储原始权重的引用（用于模型编辑）"""
        print("存储原始权重引用...")
        for name, param in self.model.named_parameters():
            self._original_weights[name] = param
        print(f"已存储 {len(self._original_weights)} 个权重参数")

    def _patch_lm_head(self):
        """处理lm_head以确保类型兼容性"""
        if not hasattr(self.model, 'lm_head'):
            return
            
        print("修补lm_head以确保类型兼容性...")
        
        # 保存原始的lm_head前向传播方法
        import types
        import torch.nn.functional as F
        original_lm_head_forward = self.model.lm_head.forward
        
        # 定义新的前向传播方法，确保输入是float32类型
        def patched_forward(self, hidden_states):
            # 确保hidden_states是float32类型
            if isinstance(self.weight, torch.Tensor) and self.weight.dtype != torch.float32:
                weight = self.weight.to(torch.float32)
            return F.linear(hidden_states, weight, self.bias)
        
        # 替换方法
        self.model.lm_head.forward = types.MethodType(patched_forward, self.model.lm_head)
        print("已修补lm_head前向传播方法")
        
        # 存储原始方法以便恢复
        self._original_lm_head_forward = original_lm_head_forward
        
        # 添加钩子，在调用lm_head前确保输入类型正确
        def pre_lm_head_hook(module, input_tuple):
            if not input_tuple:
                return input_tuple
            # 确保输入张量是float32类型
            processed_inputs = []
            for x in input_tuple:
                if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
                    processed_inputs.append(x.to(torch.float32))
                else:
                    processed_inputs.append(x)
            return tuple(processed_inputs)
        
        # 注册钩子
        self._lm_head_hook = self.model.lm_head.register_forward_pre_hook(pre_lm_head_hook)
        print("已为lm_head添加类型转换钩子")

    def _should_quantize_module(self, name: str, module: nn.Module) -> bool:
        """判断是否应该量化该模块的激活值
        Args:
            name: 模块名称
            module: 模块实例
        Returns:
            是否应该量化
        """
        # print(f"skip_modules:{self.skip_modules}")
        # print(f"name:{name}")
        if any(skip_str in name for skip_str in self.skip_modules):
            # print(f"skip name:{name}")
            return False
        # 只对Linear层的输出应用激活值量化
        return isinstance(module, nn.Linear) and 'lm_head' not in name

    def get_parameter(self, param_name: str) -> torch.Tensor:
        """获取模型参数，优先从原始权重中获取（用于模型编辑）"""
        try:
            # 首先尝试从原始权重中获取
            if param_name in self._original_weights:
                return self._original_weights[param_name]
            
            # 如果不在原始权重中，尝试从模型中获取
            return self.model.get_parameter(param_name)
            
        except Exception as e:
            raise LookupError(f"无法找到参数 {param_name}: {str(e)}")

    def state_dict(self, *args, **kwargs):
        """获取模型状态字典"""
        return self.model.state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        """获取模型命名参数"""
        for name, param in self.model.named_parameters(*args, **kwargs):
            yield name, param
            
    def __getattr__(self, name):
        """代理访问底层模型的属性"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 如果当前类没有该属性，尝试从底层模型获取
            if hasattr(self.model, name):
                return getattr(self.model, name)
            # 如果底层模型也没有该属性，尝试从模型的config中获取
            if hasattr(self.model, 'config') and hasattr(self.model.config, name):
                return getattr(self.model.config, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def named_modules(self, *args, **kwargs):
        for name, module in self.model.named_modules(*args, **kwargs):
            yield name, module

    def forward(self, *args, **kwargs):
        """前向传播，设备转换但保持数据类型"""
        # 处理输入设备不匹配
        model_device = next(self.model.parameters()).device
        
        # 检查并转移device，但不改变数据类型
        def move_to_device(x):
            if isinstance(x, torch.Tensor) and x.device != model_device:
                return x.to(device=model_device)
            return x
        
        # 处理位置参数和关键字参数的设备
        processed_args = [move_to_device(arg) for arg in args]
        
        processed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                processed_kwargs[key] = move_to_device(value)
            elif isinstance(value, dict):
                processed_dict = {k: move_to_device(v) if isinstance(v, torch.Tensor) else v 
                                  for k, v in value.items()}
                processed_kwargs[key] = processed_dict
            else:
                processed_kwargs[key] = value
        
        # 直接调用模型，钩子会处理激活值量化
        try:
            return self.model(*processed_args, **processed_kwargs)
        except RuntimeError as e:
            error_msg = str(e)
            print(f"前向传播出错: {error_msg}")
            
            # 处理特定的类型不匹配错误
            if "expected scalar type Float but found Half" in error_msg:
                print("检测到Float/Half类型不匹配，应用紧急修复...")
                
                # 创建临时钩子来强制转换所有激活值
                hooks = []
                
                def cast_outputs_hook(module, inputs, outputs):
                    if isinstance(outputs, torch.Tensor) and outputs.dtype != torch.float32:
                        print(f"outputs.dtype:{outputs.dtype}")
                        return outputs.to(torch.float32)
                    elif isinstance(outputs, tuple):
                        print(f"tuple.output.dtype:{tuple[0].dtype}")
                        return tuple(o.to(torch.float32) if isinstance(o, torch.Tensor) and o.dtype != torch.float32 else o for o in outputs)
                    return outputs
                
                # 添加钩子到所有线性层
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        hooks.append(module.register_forward_hook(cast_outputs_hook))
                
                try:
                    # 重试
                    result = self.model(*processed_args, **processed_kwargs)
                    
                    # 移除钩子
                    for hook in hooks:
                        hook.remove()
                    
                    return result
                except Exception as e2:
                    # 移除钩子
                    for hook in hooks:
                        hook.remove()
                    print(f"紧急修复失败: {e2}")
            
            raise
        
    def __del__(self):
        """析构函数，确保恢复原始方法"""
        if hasattr(self, '_original_lm_head_forward') and hasattr(self.model, 'lm_head'):
            try:
                import types
                self.model.lm_head.forward = types.MethodType(self._original_lm_head_forward, self.model.lm_head)
            except:
                pass

    def calibrate(self, sample_inputs: Dict[str, torch.Tensor], edit_output=None, insert_module_name=None):
        print("校准激活值量化参数...")
        self.eval()
        activation_stats = {}

        # 注册 hook，统一调用 activation_quantizer.calibrate
        def calibration_hook(module_name):
            def hook(module, input, output):
                # 如果提供了 edit_output 函数，先应用它
                if insert_module_name is not None:
                    # print(f"insert_module_name:{insert_module_name}")
                    # print(f"module_name:{module_name}")
                    if insert_module_name == module_name:
                        if edit_output is not None:
                            print(f"edit_output:{edit_output}")
                            # print(f"output:{output}")
                            print(f"insert module_name:{module_name}")
                            output = edit_output(output, module_name)
                
                if isinstance(output, torch.Tensor):
                    # 为每个模块创建独立的量化器
                    if module_name not in self.module_quantization_params and module_name != insert_module_name:
                        print(f"module_name:{module_name}")
                        self.module_quantization_params[module_name] = ActivationStaticQuantizer(bits=16, symmetric=True)
                    
                    # 使用模块特定的量化器进行校准
                    if module_name != insert_module_name:
                        self.module_quantization_params[module_name].calibrate(output, module_name=module_name)
                    
                    # 收集统计信息
                    if module_name not in activation_stats:
                        activation_stats[module_name] = []
                    if module_name != insert_module_name:
                        activation_stats[module_name].append(self.module_quantization_params[module_name].stats[module_name])
                    
                elif isinstance(output, tuple):
                    # 如果提供了 edit_output 函数，先应用它
                    if edit_output is not None:
                        output = edit_output(output, module_name)
                        
                    for o in output:
                        if isinstance(o, torch.Tensor):
                            # 为每个模块创建独立的量化器
                            if module_name not in self.module_quantization_params:
                                self.module_quantization_params[module_name] = ActivationStaticQuantizer(bits=16, symmetric=True)
                            
                            # 使用模块特定的量化器进行校准
                            self.module_quantization_params[module_name].calibrate(o, module_name=module_name)
                            
                            # 收集统计信息
                            if module_name not in activation_stats:
                                activation_stats[module_name] = []
                            activation_stats[module_name].append(self.module_quantization_params[module_name].stats[module_name])
                            break
                return output
            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if self._should_quantize_module(name, module):
                hooks.append(module.register_forward_hook(calibration_hook(name)))

        # 前向传播以收集所有量化信息
        with torch.no_grad():
            outputs = self.model(**sample_inputs)

        # 移除所有 hook
        for h in hooks:
            h.remove()

        # print("\n===== 每层激活值量化参数统计 =====")
        # print(f"{'模块名称':<50} {'max':<10} {'top1%':<10} {'比例':<10} {'使用特殊策略':<10}")
        # print("-" * 90)
        for name, stats_list in activation_stats.items():
            if stats_list:  # 确保有统计数据
                stats = stats_list[-1]  # 使用最后一次的统计数据
                max_val = stats['max_val']
                top1p = stats['percentile_99']
                ratio = max_val / top1p if top1p != 0 else float('inf')
                special = "是" if ratio > 4 else "否"
                short_name = name[-50:] if len(name) > 50 else name
                # print(f"{short_name:<50} {max_val:<10.4f} {top1p:<10.4f} {ratio:<10.2f} {special:<10}")

        # print("=" * 90)
        
        # 设置激活值量化的钩子
        self._setup_hooks()

    def save_pretrained(self, save_directory: str):
        """保存量化后的模型"""
        self.model.save_pretrained(save_directory)
        
        # 保存量化配置
        quantization_config = {
            "activation_bits": self.activation_quantizer.bits,
            "activation_symmetric": self.activation_quantizer.symmetric,
            "min_val": self.activation_quantizer.min_val.item() if self.activation_quantizer.min_val is not None else None,
            "max_val": self.activation_quantizer.max_val.item() if self.activation_quantizer.max_val is not None else None,
            "scale": self.activation_quantizer.scale.item() if self.activation_quantizer.scale is not None else None,
            "zero_point": self.activation_quantizer.zero_point.item() if self.activation_quantizer.zero_point is not None else None,
            "skip_modules": self.skip_modules,
            "module_quantization_params": {
                name: {
                    "min_val": quantizer.min_val.item() if quantizer.min_val is not None else None,
                    "max_val": quantizer.max_val.item() if quantizer.max_val is not None else None,
                    "scale": quantizer.scale.item() if quantizer.scale is not None else None,
                    "zero_point": quantizer.zero_point.item() if quantizer.zero_point is not None else None,
                    "percentile_99": quantizer.percentile_99,
                    "use_special_strategy": quantizer.use_special_strategy
                }
                for name, quantizer in self.module_quantization_params.items()
            }
        }
        torch.save(quantization_config, f"{save_directory}/activation_quantization_config.pt")

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, save_directory: str, device: str):
        """从保存的文件加载量化模型"""
        try:
            quantization_config = torch.load(f"{save_directory}/activation_quantization_config.pt")
            skip_modules = quantization_config.get("skip_modules", [])
        except:
            print("未找到量化配置，使用默认设置")
            quantization_config = {}
            skip_modules = []
            
        instance = cls(
            model=model,
            device=device,
            skip_modules=skip_modules
        )
        
        # 恢复量化参数
        if "min_val" in quantization_config and quantization_config["min_val"] is not None:
            instance.activation_quantizer.min_val = torch.tensor(quantization_config["min_val"])
            instance.activation_quantizer.max_val = torch.tensor(quantization_config["max_val"])
            instance.activation_quantizer.scale = torch.tensor(quantization_config["scale"])
            instance.activation_quantizer.zero_point = torch.tensor(quantization_config["zero_point"])
            print("已加载预校准的激活值量化参数")
        
        # 恢复每个模块的量化参数
        if "module_quantization_params" in quantization_config:
            for name, params in quantization_config["module_quantization_params"].items():
                quantizer = ActivationStaticQuantizer(bits=16, symmetric=False)
                if params["min_val"] is not None:
                    quantizer.min_val = torch.tensor(params["min_val"])
                    quantizer.max_val = torch.tensor(params["max_val"])
                    quantizer.scale = torch.tensor(params["scale"])
                    quantizer.zero_point = torch.tensor(params["zero_point"])
                    quantizer.percentile_99 = params["percentile_99"]
                    quantizer.use_special_strategy = params["use_special_strategy"]
                    quantizer.is_calibrated = True
                instance.module_quantization_params[name] = quantizer
            print(f"已加载 {len(instance.module_quantization_params)} 个模块的量化参数")
            
        return instance 

class MixedPrecisionQuantizer:
    """混合精度量化器，支持不同层使用不同的量化精度"""
    
    def __init__(self, default_bits: int = 8):
        self.default_bits = default_bits
        self.quantizers = {}  # 存储不同层的量化器
        self.layer_bits = {}  # 存储每层的量化位数
        
    def set_layer_bits(self, layer_name: str, bits: int):
        """设置特定层的量化位数
        Args:
            layer_name: 层名称
            bits: 量化位数
        """
        self.layer_bits[layer_name] = bits
        if layer_name not in self.quantizers:
            self.quantizers[layer_name] = StaticQuantizer(bits=bits)
            
    def get_quantizer(self, layer_name: str) -> StaticQuantizer:
        """获取特定层的量化器
        Args:
            layer_name: 层名称
        Returns:
            量化器实例
        """
        if layer_name not in self.quantizers:
            bits = self.layer_bits.get(layer_name, self.default_bits)
            self.quantizers[layer_name] = StaticQuantizer(bits=bits)
        return self.quantizers[layer_name]
        
    def quantize_tensor(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """量化特定层的tensor
        Args:
            x: 输入tensor
            layer_name: 层名称
        Returns:
            量化后的tensor
        """
        quantizer = self.get_quantizer(layer_name)
        return quantizer.quantize_tensor(x)
        
    def dequantize_tensor(self, q_x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """反量化特定层的tensor
        Args:
            q_x: 量化的tensor
            layer_name: 层名称
        Returns:
            反量化后的tensor
        """
        quantizer = self.get_quantizer(layer_name)
        return quantizer.dequantize_tensor(q_x)
        
    def calibrate(self, x: torch.Tensor, layer_name: str):
        """校准特定层的量化参数
        Args:
            x: 输入tensor
            layer_name: 层名称
        """
        quantizer = self.get_quantizer(layer_name)
        quantizer.calibrate(x) 