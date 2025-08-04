import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from torch.autograd import Function


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 确保输入张量x在内存中是连续存储的
        x = x.contiguous()
        # 保存后续反向传播所需的小波滤波器
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        # 记录输入x的原始形状以便后续恢复
        ctx.shape = x.shape

        # 获取输入的通道数
        dim = x.shape[1]
        # 分别利用各滤波器对x执行二维卷积操作，提取不同频带信息
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 将低频和高频子带在通道维度上合并
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            # 提取前向保存的小波滤波器
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            # 将梯度张量dx调整为包含4个子带的形式
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            # 拼接滤波器，并重复C次以匹配每个通道
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            # 利用转置卷积恢复输入的梯度
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        # 利用pywt库构建指定的小波对象
        w = pywt.Wavelet(wave)
        # 获取反转后的高通和低通滤波器
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        # 利用外积计算二维滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        # 为滤波器增加必要的维度，并注册为缓冲变量
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))
        # 将滤波器数据类型设置为float32
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        # 应用自定义DWT函数，返回分解后的四个子带拼接结果
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        # 保存滤波器以便反向传播时使用
        ctx.save_for_backward(filters)
        # 记录输入x的原始形状
        ctx.shape = x.shape

        B, _, H, W = x.shape
        # 将x重塑以区分四个子带
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        # 重复滤波器以匹配每个通道
        filters = filters.repeat(C, 1, 1, 1)
        # 通过转置卷积进行上采样重构
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            # 提取保存的滤波器
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            # 分离滤波器的各个部分
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            # 利用卷积操作提取各个子带的梯度
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            # 合并四个子带梯度
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        # 构建用于图像重构的小波对象
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        # 计算二维重构滤波器
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        # 为每个滤波器添加维度以匹配卷积核格式
        w_ll = w_ll.unsqueeze(0).unsqueeze(0)
        w_lh = w_lh.unsqueeze(0).unsqueeze(0)
        w_hl = w_hl.unsqueeze(0).unsqueeze(0)
        w_hh = w_hh.unsqueeze(0).unsqueeze(0)
        # 合并所有滤波器
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        # 注册为缓冲区变量，保证在训练过程中不会更新
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        # 利用IDWT_Function执行逆小波变换重构图像
        return IDWT_Function.apply(x, self.filters)


class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()
        # 定义两个3x3卷积层，保持通道数不变
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(out1))
        # 使用残差连接，将原始输入与处理结果相加
        out2 = out2 + x
        return out2


class Fusion(nn.Module):
    """
    本模块利用离散小波变换（DWT）将输入图像分解成低频与高频部分，
    分别对两部分进行独立处理，再利用逆离散小波变换（IDWT）重构出复原后的图像。
    """

    def __init__(self, in_channels, wave):
        super(Fusion, self).__init__()
        # 初始化2D小波分解模块
        self.dwt = DWT_2D(wave)
        # 通过1x1卷积将高频部分（三个分支）压缩到原始通道数
        self.convh1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # 利用残差网络模块提取高频细节
        self.high = ResNet(in_channels)
        # 再通过1x1卷积将高频部分恢复为原始通道数的三倍
        self.convh2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=True)
        # 对低频分量与辅助输入拼接后进行通道压缩
        self.convl = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # 采用残差网络模块强化低频信息
        self.low = ResNet(in_channels)
        # 初始化逆小波变换模块以恢复原始图像尺寸
        self.idwt = IDWT_2D(wave)

    def forward(self, x1, x2):
        # 获取x1的形状信息：批量、通道、高度、宽度
        b, c, h, w = x1.shape
        # 对x1进行小波分解，输出形状为 [B, in_channels*4, H/2, W/2]
        x_dwt = self.dwt(x1)
        # 将分解结果按通道拆分为低频与三个高频子带
        ll, lh, hl, hh = x_dwt.split(c, 1)
        # 拼接三个高频子带形成高频特征
        high = torch.cat([lh, hl, hh], 1)  # [B, in_channels*3, H/2, W/2]
        # 处理高频信息：降维、残差增强、再恢复通道数
        high1 = self.convh1(high)  # [B, in_channels, H/2, W/2]
        high2 = self.high(high1)  # [B, in_channels, H/2, W/2]
        highf = self.convh2(high2)  # [B, in_channels*3, H/2, W/2]

        # 获取低频部分ll和辅助输入x2的尺寸，若不匹配则进行填充对齐
        b1, c1, h1, w1 = ll.shape
        b2, c2, h2, w2 = x2.shape
        if h1 != h2:
            x2 = F.pad(x2, (0, 0, 1, 0), "constant", 0)
        # 拼接低频信息与辅助输入，并通过1x1卷积降维
        low = torch.cat([ll, x2], 1)  # [B, in_channels*2, H/2, W/2]
        low = self.convl(low)  # [B, in_channels, H/2, W/2]
        # 利用残差网络处理低频部分
        lowf = self.low(low)  # [B, in_channels, H/2, W/2]

        # 合并低频与高频特征后执行逆小波变换重构图像
        out = torch.cat((lowf, highf), 1)  # [B, in_channels*4, H/2, W/2]
        out_idwt = self.idwt(out)  # 重构至 [B, in_channels, H, W]
        return out_idwt