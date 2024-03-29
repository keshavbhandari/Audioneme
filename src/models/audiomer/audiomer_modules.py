from performer_pytorch import Performer
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import math

from src.models.audiomer.audiomer_utils import to_channels_last, to_frames_last, make_divisible


class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        out_channels: int = -1,
        **kwargs: dict,
    ):
        super(SqueezeExcitation, self).__init__()
        assert in_channels > 0

        num_reduced_channels = make_divisible(
            max(out_channels, 8) // reduction, 8
        )

        self.fc1 = nn.Conv1d(in_channels, num_reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(num_reduced_channels, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = F.adaptive_avg_pool1d(inp, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x).sigmoid()
        return x


class SepConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False
    ):
        assert(stride < kernel_size)
        super(SepConv1d, self).__init__()
        padding = kernel_size - stride#+ 1
        self.depthwise = torch.nn.Conv1d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=in_channels,
                                         bias=bias)
        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.pointwise = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=bias, padding=1)

    def forward(self, x):
        # x.shape -> (b, channels, frames)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=False,
        expansion_factor=2,
        use_se=True
    ):
        super().__init__()
        assert(stride > 1)
        self.use_se = use_se
        self.sep_conv = SepConv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, bias=bias)
        if self.use_se:
            self.se = SqueezeExcitation(
                in_channels=out_channels)

    def forward(self, inp):
        x = self.sep_conv(inp)
        if self.use_se:
            x = x * self.se(x)
        return x

class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, expansion_factor, use_se, equal_strides):
        super().__init__()
        context_stride = stride // 2 if equal_strides else stride
        self.to_q = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride // 2,
            bias=False,
            expansion_factor=expansion_factor,
            use_se=use_se
        )
        self.to_context = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=context_stride,
            bias=False,
            expansion_factor=expansion_factor,
            use_se=use_se
        )

    def forward(self, x):
        q = self.to_q(x)
        q = to_channels_last(q)

        context = self.to_context(x)
        context = to_channels_last(context)
        return q, context


class AudiomerEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dim_head=64,
        depth=1,
        num_heads=6,
        expansion_factor=2,
        use_attention=True,
        use_se=True,
        equal_strides=False
        ):

        super().__init__()
        stride = kernel_size - 1
        self.use_attention = use_attention

        self.conv = ConvEmbedding(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            expansion_factor=expansion_factor,
            use_se=use_se,
            equal_strides=equal_strides
        )
        if self.use_attention:
            self.performer = Performer(
                dim=out_channels,
                depth=depth,
                heads=num_heads,
                dim_head=dim_head,
                ff_glu=True,
                attn_dropout=0.2,
                use_scalenorm=True,
                ff_mult=expansion_factor
            )

    def forward(self, x):
        # (b, in_channels, input_frames) -> (b, num_frames, out_channels)
        q, context = self.conv(x)
        # (b, num_frames, out_channels) -> (b, num_frames, out_channels)
        if self.use_attention:
            out = q + self.performer(q, context=context,
                                    context_mask=torch.ones_like(context).bool())
        else:
            out = q
        # (b, num_frames, out_channels) -> (b, out_channels, num_frames)
        out = to_frames_last(out)
        return out


class AudiomerEncoder(nn.Module):
    def __init__(self, config, kernel_sizes, dim_head, depth, num_heads, use_residual, use_cls, expansion_factor, input_size, use_se, use_attention, equal_strides, **kwargs):
        super().__init__()
        assert(len(kernel_sizes) == len(config) - 1)
        if use_cls:
            input_size += 128
        self.layers = []
        self.identity_layers = []
        self.use_residual = use_residual
        self.use_cls = use_cls
        paddings = [3, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
        for i in range(len(config) - 1):
            in_channels, out_channels = config[i], config[i+1]
            padding = paddings[i]
            self.layers.append(
                AudiomerEncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    num_heads=num_heads,
                    depth=depth,
                    dim_head=dim_head if isinstance(
                        dim_head, int) else dim_head[i],
                    expansion_factor=expansion_factor,
                    use_se=use_se,
                    use_attention=use_attention,
                    equal_strides=equal_strides
                )
            )
            if use_residual:
                self.identity_layers.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_sizes[i] // 2,
                            stride=kernel_sizes[i] // 2,
                            bias=False,
                            padding=padding
                        ),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
            input_size = math.floor(
                (input_size + kernel_sizes[i] // 2 + 1) / (kernel_sizes[i] // 2)) + 1

        self.layers = nn.ModuleList(self.layers)
        if use_residual:
            self.identity_layers = nn.ModuleList(self.identity_layers)
        else:
            self.identity_layers = [None] * len(self.layers)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(
                1, 1, 128))

    def forward(self, x):
        if self.use_cls:
            cls_tokens = repeat(
                self.cls_token, '() n d -> b n d', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=2)

        for (layer, id_layer) in zip(self.layers, self.identity_layers):
            x_copy = x
            x = layer(x)
            if self.use_residual:
                x_copy = id_layer(x_copy)
                x = x + x_copy
        return x