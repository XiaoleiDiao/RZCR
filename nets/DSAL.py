# -------------------------------------#
#   The attention layer: Dual spatial attention layer (DSAL)
# -------------------------------------#

import torch
import torch.nn as nn

# First attention step
class SpatialAttention_S1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_S1, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化高宽为1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid active function

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # take the average of all feature points in the channel dimension  -->  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # take the maximum of all feature points in the channel dimension -->  b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)  # concatenate in the first dimension --> b,2,h,w
        x = self.max_pool(x)
        x = self.conv1(x)  # Converted to dimensions, it becomes b,1,h,w
        return self.sigmoid(x)  # sigmoid active function

# Second attention step
class SpatialAttention_S2(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_S2, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Average pooling
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.avg_pool(x)
        x = self.conv1(x)
        return self.sigmoid(x)  # sigmoid active function


# Dual spatial attention layer (DSAL)
class DSAL_layer(nn.Module):
    def __init__(self, kernel_size=7):
        super(DSAL_layer, self).__init__()
        self.attention_1 = SpatialAttention_S1(kernel_size=kernel_size)
        self.attention_2 = SpatialAttention_S2(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.attention_1(x)
        x = x * self.attention_1(x)
        return x

if __name__ == '__main__':
    input=torch.randn(8,3,416,416)
    model = DSAL_layer(kernel_size = 7)
    out=model(input).cuda
    print(out)
