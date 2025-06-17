from torch import nn
import torch

class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm,self).__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2):
        x = torch.add(x1, x2)  # Residual connection
        return self.normalize(x)
    

# # 创建 AddAndNorm 层
# add_and_norm = AddAndNorm(hidden_layer_size=16)
#
# # 生成模拟输入
# x1 = torch.rand(5, 10, 16)  # (batch_size=5, time_steps=10, hidden_dim=16)
# x2 = torch.rand(5, 10, 16)  # 经过某个网络变换后的输出
#
# # 前向传播
# output = add_and_norm(x1, x2)
#
# # 输出形状
# print("输出形状：", output.shape)  # 预期: (5, 10, 16)
