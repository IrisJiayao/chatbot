from torch import nn
import torch
from time_distributed_linear import LinearLayer

class GLU(nn.Module):
    def __init__(self, input_size, hidden_layer_size,dropout_rate=None, use_time_distributed=True, batch_first=False):
        super(GLU,self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.activate_layer = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first=batch_first)
        self.gated_layer = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first=batch_first)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        if self.dropout_rate is not None:
            x = self.dropout(x)

        activation = self.activate_layer(x)
        gated = self.sigmoid(self.gated_layer(x))

        return torch.mul(activation, gated),gated


# # 创建 `GLU` 层
# glu = GLU(input_size=8, hidden_layer_size=16, dropout_rate=0.1)
#
# # 生成时间序列输入 `(batch_size=5, time_steps=10, feature_dim=8)`
# x = torch.rand(5, 10, 8)
#
# # 前向传播
# output, gate = glu(x)
#
# # 输出形状
# print("GLU 输出形状：", output.shape)  # 预期: (5, 10, 16)
# print("门控权重形状：", gate.shape)  # 预期: (5, 10, 16)
