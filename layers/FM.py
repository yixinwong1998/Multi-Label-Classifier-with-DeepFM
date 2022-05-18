import torch
import torch.nn as nn


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        # embedding的输入
        # inputs尺寸：[样本数, 稀疏特征数26, 4]
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)  # 加和 -> 平方 -> [样本数, 1, 4]
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)  # 点乘(平方) -> 加和 -> [样本数, 1, 4]
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # 加和 -> [样本数, 1]

        return cross_term
