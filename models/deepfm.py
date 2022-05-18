# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Modifier:
    Yixin Wang, Shenzhen University
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
import torch
import torch.nn as nn

from models.basemodel import BaseModel
from inputs import combined_dnn_input
from layers import DNN, FM


class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        # 父类初始化
        # linear_feature_columns, dnn_feature_columns 重命名的元组序列
        # [SparseFeat(...), SparseFeat(...),...,DenseFeat(...), DenseFeat(...)...]
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)  # python2 写法？

        self.use_fm = use_fm  # bool
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            # 注意第二个维度是1，表示最后dnn输出只有一个值，如果改成多分类/多标签要修改一下， 1->5
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 5, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.to(device)

    def forward(self, X):

        # self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        # torch.Size([32, 39]) [样本数, 特征数]
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        logit = self.linear_model(X)  # 第一次输出  -> [样本数, 1]

        if self.use_fm and len(sparse_embedding_list) > 0:
            # 按最后的维度，对特征列表拼接[tensor1,...,tensor*]，26个特征tensor，每个tensor尺寸为：样本数×4
            # 拼接为一个大tensor，尺寸为[样本数, 稀疏特征数26, 4]
            fm_input = torch.cat(sparse_embedding_list, dim=1)  # 这里dim=1
            logit += self.fm(fm_input)  # 第二次输出  -> [样本数, 1]

        if self.use_dnn:
            # 展平输入 [样本数 * 117]
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)  # dnn_最后一层 -> [样本数, 128]
            # 1
            fm_dnn_output = logit + dnn_output  # [样本数, 1] + [样本数, 128] -> [样本数, 128]
            multilabel_logit = self.dnn_linear(fm_dnn_output)  # dnn_linear最后一层 -> [样本数, 5]
            # 2
            # dnn_logit = self.dnn_linear(dnn_output)
            # logit += dnn_logit

        # multilabel_logit/logit
        y_pred = self.out(multilabel_logit)  # 线性层输出(FM一阶交互) + fm输出(FM二阶交互) + dnn输出

        return y_pred
