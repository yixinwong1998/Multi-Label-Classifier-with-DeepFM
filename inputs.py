# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Modifier:
    Yixin Wang, Shenzhen University
"""


import torch
import torch.nn as nn
import numpy as np
from layers.utils import concat_fun
from collections import OrderedDict, namedtuple

DEFAULT_GROUP_NAME = "default_group"


# 继承namedtuple类
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):

    __slots__ = ()  # 实例对象不能添加属性

    # 实例创建对象的时候重写↓
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        # return namedtuple父类
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    # dict.keys() : 以列表返回一个字典所有的键
    return list(features.keys())


# def get_inputs_list(inputs):
#     return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def build_input_features(feature_columns):  # feature_columns -> list
    # Return OrderedDict: {feature_name:(start, start+dimension)}
    #  OrderedDict是记住键首次插入顺序的字典。如果新条目覆盖现有条目，则原始插入位置保持不变。
    features = OrderedDict()  # 顺序敏感
    # OrderedDict([('C1', (0, 1)), ('C2', (1, 2)), ..., ('C26', (25, 26)), ('I1', (26, 27)), ('I2', (27, 28)),
    #      ..., ('I12', (37, 38)), ('I13', (38, 39))])
    start = 0
    for feat in feature_columns:
        feat_name = feat.name  # namedtuple.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)  # dimension是DenseFeat的属性
            start += feat.dimension
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        # ↑ [样本数 * 104]
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        # ↑ [样本数 * 13]
        return concat_fun([sparse_dnn_input, dense_dnn_input])  # ← [样本数 * 117]
        # def concat_fun(inputs, axis=-1):
        #     if len(inputs) == 1:
        #         return inputs[0]
        #     else:
        #         return torch.cat(inputs, dim=axis)
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
    # lambda作为一个表达式，定义了一个匿名函数
    # filter(函数，序列)函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新对象
    # 对于每个特征（也就是视为特征域）然后去构造一个特征域的嵌入层，再把所有的特征域以ModuleDict汇总起来，可索引
    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in sparse_feature_columns}
    )
    # embedding_size: 用于linear层时隐向量维度为1，其余一般默认为4，这好像也可以针对不同的问题进行修改
    # 这里embedding_size属于一个行为属性，见basemodel中的@property embedding_size()
    # embedding 本例是对特征做embedding，（降维，转换，提取），不同的特征尺度范围不同，vocabulary_size就代表在该特征下有多少种类或者说跨度范围多少
    # NLP词嵌入中是将每个“词”做一个特征然后去做词嵌入

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


def get_dense_input(X, features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(
        x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        lookup_idx = np.array(features[fc.name])
        input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].float()
        dense_input_list.append(input_tensor)
    return dense_input_list

