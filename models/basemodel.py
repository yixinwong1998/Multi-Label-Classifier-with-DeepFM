# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Modifier:
    Yixin Wang, Shenzhen University
"""
from __future__ import print_function

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from inputs import build_input_features, SparseFeat, DenseFeat, create_embedding_matrix
from layers import PredictionLayer
from layers.utils import slice_arrays
from callbacks import History


class Linear(nn.Module):  # 就DeepFM，这部分应该是FM的一阶部分
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()

        self.feature_index = feature_index
        # OrderedDict([('C1', (0, 1)), ('C2', (1, 2)), ..., ('C26', (25, 26)), ('I1', (26, 27)), ('I2', (27, 28)),
        #      ..., ('I12', (37, 38)), ('I13', (38, 39))])   ↑
        self.device = device

        # feature_columns: [SparseFeat(...), SparseFeat(...),...,DenseFeat(...), DenseFeat(...)...]
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        # 线性层的embedding矩阵输入包括稀疏和稠密的两类，但是c_e_m只会对稀疏操作
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False, device=device)
        # embedding_dict ↓↓↓↓
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")
        # 注意隐向量维度为1，其他的默认的为4，因为这里linear=True

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
            # 初始化权重或者说是已经初始化的embedding向量

        # 只有稠密部分可训练
        # 这部分增加可训练可学习的参数有点不太懂
        if len(self.dense_feature_columns) > 0:
            # DenseFeat(name='I9', dimension=1, dtype='float32')
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(device))
            # Parameter的尺寸是[稠密特征数量 × 1] torch.Size([13, 1]) “I*”
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        # 取出X中稀疏向量的列传入到对应的嵌入层中去，最终出来一个list，看看他是怎么拼接的？
        # 这一部分应该是[*, *,...,*, *](一个特征对应一个元素)，注意这里是class Linear的self.embedding_dict
        # 最后尺寸就是[特征embedding],26个，每个embedding包含一个tensor，tensor维度(样本数 × 1 × 1)
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]
        # 不同上, 尺寸就是[特征values],13个, 每个values包含一个tensor, tensor维度(样本数 × 1)
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for
                            feat in self.dense_feature_columns]

        linear_logit = torch.zeros([X.shape[0], 1]).to(sparse_embedding_list[0].device)  # 初始化，每次从前往后训练的时候

        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            # 按最后的维度，对特征列表拼接[tensor1,...,tensor*]
            # 拼接为一个大tensor，尺寸为[样本数, 1, 稀疏特征数26]
            # 这里拼接的dim=-1

            # deepfm下面的if代码没有用到
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)

            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            # 压缩维度？keepdim=False会reshape保留没有操作的维度
            # 压缩为尺寸为[样本数, 1]的矩阵，累加样本的特征, FM 中的一阶特征累加

            linear_logit += sparse_feat_logit

        if len(dense_value_list) > 0:
            # matmul: Matrix product of two tensors.
            # cat()得到尺寸为[样本数, 稠密特征数13]的矩阵
            # [样本数, 稠密特征数13] × [稠密特征数13, 1] -> [样本数, 1]
            dense_value_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)  # 矩阵乘法
            linear_logit += dense_value_logit

        return linear_logit  # 返回尺寸为[样本数, 1]的tensor，累加样本的特征


class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        # linear_feature_columns, dnn_feature_columns 重命名的元组序列
        # [SparseFeat(...), SparseFeat(...),...,DenseFeat(...), DenseFeat(...)...]
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns
        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError("`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        # OrderedDict([('C1', (0, 1)), ('C2', (1, 2)), ..., ('C26', (25, 26)), ('I1', (26, 27)), ('I2', (27, 28)),
        #      ..., ('I12', (37, 38)), ('I13', (38, 39))])   ↑ return OrderedDict object, feature_index

        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(linear_feature_columns, self.feature_index, device=device)
        # 返回[样本数×1]的tensor, linear_model is learnable

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        # A `History` object. Its `History.history` attribute is a record of training loss values and metrics values
        # at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        self.history = History()  # from ..callbacks import History

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """

        # x: 字典{特征名: samples_values(Series:(160,1)}  特征数量39
        if isinstance(x, dict):
            # OrderedDict([('C1', (0, 1)), ('C2', (1, 2)), ..., ('C26', (25, 26)), ('I1', (26, 27)), ('I2', (27, 28)),
            #      ..., ('I12', (37, 38)), ('I13', (38, 39))])   ↑ return OrderedDict object, feature_index
            x = [x[feature] for feature in self.feature_index]  # [Series(160,1), Series(160,1),....]  特征数量39

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]
        # DeepFM模型中采用下面方式生成验证集
        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):  # 判断是否有“shape”属性或方法
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                # split_at: an integer index (start index) or a list/array of indices
                split_at = int(len(x[0]) * (1. - validation_split))
            # [Series(0:split_at_index,1), Series(0:split_at_index,1),....]  特征数量39
            x, val_x = (slice_arrays(x, 0, split_at), slice_arrays(x, split_at))
            # [Series(split_at_index:end,1), Series(0:split_at_index:end,1),....]  特征数量39
            y, val_y = (slice_arrays(y, 0, split_at), slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []

        for i in range(len(x)):
            if len(x[i].shape) == 1:  # Series只有一个尺寸
                x[i] = np.expand_dims(x[i], axis=1)  # 将每个特征变成一个ndarray，二维
                # x = [[训练数*1], [训练数*1], [训练数*1],...] 特征数量 : [array, array,...]

        # np.concatenate(): [array, array,...] -> [array] (训练样本数*特征数)
        train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y))
        # torch.utils.data.TensorDataset(*tensors)
        # Dataset wrapping tensors. in tuple ↓↓↓
        # Each sample will be retrieved by indexing tensors along the first dimension.
        # train_tensor_data[1] -> (tensor([0.0000e+00, 5.6000e+01, ...], dtype=torch.float64),
        #                          tensor([0]))
        # train_tensor_data[0:2] -> (tensor([[1.9000e+01, 3.6000e+01, ...],
        #                                    [0.0000e+00, 5.6000e+01, ...]], dtype=torch.float64),
        #                            tensor([[0],
        #                                    [0]]))

        if batch_size is None:
            batch_size = 256  # 默认批训练规模

        model = self.train()
        # model.train()和model.eval()的区别主要在于Batch Normalization和Dropout两层。
        # 如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。
        # model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        # model.train() ：启用 BatchNormalization 和 Dropout
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:  # 多个GPU同时训练，默认为None
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)  # 训练样本数
        # iteration
        # 一个epoch包含的步数（每一步是一个batch的数据送入）
        # 一旦決定好批次量，就可以算出會有幾個批次；同理，決定好一期(epoch)中要有幾個批次，也就可以算出批次量是多少。
        steps_per_epoch = (sample_num - 1) // batch_size + 1  # 參數更新是在每次batch結束

        # configure callbacks
        # 回调检查，回调(callbacks)是可以在训练的各个阶段(一轮epoch，一代iteration)执行动作的对象
        # 回调可以在每批训练后写TensorBoard日志以监控指标、保存、停止训练，利用回调保存最佳模型
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)  # CallbackList实例对象
        callbacks.set_model(self)  # self: BaseModel
        callbacks.on_train_begin()
        callbacks.set_model(self)  # self: BaseModel
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)  # 增加属性 self: BaseModel
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        # initial_epoch 默认为0
        for epoch in range(initial_epoch, epochs):  # 每一轮所有数据的训练，以下参数初始化重新来↓
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                # with ↓↓↓
                # 上下文管理器，具有 __enter__() 处理异常的准备工作 , __exit__() 处理异常
                # tqdm上下文管理器：
                #     def __enter__(self):
                #         return self
                #
                #     def __exit__(self, exc_type, exc_value, traceback):
                #         try:
                #             self.close()
                #         except AttributeError:
                #             # maybe eager thread cleanup upon external error
                #             if (exc_type, exc_value, traceback) == (None, None, None):
                #                 raise
                #             warn("AttributeError ignored", TqdmWarning, stacklevel=2)
                # t 接收 __enter__() 的返回
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:  # 每轮epoch下的输入次数 # DeepFM verbose=2
                    # with body 主要业务逻辑
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        # numpy.squeeze() 删除数组中所有单维条目，最终就剩下一维向量
                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        # 原本的reduction='sum'
                        loss = loss_func(y_pred, y.squeeze())  # 计算损失，使用nn.Functional的函数
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss  # self.aux_loss辅助损失DeepFM暂时没有用

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()  # 用total_loss 更新梯度
                        optim.step()

                        # param verbose: Integer. 0, 1, or 2.
                        # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                        if verbose > 0:  # DeepFM verbose=2
                            for name, metric_fun in self.metrics.items():  # items()返回可遍历的key/value对，{指标：指标函数}
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
            # 用户中断程序引发错误
            except KeyboardInterrupt:  # 可能发生的用户中断
                t.close()  # Cleanup and (if leave=False) close the progressbar.
                raise

            t.close()  # 无论怎样都会执行，有点相似finally

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:  # 是否执行验证集，默认为True
                eval_result = self.evaluate(val_x, val_y, batch_size)  # 传入验证集
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            if verbose > 0:  # DeepFM verbose=2
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(self, x, y, batch_size=256):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        # val_x, val_y  # 传入验证集
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():  # self.metrics {指标：指标函数}
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """

        # val_x, val_y  # 传入验证集
        # test_model_input  # 传入测试集
        model = self.eval()  # self.train(False) # model.eval() ：不启用 BatchNormalization 和 Dropout
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        # 最后尺寸就是[特征embedding],26个，每个embedding包含一个tensor，tensor维度(样本数 × 1 × 4)
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        # 尺寸就是[特征values],13个, 每个values包含一个tensor, tensor维度(样本数 × 1)
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):  # 正则化损失
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):  # 添加辅助损失
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer, loss=None, metrics=None, ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)  # 计算损失
        self.metrics = self._get_metrics(metrics)  # 获得一个字典  {指标：指标函数}

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            # 可以直接接收自定义的优化器
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
                # loss_func = F.multilabel_soft_margin_loss(reduction='mean')
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        #  输入: metrics=["binary_crossentropy", "auc"]
        metrics_ = {}
        if metrics:
            for metric in metrics:  # 可计算多个统计指标
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss  # sklearn内置
                        # metrics_[metric] = F.multilabel_soft_margin_loss(reduction='mean')
                if metric == "auc":
                    metrics_[metric] = roc_auc_score  # sklearn内置
                if metric == "mse":
                    metrics_[metric] = mean_squared_error  # sklearn内置
                if metric == "accuracy" or metric == "acc":
                    # 这里计算准确率，概率大于0.5则为1，小于0.5则为0
                    # y_true, y_pred 为 1D
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))  # sklearn内置
                if metric == "precision":
                    metrics_[metric] = lambda y_true, y_pred: precision_score(y_true, np.where(y_pred > 0.5, 1, 0), average='samples')
                self.metrics_names.append(metric)

        return metrics_

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(
            feature_columns) else []
        # set会去除重复元素
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]
