# Author: yixin_wang
# Time: 2022/5/12
"""
请利用文件“Molecular_Descriptor.xlsx”提供的729个分子描述符，针对文件“ADMET.xlsx”中提供的1974个化合物的ADMET数据，
分别构建化合物的Caco-2、CYP3A4、hERG、HOB、MN的分类预测模型.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from inputs import SparseFeat, DenseFeat, get_feature_names
from models import DeepFM


def sparse_or_dense(X):
    """
    Judge whether a feature is sparse or dense
    得到处理后的数据集，标记稠密特征&稀疏特征
    """
    X = np.array(X)
    sparse_cols = []
    dense_cols = []
    for i in range(X.shape[1]):
        col = X[:, i]
        if any(col % 1 != 0):
            dense_cols.append(i)
        else:
            sparse_cols.append(i)
    return sparse_cols, dense_cols


# 导入数据
df = pd.read_excel("Molecular_Descriptor.xlsx")  # (1974, 730)  [, index_col=0] 指定某列为行索引
data = df.iloc[:, 1:df.shape[1]]  # remove the first column with description of molecular
target = pd.read_excel("ADMET.xlsx")  # (1974, 6)
label = target.iloc[:, 1:target.shape[1]]  # remove the first column with description of molecular

# 特征选择 (PCA-kmeans/随机森林特征重要性排序)
fs_selector = VarianceThreshold(threshold=0.0)
# df = fs_selector.fit_transform(X=df.iloc[:, :])
# df = fs_selector.fit_transform(X=df.iloc[:, 1:df.shape[1]])  # remove the first column with description of molecular
# output -> ndarray: threshold=0.0 -> (1974, 504), threshold=0.1 -> (1974, 331)
fs_selector.fit(data)
columns_variances = fs_selector.variances_
# selected_fs = list(filter(lambda X: X != 0, columns_variances))  # -> (504) return variances value
selected_fs_index = [i for i in range(len(columns_variances)) if columns_variances[i] != 0]
data = data.iloc[:, selected_fs_index]
# 划分稀疏特征和稠密特征
sparse_cols, dense_cols = sparse_or_dense(data)  # -> list, index values

columns = data.columns.values  # nparray(["nAcid",	'ALogP', 'ALogp2', 'AMR'...])  [.tolist()]

sparse_features = columns[sparse_cols].tolist()
dense_features = columns[dense_cols].tolist()

data[sparse_features] = data[sparse_features].fillna('-1', )  # 字符型数据
data[dense_features] = data[dense_features].fillna(0, )

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))  # min, max = feature_range
data[dense_features] = mms.fit_transform(data[dense_features])

# 2.count #unique features for each sparse field,and record dense feature field name
# 理解下来就是创建每个特征域
# SparseFeat重命名的元组，可通过关键字访问(域)
# eg: SparseFeat(name='C1', vocabulary_size=27, embedding_dim=4, use_hash=False,
#                       dtype='int32', embedding_name='C1', group_name='default_group')
# eg: DenseFeat(name='I9', dimension=1, dtype='float32')
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] + \
                         [DenseFeat(feat, 1, ) for feat in dense_features]

# [SparseFeat(...), SparseFeat(...),...,DenseFeat(...), DenseFeat(...)...]
dnn_feature_columns = fixlen_feature_columns  # namedtuple
linear_feature_columns = fixlen_feature_columns
# 相同的两个列表加起来，里头是namedtuple对象实例 ↓
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
# ↑：['C1', 'C2', 'C3',..., 'C26', 'I1', 'I2', ..., 'I13']

# 3.generate input data for model
# train, test: DataFrame(200,40)
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=2020)  # 划分训练集和测试集，样本维度
# 字典{特征名: samples_values(Series:(160,1)}
train_model_input = {name: train_x[name] for name in feature_names}
# 字典{特征名: samples_values(Series:(40,1)}
test_model_input = {name: test_x[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
               task='binary', l2_reg_embedding=1e-5, device=device)

model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "precision"], )

history = model.fit(train_model_input, train_y.values, batch_size=32, epochs=10, verbose=2, validation_split=0.2)

pred_ans = model.predict(test_model_input, 32)  # 测试集

print("")
print("test LogLoss", round(log_loss(test_y.values, pred_ans), 4))  # 四舍五入4位小数
# print("test AUC", round(roc_auc_score(test_y.values, pred_ans), 4))