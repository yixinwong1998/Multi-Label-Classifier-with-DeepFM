# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com
Modifier:
    Yixin Wang, Shenzhen University
"""
import numpy as np
import torch


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Arguments:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """

    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]  # 变成list类型

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):  # 判断传入的start是否一个[start:stop]列表，列表类型有__len__
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):  # 适应不同类型的数据，但是list一般没有shape属性
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]  # x[start] -> list里面的Series[start], start为list
        else:  # start 传入一个 int
            if len(arrays) == 1:
                return arrays[0][start:stop]  # stop为空默认到最后面
            return [None if x is None else x[start:stop] for x in arrays]  # x[start:stop] -> list里面的Series[start:stop]
    else:  # 输入x不是一个list，也不是ndarray 后续的代码可能要根据不同的数据进而分析
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]
