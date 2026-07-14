import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import savgol_filter
from copy import deepcopy


class pppretreatment:

    def SG(self,data,w=5,p=1,d=0):#初始参数w=3;p=1;d=0
        """
        SG平滑 w代表窗口 p代表多项式 d代表导数
        多项式的阶数决定了用于你和的多项式的程度，更高的多项式阶可以捕获更复杂的趋势，但也有可能引入不必要的震荡
        d为0代表不求导，d为1代表1介导  过程即先平滑后求导
        """
        data_copy = deepcopy(data)
        if isinstance(data_copy, pd.DataFrame):
            data_copy = data_copy.values

        data_copy = savgol_filter(data_copy,w,polyorder=p,deriv=d)
        return data_copy


    def StandarScaler(self,data):
        scaler = MinMaxScaler()

        #使用StandarScaler 对数据进行标准化
        standardized_data = scaler.fit_transform(data)
        return standardized_data

    def area_normalize(self,data):
        data = np.array(data)
        row_sums = np.sum(data, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-8
        return data / row_sums

    def msc(self,input_data,reference=None):
        """
        多元散射校正（MSC）的实现
        :param input_data: numpy array,形状为（样本数，波长数）的光谱数据
        :param reference: 可选，用于校正的参考光谱，如果为None，则使用输入所有数据的平均光谱
        :return: 校正后的数据
        """
        if reference is None:
            reference = np.mean(input_data,axis=0)

        #初始化校正后的数据数组
        corrected_data = np.zeros_like(input_data)

        #对每个样本进行处理
        for i in range(input_data.shape[0]):
            #获取当前样本
            sample = input_data[i,:]

            #计算回归系数
            fit = np.polyfit(reference,sample,1,full=True)

            #应用矫正
            corrected_data[i,:] = (sample-fit[0][1]) / fit[0][0]

        return corrected_data

    def snv(self,input_data):
        #对每一行应用snv转换
        snv_transformer = (input_data - input_data.mean(axis=1,keepdims=True)) / input_data.std(axis=1,keepdims=True)
        return snv_transformer