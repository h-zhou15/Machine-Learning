# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:07:36 2019

@author: hao

include all functions which could be used for ex1

df2np: dataframe transfer to numpy array for ex1 data
test_perceptron: test the perceptron return error ratio for test set

数据集特点
x ： dims=(i,j) 表示有个i个特征，j个样本
y :  dims=(i,j) 表示输出有i个特征，总共有j个样本的输出

"""
import numpy as np
import pandas as pd

def df2np(df_feature,df_label):
    (x_nrows,x_ncols)=df_feature.shape
    (y_nrows,y_ncols)=df_label.shape
    
    x=np.ones([x_nrows+1,x_ncols-1])    #feature [1,x1,...,xnrows]
    y=np.zeros([y_nrows,1])
    
    for id_col in range(x_ncols-1):
        a=df_label.iloc[id_col,1]
        if a=='endothelial cell':
            y[id_col]=1 #endothelial cell class label=1 
        else:
            y[id_col]=0 #fibe class label=0
        for id_row in range(x_nrows):
            #x[id_row+1,id_col]=y[id_col]*df_feature.iloc[id_row,id_col+1]
            x[id_row+1,id_col]=df_feature.iloc[id_row,id_col+1]
    
    y=np.transpose(y)
    return x,y

def cal_error_rate(y,d):
    nums=d.shape[1]
    error_sample=0
    for i in range(nums):
        if y[:,i]!=d[:,i]:
            error_sample+=1
    print('error rate:',error_sample,' / ',nums,' = ',error_sample/nums)
    return error_sample/nums,error_sample,nums
