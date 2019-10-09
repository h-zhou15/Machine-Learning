# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 20:54:46 2019

@author: hao

ML Ex2 multiple-layer perceptron
"""
import pandas as pd
import numpy as np
from func import*
from MLP import*

path='E:\\2019-2020秋季学期\\机器学习\\上机练习\\Ex2\\'
# read csv file
# experiment 1
print('----------------- Experiment 1 -----------------')
TrainingSet1_feature_data=pd.read_csv(open(path+'data1\\train_10gene_sub.csv'))
TrainingSet1_label_data=pd.read_csv(open(path+'data1\\train_10gene_label_sub.csv'))
TrainingSet2_feature_data=pd.read_csv(open(path+'data1\\train_10gene.csv'))
TrainingSet2_label_data=pd.read_csv(open(path+'data1\\train_label.csv'))
TestSet_feature_data=pd.read_csv(open(path+'data1\\test_10gene.csv'))
TestSet_label_data=pd.read_csv(open(path+'data1\\test_label.csv'))
TestSet2_feature_data=pd.read_csv(open(path+'data1\\test2_10gene.csv'))
TestSet2_label_data=pd.read_csv(open(path+'data1\\test2_label.csv'))

(TrainingSet1_x,TrainingSet1_y)=df2np(TrainingSet1_feature_data,TrainingSet1_label_data)
(TrainingSet2_x,TrainingSet2_y)=df2np(TrainingSet2_feature_data,TrainingSet2_label_data)
(TestSet_x,TestSet_y)=df2np(TestSet_feature_data,TestSet_label_data)
(TestSet2_x,TestSet2_y)=df2np(TestSet2_feature_data,TestSet2_label_data)


TrainingSet1_MLP=MLP(nlayers=4,
                   hidden_nodes=5,
                   input_nodes=TrainingSet1_x.shape[0],
                   output_nodes=TrainingSet1_y.shape[0],
                   error_tolerance=0.0001,
                   rate=1,
                   MAX_ITER=5000,
                   train_filename='TrainingSet1',
                   path=path+'res\\')

TrainingSet1_MLP.train(TrainingSet1_x,TrainingSet1_y)
TrainingSet1_MLP.save_data()

TrainingSet1_CVMLP=MLP(nlayers=4,
                   hidden_nodes=5,
                   input_nodes=TrainingSet1_x.shape[0],
                   output_nodes=TrainingSet1_y.shape[0],
                   error_tolerance=0.0001,
                   rate=1,
                   MAX_ITER=5000,
                   train_filename='TrainingSet1_CV',
                   path=path+'res\\')
TrainingSet1_CVMLP.CV_train(TrainingSet1_x,TrainingSet1_y)
TrainingSet1_CVMLP.save_data()

#用testset和trainingset2来测试
TrainingSet1_MLP.predict_MLP(TrainingSet1_x,TrainingSet1_y)
TrainingSet1_MLP.predict_MLP(TrainingSet2_x,TrainingSet2_y)
TrainingSet1_MLP.predict_MLP(TestSet_x,TestSet_y)
TrainingSet1_MLP.predict_MLP(TestSet2_x,TestSet2_y)

TrainingSet1_CVMLP.predict_MLP(TrainingSet1_x,TrainingSet1_y)
TrainingSet1_CVMLP.predict_MLP(TrainingSet2_x,TrainingSet2_y)
TrainingSet1_CVMLP.predict_MLP(TestSet_x,TestSet_y)
TrainingSet1_CVMLP.predict_MLP(TestSet2_x,TestSet2_y)
#- experiment 2
print('----------------- Experiment 2 -----------------')
TrainingSet2_MLP=MLP(nlayers=4,
                     hidden_nodes=5,
                     input_nodes=TrainingSet2_x.shape[0],
                     output_nodes=TrainingSet1_y.shape[0],
                     error_tolerance=0.0001,
                     rate=1,
                     train_filename='TrainingSet2',
                     MAX_ITER=5000)
TrainingSet2_MLP.train(TrainingSet2_x,TrainingSet2_y)
TrainingSet2_MLP.save_data()    #存储结果

TrainingSet2_CVMLP=MLP(nlayers=4,
                     hidden_nodes=5,
                     input_nodes=TrainingSet2_x.shape[0],
                     output_nodes=TrainingSet1_y.shape[0],
                     error_tolerance=0.0001,
                     rate=1,
                     train_filename='TrainingSet2_CV',
                     MAX_ITER=5000)
TrainingSet2_CVMLP.CV_train(TrainingSet2_x,TrainingSet2_y,n=3)
TrainingSet2_CVMLP.save_data()

TrainingSet2_MLP.predict_MLP(TrainingSet1_x,TrainingSet1_y)
TrainingSet2_MLP.predict_MLP(TrainingSet2_x,TrainingSet2_y)
TrainingSet2_MLP.predict_MLP(TestSet_x,TestSet_y)
TrainingSet2_MLP.predict_MLP(TestSet2_x,TestSet2_y)

