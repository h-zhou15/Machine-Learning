# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:46:02 2019

@author: hao

MLP:BP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hiddenlayer import*
from Node import*
from func import*

class MLP(object):
    """
    multiple-layer perceptron
    
    Parameter:
    nlayers: the number of layers including input layer
    
    """
    def __init__(self,nlayers=3,
                 input_nodes=11,
                 hidden_nodes=3,
                 output_nodes=2,
                 rate=1,
                 error_tolerance=0.0001,
                 MAX_ITER=2000,
                 train_filename='Train',
                 path='.\\res\\',
                 active_func='sigmoid'):
        self.nlayers=nlayers
        self.active_func=active_func
        self.input_nodes=input_nodes
        self.out_nodes=output_nodes
        self.hidden_nodes=hidden_nodes
        self.error_tolerance=error_tolerance
        self.rate=rate
        self.train_filename=train_filename
        self.path=path
        self.mse=[]
        self.CVmse=[]
        self.lastlayer_nodes=self.hidden_nodes
        self.define_MLP()
        self.MAX_ITER=MAX_ITER
    
    def define_MLP(self):
        # 1th layer is the input layer
        self.layer=[]
        for i in range(self.nlayers):
            if i==0:
                self.layer.append(hiddenlayer(layer_attribute='input',
                                              rate=self.rate,
                                              layer_nodes=self.input_nodes,
                                              active_func=self.active_func))
            elif (i==self.nlayers-1):
                self.layer.append(hiddenlayer(layer_attribute='output',
                                              rate=self.rate,
                                              layer_nodes=self.out_nodes,
                                              lastlayer_nodes=self.lastlayer_nodes,
                                              active_func=self.active_func))
            elif (i==1):
                self.layer.append(hiddenlayer(layer_attribute='hidden',
                                              rate=self.rate,
                                              layer_nodes=self.hidden_nodes,
                                              lastlayer_nodes=self.input_nodes,
                                              active_func=self.active_func))
            else:
                self.layer.append(hiddenlayer(layer_nodes=self.hidden_nodes,
                                              rate=self.rate,
                                              lastlayer_nodes=self.hidden_nodes,
                                              active_func=self.active_func))
                
    def train(self,x,d):
        '''
        x: input vector. row vector = (i,1) 增广向量
        d: the real output value
        
        '''
        # self.error=[]
        # forward propagating

        iter_num=0
        while 1:
            nums=x.shape[1]
            k=np.random.randint(0,x.shape[1],1)
            x0=x[:,k]
            d0=d[:,k]
            y0_out=self.forward_propagating(x0)
            #update Wij
            self.back_propagating(y0_out,d0)
            
            # test all samples
            (t,c)=self.scan_samples(x,d)
            self.mse.append(t/nums)
            
            if ((iter_num+1)%100==0):
                print(iter_num+1,'th iteration, the MSE is ',self.mse[iter_num])
            
            if (self.mse[iter_num]<=self.error_tolerance):
                print(iter_num+1,'th iteration, the MSE is ',self.mse[iter_num])
                print('Training Convergence')
                break
            iter_num+=1
            if (iter_num>=self.MAX_ITER):
                print('The maximum number of iterations has been reached')
                break
        self.learning_curve()
        
    def CV_train(self,x,d,n=4):
        iter_num=0
        nums=x.shape[1]
        l=(int)(nums/n)
        test_x_subset=x[:,0:l]
        test_y_subset=d[:,0:l]
        train_x_subset=x[:,l:nums]
        train_y_subset=d[:,l:nums]
        
        print('------Cross Validation begins-------')
        while 1:
            ntrain_samples=train_x_subset.shape[1]
            ntest_samples=test_x_subset.shape[1]
            for i in range(ntrain_samples):
                x0=train_x_subset[:,i]
                d0=train_y_subset[:,i]
                y0_out=self.forward_propagating(x0)
                self.back_propagating(y0_out,d0)
            (t,c)=self.scan_samples(test_x_subset,test_y_subset)
            self.CVmse.append(t/ntest_samples)
            if ((iter_num+1)%100==0):
                print(iter_num+1,'th CV iteration, the MSE is ',self.CVmse[iter_num])
            
            if (self.CVmse[iter_num]<=self.error_tolerance):
                print(iter_num+1,'th CV iteration, the MSE is ',self.CVmse[iter_num])
                print('CV Training Convergence')
                break
            iter_num+=1
            if (iter_num>=self.MAX_ITER):
                print('The maximum number of iterations has been reached')
                break
        self.learning_crve_CV()       
    
    def forward_propagating(self,x):
        '''
        get the value when given input 
        '''
        #y=[]
        #print('Forward propagation')
        for i in range(self.nlayers):
            if (self.layer[i].layer_attribute=='input'):
                #input layers
                y=self.layer[i].output(x)
                y_hidden=y
            elif (self.layer[i].layer_attribute=='output'):
                y_out=self.layer[i].output(y_hidden)
            else:
                y_hidden=self.layer[i].output(y_hidden)
            #print(i,'/',self.nlayers,'||y_hidden:',y_hidden)
        '''
        if (y_out<0.3):
            y_out=0
        elif (y_out>0.4):
            y_out=1
#        else:
        '''
        self.y=y_out
        
        return self.y
    
    def back_propagating(self,y_out,d):
        '''
        back_proagating processing:
        y_out is the output of the net at present
        d is the true value
        '''
        #print('Back propagation')
        #inverse scan
        # here, the last layer of the input layer is empty 
        for i in range(self.nlayers-1,-1,-1):
            #print(self.layer[i].layer_attribute)
            if (self.layer[i].layer_attribute=='output'):
                # the next layer of output layer is empty
                self.layer[i].update_Wij(self.layer[i-1],[],d)
            elif (self.layer[i].layer_attribute=='input'):
                self.layer[i].update_Wij([],self.layer[i+1],d)
            else:
                self.layer[i].update_Wij(self.layer[i-1],self.layer[i+1],d)
                
    def scan_samples(self,x,d):
        MSE_error=0
        y=np.zeros(d.shape)
        for i in range(x.shape[1]):
            #print('scan samples:',i,'/',x.shape[1])
            y[:,i]=self.forward_propagating(x[:,i])
            #MSE_error=MSE_error+(y[i]-d[i])**2
            MSE_error=MSE_error+self.cal_mse(y[:,i],d[:,i])
        return MSE_error,y
        
    # calculate the mse error for each sample
    def cal_mse(self,y,d):
        y_features=y.shape[0]
        y_error=0
        for j in range(y_features):
            y_error=y_error+(y[j]-d[j])**2
        return y_error
    
    def learning_curve(self):
        x=np.linspace(1,len(self.mse),len(self.mse))
        plt.figure(1)
        plt.ion()
        plt.plot(x,self.mse,'b')
        plt.title(str(self.nlayers)+' layers training MSE')
        plt.xlabel('Train times')
        plt.ylabel('Train error')
        #plt.pause(0.000001)
        
    def learning_crve_CV(self):
        x=np.linspace(1,len(self.CVmse),len(self.CVmse))
        plt.figure(2)
        plt.ion()
        plt.plot(x,self.CVmse,'b')
        plt.title(str(self.nlayers)+' layers CV training MSE')
        plt.xlabel('Train times')
        plt.ylabel('Train error')
        
 #   def display_output(self):
 
    def predict_MLP(self,x,d=[]):
        (a,y_out)=self.scan_samples(x,d)
        '''
        set threshold value
        '''
        y_threshold=0.5
        y_out[y_out<=y_threshold]=0
        y_out[y_out>y_threshold]=1
        
        self.predict=y_out
        print('prediction has been finished')
        (a,b,c)=cal_error_rate(y_out,d)
        if (d!=[]):
            plt.figure()
            plt.title('prediction result')
            for i in range(y_out.shape[1]):
                plt.plot(i,d[:,i],'bo')
                plt.plot(i,y_out[:,i],'rv')
            plt.legend(['True','Prediction'])
            plt.show()
    
    def get_mat_Wij(self):
        
        writer=pd.ExcelWriter(self.path+self.train_filename+'_Weight.xlsx')
        for i in range(self.nlayers):
            if (self.layer[i].layer_attribute!='input'):
                w=self.layer[i].get_Wij(self.layer[i-1])
                df=pd.DataFrame(w)
                df.to_excel(writer,sheet_name=self.layer[i].layer_attribute+str(i))
        writer.save()
    
    def save_data(self):
        self.get_mat_Wij()
        self.mse=np.array(self.mse)
        self.CVmse=np.array(self.CVmse)
        writer=pd.ExcelWriter(self.path+self.train_filename+'_error.xlsx')
        mse=pd.DataFrame(self.mse)
        CVmse=pd.DataFrame(self.CVmse)
        mse.to_excel(writer,sheet_name='mse')
        CVmse.to_excel(writer,sheet_name='CVmse')
        writer.save()