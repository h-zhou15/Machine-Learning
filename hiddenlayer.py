# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:23:09 2019

@author: hao

The subclass of the MLP 

The hidden layer class for MLP

including input layer and output layer

"""
import numpy as np
import pandas as pd
import math
from Node import*

class hiddenlayer(object):
    """
    hidden layer for MLP 
    describe this layer with last layer and the next layer
    Parameter:
    layer_nodes:the number of the node for this hidden layer
    Wij: the weight matrix from the last hidden layer to this hidden layer
        the dims depends on the nnodes of this layer and the nnodes of last layer
        eg. i nodes for last layer; j nodes for this layer
            the dims of W = Mat(i,j)
    active_func : active function , default ## sigmoid function
    ###default vector is row vector
    """
    def __init__(self,
                 layer_attribute='hidden',
                 layer_nodes=5,
                 rate=1,
                 lastlayer_nodes=3,
                 active_func='sigmoid'):
        self.layer_attribute=layer_attribute
        self.layer_nodes=layer_nodes
        self.active_func=active_func
        self.rate=rate
        self.node=[]
        self.lastlayer_nodes=lastlayer_nodes
        self.delta=np.zeros([self.layer_nodes,1])
        self.delta_Wij=[]
        self.set_nodes()

    def set_nodes(self):
        for i in range(self.layer_nodes):
            if (self.layer_attribute=='input'):
                self.node.append(Node(node_attribute=self.layer_attribute,
                                      Wij=1,
                                      Wjk=[],
                                      active_func=self.active_func))
            else:
                Wij_=np.random.uniform(-1,1,[self.lastlayer_nodes,1])
                #Wij_=0.5*np.ones([self.lastlayer_nodes,1])
                #print(Wij_)
                self.node.append(Node(node_attribute=self.layer_attribute,
                                      Wij=Wij_,
                                      Wjk=[],
                                      active_func=self.active_func))       
    # Wij 这里还有问题，理论上需要知道上一层的节点数，这个应该是先验知识，这里可能
    # 需要再重新考虑一下，该怎么构造。
    
    def output(self,x):
        '''
        x: last layer output,dims=(i,1), row vector, i indicates x with i features
        Wij: mat(i,j) defined ， is a matrix not a number 
        
        return 
        y should be same as x , dims =(j,1) 
        '''
        #(lastlayer_nodes,layer_nodes)=Wij.shape
        y=np.zeros([self.layer_nodes,1])
        if (self.layer_attribute=='input'):
            for j in range(self.layer_nodes):
                x_in=x[j]
                #print(j,'/',self.layer_nodes)
                y[j]=self.node[j].output(x_in)
        else:
        #for i in range(lastlayer_nodes):
            #print(self.layer_attribute)
            for j in range(self.layer_nodes):
                x_in=0
                for i in range(len(self.node[j].Wij)):
                    #print(i,'/',len(self.node[j].Wij))
                    x_in=x_in+self.node[j].Wij[i]*x[i]
                
                y[j]=self.node[j].output(x_in)
                #print('x_in:',x_in,'y[j]:',self.node[j].output(x_in))
        self.y=y
        #print(self.layer_attribute,'/y_out:',self.y)
        # self.y:dims=(j,1) , the output for every node of this layer
        return self.y
    
    def update_Wij(self,last_layer,next_layer,d):
        # HAPPY NATIONAL DAY
        '''
        d: the real output value
        w_: dims=(i,1), i is the number of last layer nodes 
        when this layer is the output layer, the next layer does not exist.
        so there is a problem. 
        
        '''
        # update the delta of this layer
        if (last_layer==[]):
            return
       # w_=np.zeros([last_layer.layer_nodes,1])
        self.get_layer_delta(next_layer,d)
        
        for j in range(self.layer_nodes):
            delta_w=np.zeros([last_layer.layer_nodes,1])
            w_=np.zeros([last_layer.layer_nodes,1])
            for i in range(last_layer.layer_nodes):
                #w_ is a row vector 
                delta_w[i]=self.rate*self.node[j].delta*last_layer.node[i].y
                w_[i]=self.node[j].Wij[i]+delta_w[i]
                #print('delta_1:',delta_w)
#            if (self.layer_attribute=='hidden'):
#                print(j,'node w:',w_)
            
            self.delta_Wij.append(delta_w)
            self.node[j].update_Wij(w_) # update the weight val of the jth node
        
#       emm , maybe it is better
    def update_Wjk(self,next_layer):
        # get the value to the node 
        # Wjk: jth node of l layer to the kth node of l+1 layer
        if next_layer==[]:
            return
        for j in range(self.layer_nodes):
            W_=np.zeros([next_layer.layer_nodes,1])
            for k in range(next_layer.layer_nodes):
                W_[k]=next_layer.node[k].Wij[j]
            self.node[j].update_Wjk(W_)
        
    def get_layer_delta(self,next_layer,d):
        '''
        calculate the error delta which is defined in p96. 模式识别
        last_layer is the hiddenlayer obj 
        
        '''
        if (self.layer_attribute=='input'):
            return
        
        if (self.layer_attribute=='output'):
            for j in range(self.layer_nodes):
                # output delta is different
                #print('j,node.y,d:',j,self.node[j].y,d[j])
                self.node[j].delta=self.node[j].y*(1-self.node[j].y)*(d[j]-self.node[j].y)
        #elif (self.layer_attribute=='hidden'):
        else:
            '''
            sth wrong here. Wjk unknown        
            2019/10/01
            maybe we could get Wjk with the layer relationship,
            unncessary to define the get_Wjk function
            '''
            # update every node for this layer
            #self.update_Wjk(next_layer)
            for j in range(self.layer_nodes):
                a=0
                for k in range(next_layer.layer_nodes):
                   # print('j',j,'||k/nodes:',k,'/',next_layer.layer_nodes)
                    #Wjk=self.node[j].Wjk[k]
                    Wjk=next_layer.node[k].Wij_old[j]
                    a=a+next_layer.node[k].delta*Wjk
                self.node[j].delta=self.node[j].y*(1-self.node[j].y)*a
    
    def get_Wij(self,last_layer):
        self.Wij=np.zeros([self.layer_nodes,last_layer.layer_nodes])
        for j in range(self.layer_nodes):
            for i in range(last_layer.layer_nodes):
                self.Wij[j,i]=self.node[j].Wij[i]
        return self.Wij
        
        