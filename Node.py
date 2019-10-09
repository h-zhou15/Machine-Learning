# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:43:19 2019

@author: hao

the subclass of the hidden layer 
the node struct

"""
import numpy as np
import pandas as pd
import math

class Node(object):
    """
    the node struct for MLP and hidden layers
    the activation function is nolinear function, so the calculation between node
    and node is useless
     
    Parameter
    x_in: the input of this node including the linear combination of the all nodes
           of the last layers 
    y: the output of this node after applying the activation function 
    active_func: the str of the activation function; default is sigmoid
    Wij: the weight of the ith node of l-1 layer to the jth node of l layer,
         just a number , not a vector
    rate: the learning rate , default 0.1
    
    all parameter are defined according to 模式识别
    """
    
    def __init__(self,
                 node_attribute='hidden',
                 Wij=[],
                 Wjk=[],
                 active_func='sigmoid'):
        self.node_attribute=node_attribute
        self.Wij=Wij
        self.Wjk=Wjk
        self.Wij_old=Wij
        self.Wjk_old=Wjk
        self.active_func=active_func
        self.delta=0
        self.y=0
        
    def output(self,x_in):
        if (self.node_attribute=='input'):
            self.y=x_in
        else:
            #if (x_in>0 and self.node_attribute=='output'):
                #print('x_in:',x_in)
            self.y=self.activeFunc(x_in)
#        print('Wij:',self.Wij)
        #print('node:',self.node_attribute,'y',self.y,'x_in:',x_in,'Wij:',self.Wij)
        return self.y
    
    def update_Wij(self,w_):
        # w_ dims= (i,1), i is the nodes of l-1 layer
        self.Wij_old=self.Wij
        self.Wij=w_
        return self.Wij
    
    def update_Wjk(self,w_):
        self.Wjk_old=self.Wjk
        self.Wjk=w_
        return self.Wjk
    
    
    def activeFunc(self,x):
        if (self.active_func=='sigmoid'):
            a=1/(1+math.exp(-x))
        return a