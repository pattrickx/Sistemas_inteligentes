"""# Perceptron"""
import pandas as pd
import numpy as np
class Perceptron:
    def __init__(self,input_values,output_values,learn_rate,activation_function):
        self.input_values=input_values
        self.output_values=output_values
        self.learn_rate=learn_rate
        self.activation_function=activation_function
        self.W= np.random.rand(input_values.shape[1])
        self.theta = np.random.rand(1)[0]
        
        self.Wi=np.concatenate(([self.theta],self.W))
    def evaluate(self,x):
        u= np.dot(x,self.W)-self.theta
        return self.activation_function(u)
    def train (self,n=-1):
        epochs = 1
        error = True
        # print(f'inicial W: {self.W}')
        # print()
        while error and (n>0 or n==-1):
            error = False
            # print('epoca',epochs)
            
            for x,d in zip(self.input_values,self.output_values):
                u= np.dot(x,self.W)-self.theta
                y= self.activation_function(u)
                
                # print(f'input: {x}, ouput: {y}, expected: {d}')

                if y!=d:
                    # print('recalculando W')
                    # print(f'atual W: {self.W}')

                    self.W=self.W+self.learn_rate*(d-y)*x
                    self.theta=self.theta+self.learn_rate*(d-y)*-1
                    error = True
                    
                    # print(f'novo W: {self.W}')
                    break
            if n >0:
                n-=1
            epochs+=1
            wf=np.concatenate(([self.theta],self.W))
        return self.Wi,wf,epochs
        print(f'final W: {self.W}')

    def binary_step(x):
        return 1 if x>=0 else 0
    def sign_function(x):
        return 1 if x>=0 else -1

