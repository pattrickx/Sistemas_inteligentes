
import pandas as pd
import numpy as np

class Adaline:
    def __init__(self,input_values,output_values,learn_rate,activation_function,precision=1e-6):
        ones_column = np.ones((len(input_values),1))*-1
        self.input_values=np.append(ones_column,input_values, axis=1)
        self.output_values=output_values
        self.learn_rate=learn_rate
        self.activation_function=activation_function
        self.W= np.random.rand(input_values.shape[1]+1)
        self.Wi= self.W.copy()
        self.precision = precision
        self.hist_EQM=[]
        # self.theta = np.random.rand(1)[0]
    def EQM(self,w):
        eqm = 0
        for values in zip(self.input_values, self.output_values):
            u = np.dot(values[0], w)
            eqm += (values[1] - u) ** 2
        return eqm/len(self.input_values)

    def evaluate(self,x):
        x= np.append([[-1]],[x], axis=1)
        u= np.dot(x,self.W)
        return self.activation_function(u)

    def train (self):
        epochs = 1
        eqmf=0
        while True:
           
            eqma = self.EQM(self.W)
            self.hist_EQM.append(eqma)
            for x,d in zip(self.input_values,self.output_values):
                u= np.dot(x,self.W)
                self.W=self.W+self.learn_rate*(d-u)*x

            epochs+=1

            eqmf = self.EQM(self.W)
            
            if abs(eqmf - eqma) <= self.precision:
                break
        self.hist_EQM.append(eqmf)
        return self.Wi,self.W,epochs,self.hist_EQM

    def binary_step(x):
        return 1 if x>=0 else 0
    def sign_function(x):
        return 1 if x>=0 else -1