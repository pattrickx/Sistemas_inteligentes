# -*- coding: utf-8 -*-
"""MLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w0SP4pwagjMU9qm7RQlBAomyPT-7-jLf
"""
import numpy as np
import math
def binary_step(u):
    return 1 if x>=0 else 0
def sign_function(u):
    return 1 if u>=0 else -1
def sigmoid(u):
    return 1/(1+np.exp(-u))
def sigmoid_derivative(u):
    return 1 - sigmoid(u)
def TanH(u):
    return (1-math.e**(-2*u))/(1+math.e**(-2*u))
def TanHDerivative(u):
    return 1-TanH(u)
    #return 1 - ((1-math.e**(-2*u))/(1+math.e**(-2*u)))


class MLP:

    def __init__(self,input_values,output_values,layers,learning_rate=1e-2,
                 precision=1e-6,activation_function=TanH,derivative_function=TanHDerivative):
       ones_column = np.ones((len(input_values), 1)) * -1
       self.input_values = np.append(ones_column, input_values, axis=1)
       
       self.output_values = output_values
       self.learning_rate = learning_rate
       self.precision = precision
       self.activation_function = activation_function
       self.derivative_function = derivative_function
       self.I = []
       self.Y = []
       self.W = []
       neuron_input = self.input_values.shape[1]
       for i in range(len(layers)):
            self.W.append(np.random.rand(layers[i], neuron_input))
            self.I.append(np.zeros(layers[i]))
            self.Y.append(np.zeros(layers[i]))
            neuron_input = layers[i] + 1
    
      
       self.epochs = 0
       self.eqms = []
    def run(self,x):
        y = self.propaga_total(np.append(-1,x))

        return [1 if 100*(y[0]/sum(y))>50 else 0,
               1 if 100*(y[1]/sum(y))>50 else 0,
               1 if 100*(y[2]/sum(y))>50 else 0]
    def train(self):
        print('iniciando treinamento')
        error = True
        eqm_actual = self.eqm()
        
        while error:
            error = False
            eqm_previous = eqm_actual
            # print(self.epochs)
            for x, d in zip(self.input_values,self.output_values):

                self.Y[-1] = self.propaga_total(x)
                self.retro_total(x,d)
                # self.retro_total()
                
            eqm_actual = self.eqm()
            self.eqms.append(eqm_actual)
            self.epochs+=1
            if abs(eqm_actual-eqm_previous)>self.precision:
                error=True
        print(self.epochs)
        return self.eqms 
    def propaga_camada(self,W,Y_anterior):
        i = np.dot(W, Y_anterior)
        Saida_atual = self.activation_function(i)
        return Saida_atual,i

    def propaga_total(self,x):
        Y=x.copy()
        for i,w in enumerate(self.W):
            Y,self.I[i] = self.propaga_camada(w,Y)
            if i <len(self.W)-1:
                Y = np.append(-1,Y)
            self.Y[i]=Y
        return self.Y[i]
        

    def retro_camada(self,w,w_old,y,d,u,x,out):
        if not out:
            y= y[1:]
            delta=sum(d*w_old)[1:]*self.derivative_function(u)
            delta=delta.reshape(len(delta),1)
        else:
            delta=((d-y)*self.derivative_function(u))
            delta=delta.reshape(len(delta),1)
            
        w += self.learning_rate*delta*x
        # d= np.dot(delta.T[0],w)

        return w,delta
    def retro_total(self,X,d):
        out=True
        w_old= 0
        for i,w in reversed(list(enumerate(self.W))):
            x=self.Y[i-1]
            if i == 0:
                x=X
            w,d = self.retro_camada(w,w_old,self.Y[i],d,self.I[i],x,out)
            out=False
            w_old = self.W[i]
    def eqm(self):
        eq = 0
        
        for x, d in zip(self.input_values, self.output_values):
            Y = self.propaga_total(x)
            eq += 0.5 * sum((d - Y) ** 2)
            
        return eq/len(self.output_values)