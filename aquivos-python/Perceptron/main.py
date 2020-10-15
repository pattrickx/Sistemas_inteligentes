import pandas as pd
import numpy as np
from perceptron import Perceptron
treino = pd.read_csv('dados/dataset-treinamento.csv',sep=';',decimal=',')

treino.d = treino.d.astype('int')


Y = treino['d'].values
X = treino[['x1','x2','x3']].values



teste = pd.read_csv('dados/dataset-teste.csv',sep=';',decimal=',')

Resultado=teste.copy()
Treinamentos = pd.DataFrame()
for i in range(5):

    activation_function = Perceptron.sign_function
    perceptron = Perceptron(X,Y,0.01, activation_function)
    wi,wf,ep = perceptron.train(200000)
    wi = [round(x,4) for x in wi] 
    wf = [round(x,4) for x in wf]
    w = np.concatenate(([wi],[wf],[[ep]]),axis=1)
    
    Treino = pd.DataFrame(data=w,columns=['Wi0','Wi1','Wi2','Wi3','Wf0','Wf1','Wf2','Wf3','N_Epocas'],index=[f'T{i+1}'])
    Treinamentos= pd.concat([Treinamentos,Treino])

    resultado_parcial=pd.DataFrame(columns=[f'T{i+1}'])
    respostas=[]
    for j in teste.values:
        respostas.append(perceptron.evaluate(j))

    resultado_parcial=pd.DataFrame(data=respostas,columns=[f'T{i+1}'])
    Resultado= pd.concat([Resultado,resultado_parcial],axis=1)

print(Treinamentos)

print(Resultado)