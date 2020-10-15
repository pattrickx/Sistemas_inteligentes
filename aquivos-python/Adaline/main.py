import pandas as pd
import numpy as np
from adaline import Adaline
treino = pd.read_csv('dados/dataset-treinamento.csv')

treino['d']=treino['d'].astype('int')

teste = pd.read_csv('dados/dataset-teste.csv')

Y = treino['d'].values.copy()
X = treino[['x1','x2','x3','x4']].values.copy()

Resultado=teste.copy()
Treinamentos = pd.DataFrame()
hist_EQM =[]
for i in range(5):

    activation_function = Adaline.sign_function
    adaline = Adaline(X,Y,0.0025, activation_function)
    wi,wf,ep,EQM = adaline.train()
    hist_EQM.append(EQM)
    wi = [round(x,4) for x in wi] 
    wf = [round(x,4) for x in wf]
    w = np.concatenate(([wi],[wf],[[ep]]),axis=1)

    Treino = pd.DataFrame(data=w,columns=['Wi0','Wi1','Wi2','Wi3','Wi4','Wf0','Wf1','Wf2','Wf3','Wf4','N_Epocas'],index=[f'T{i+1}'])
    Treinamentos= pd.concat([Treinamentos,Treino])

    resultado_parcial=pd.DataFrame(columns=[f'T{i+1}'])
    respostas=[]
    for j in teste.values:
        respostas.append(adaline.evaluate(j))

    resultado_parcial=pd.DataFrame(data=respostas,columns=[f'T{i+1}'])
    Resultado= pd.concat([Resultado,resultado_parcial],axis=1)

print(Treinamentos)

print(Resultado)
import matplotlib.pyplot as plt
plt.figure(figsize=(4,8))
for i in range(5):
    plt.subplot(5,1,i+1)
    plt.plot(hist_EQM[i])
    plt.title(f"T{i+1}")
plt.subplots_adjust(hspace=0.7)
plt.show()