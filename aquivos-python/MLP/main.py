
import pandas as pd
import matplotlib.pyplot as plt
import  mlp

treino = pd.read_csv("dados\Trabalho Pratico - MLP - Classificação de Padrões - teste.csv")
X = treino[["x1","x2","x3","x4"]].values
Y = treino[["d1","d2","d3"]].values

teste = pd.read_csv("dados\Trabalho Pratico - MLP - Classificação de Padrões - teste.csv")
teste.drop("Unnamed: 7",axis=1, inplace=True)

xt=teste[["x1","x2","x3","x4"]].values
yt=teste[["d1","d2","d3"]].values

rede = mlp.MLP(X,Y,[15,3],learning_rate=0.1,precision=1e-6,activation_function=mlp.sigmoid,derivative_function=mlp.sigmoid_derivative)
Eqms = rede.train()

respostas=[rede.run(i) for i in xt]

resultado_parcial=pd.DataFrame(data=respostas,columns=['y1','y2','y3'])
Resultado= pd.concat([teste,resultado_parcial],axis=1)
print(Resultado)
plt.figure(figsize=(10,5))
plt.title("Eqm por epoca")
plt.plot(Eqms)
plt.show()