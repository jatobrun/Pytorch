import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np 


# Creando una funcion de activacion en este caso la sigmoid 
def activation(x):
	return 1/(1+torch.exp(-x))

features = torch.randn((1,4)) # Crea un tensor aleatorio de 1 fila 4 columnas funciona como input
weights = torch.rand_like(features) # Tambien necesitamos el tensor de pesos
bias = torch.randn((1,1)) # Creamos el valor del bias que no es nada mas que el interceptor de la recta

y = activation(torch.sum(features*weights)+bias) #la ecuacion respectiva de la recta la pasamos como parametro en la funcion de activacion con esto logramos obtener un valor entre 0-1 va bien para determinar una clase


y=activation(torch.mm(features, weights.view(4,1)).sum()+bias)


#CREANDO UNA RED CONVOLUCIONAL FC CON UNA HIDEN LAYER

inputs = torch.rand((1,6)) #Creando unas entradas aleatorias 

n_inputs = inputs[1] #Defino el numero de entradas en una variable 
n_hidden = 2 # Defino el numero de capas intermedias que tendra la red neuronal  
n_output = 1 # Defino el numero de salidas que tendra la ultima capa 

W1 = torch.rand((n_inputs, n_hidden)) # Pesos que obtienen las entradas al pasar hacia la capa oculta (hidden layer)

W2 = torch.rand((n_hidden, n_output)) #Pesos que obtienen desde la hidden layer hacia la output layer 

#Intercepto respectivo de cada transicion entre capa
B1 = torch.rand((1, n_hidden))
B2 = torch.rand((1, n_output))



y = activation(activation(torch.mm(torch.mm(inputs, W1)+B1)),W2)+B2) #con funcion de activacion en cada capa


## Seccion especial de Numpy-Tensor



vector_1 = np.random.rand(1,4)#creando un vector aleatorio con numpy
tensor_1 = torch.from_numpy(vector_1)#el vector creado lo convertimos a tensor

vector_2 = tensor_1.numpy()#volvemos a transformar el vector en tensor
#Ojo vector_1 y vector_2 son el mismo elemento si modifico vector_2 tambien cambia vector_1








