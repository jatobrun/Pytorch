%matplotlib in line
%config InlineBacked.figure_format = 'retina'

import numpy as np 
import torch 
import helper 
import matplotlib.pyplot as plt

from torchvision import datasets, transoforms

#Defino todas las trasnformaciones que le hare a mi dataset de imagenes
transforms = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.5,),(0.5,))])

#Descargo el dataset con el que trabajaremos para entrenar la red neuronal
trainset = datasets.MNIST('/home/andres/Imágenes/deeplearning', download = True, train = True, transforms = transforms)


#Creamos un contenedor de 64 imagenes, esto ayuda a la hora de pasarle las imagenes a la red neuronal, se demora mas en procesar las 400 o 10000 imagenes de golpe a si le mando 64 imagenes al azar para entrenar del mismo dataset
trainloader = torch.utils.data.DataLoarder(trainset, batch_size=64, shufle = True)

#creamos un iterador para visualizar nuestra data 
dataiter = iter(trainloader)

image, label = dataiter.next()

print(type(image))#como salida debe mostar que es un tensor 
print(image.shape)#las dimensiones varian en cada imagen ([64,1,28,28]) el 64 significa 64 imagenes por batch, el 1 es el canal de color, 28 el ancho y 28 el largo
print(label.shape)

#visualizamos nuestra data
plt.imshow(image[2].numpy().squeeze(), cmpa='Greys_r')


#Red Neuronal sin Pytorch

def activation(x):
	return 1/(1+torch.exp(-x))

def softmax(x)
	return torch.exp(x)/torch.sum(torch.exp(x), dim =1).view(-1,1)


inputs = image.view(images.shape[0], -1) 

n_inputs = 28*28
n_hidden =256
n_output = 10


W1 = torch.rand((n_inputs, n_hidden))
W2 = torch.rand((n_hidden, n_output))

B1 = torch.rand((1, n_hidden))
B2 = torch.rand((1, n_output))


l1 = activation(torch.mm(inputs, W1)+B1)
l2 = torch.mm(l1, W2)+B2
prob = softmax(l2)
#el modelo sigue sin poder aprender

#Red Neuronal con Pytorch

from torch import nn

class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden =nn.Linear(784, 256)
		self.out = nn.Linear(256, 10)
		self.sigmoid = nn.Sigmoid
		self.softmax = nn.Softmax(dim=1)


	def forward(self, x):
		x = self.hidden(x)
		x = self.sigmoid(x)
		x = self.out(x)
		x = self.softmax(x)
		return x
				

model = Network()
#el modelo sigue sin poder aprender


#Red Neuronal con Pytorch usando el modulo functional

import torch.nn.functional as F

class Network(nn.Module):

	def __init_(self):
		super().__init__()
		self.hidden = nn.Linear(784, 256)
		self.out = nn.Linear(256, 10)

	def forward(self, x):
		
		x = F.sigmoid(self.hidden(x))
		x = F.softmax(self.out(x), dim=1)

		return x


model = Network()
#el modelo sigue sin poder aprender


## Exercise 2 

import torch 
import torch.nn.functional as F
from torch import nn

class Network(nn.Module):
	def __init__(self):
		super().__init__
		self.fc1 = nn.Linear(784, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64,10)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.softmax(self.fc3(x), dim=1)
		return x


model = Network()

dataiter = iter(trainloader)
image, label = dataiter.next()

image.resize(64, 1, 28*28)#Flatteing

img_idx = 0
ps = model.forward(image[img_idx, :])

img = images[img_idx]

helper.view_classify(img.view(1,28,28), ps)
#el modelo sigue sin poder aprender
 

#Red Neuronal con Pytorch usando el modulo Sequential 
 
n_input = 784
n_hidden = [128,64]
n_output = 10

model = nn.Sequential(nn.Linear(n_input, n_hidden[0]),
					  F.relu(),
					  nn.Linear(n_hidden[0], n_hidden[1]),
					  F.relu(),
					  nn.Linear(n_hidden[1], n_output),
					  F.softmax(dim = 1))

print(model)

image, label = next(iter(trainloader))
image.resize(64, 1, 28*28) #Flattening
ps = model.forward(image[0,:])
helper.view_classify(image[0].view(1,28,28), ps)


#el modelo sigue sin poder aprender
	

print(model[0]) #puedo visualizar cada operacion de la red, la 0 es la primera capa, la 1 es el primer relu, la 2 es la segunda capa etc
print(model[0].weight) #visualizar el valor de los pesos de esa capa
print(model[0].bias) #visualizar el valor del bias de esta capa


#sino quiero nombrar cada operacion con 0 sino por nombre podemos usar OrderedDict

from collections import OrderedDict

model = nn.Sequential(OrderedDict([
					  ('fc1', nn.Linear(n_input, n_hidden[0])),
					  ('relu1', F.relu()),
					  ('fc2', nn.Linear(n_hidden[0], n_hidden[1])),
					  ('relu2', F.relu()),
					  ('fc3', nn.Linear(n_hidden[1], n_output)),
					  ('softmax', F.softmax(dim =1))
						]))

print(model)

print(model.fc1) # = print(model[0])











