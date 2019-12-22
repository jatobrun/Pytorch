import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torch import nn
from torchvision import transforms, datasets
import torch.nn.functional as F
from google.colab import drive

drive.mount()

#DATA

transforms = transforms.Compose([torch.ToTensor(), torch.Normalize((0.5,),(0.5,))])

trainset = datasets.MNIST('/home/Documentos/MNIST', train = True, download = True, transforms = transforms)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle = True)


#BUILD FC NETWORK

model = nn.Sequential(nn.Linear(784,128), F.relu(), nn.Linear(128,64),
		      			F.relu(), nn.Linear(64,10))


#DEFINE LOSS
criterion = nn.CrossEntropyLoss()

images, labels = next(iter(trainloader))

images = images.view(images.shape[0], -1)

logits = model(images)

loss = criterion(logist, labels)

print(loss)



#Exercise

import torch 
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F


#dataset

transforms = transforms.Compose([torch.ToTensor(),
								torch.Normalize((0.5,),(0.5,))])


trainset = datasets.MNIST('/home/Andres/Documentos/MNIST', download=True, train=True, transforms = transforms)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)


#MODEL

model = nn.Sequential(nn.Linear(784, 128), F.relu(),
						nn.Linear(128,64), F.relu(),
						nn.Linear(64, 10), F.log_softmax())


criterion = nn.NLLLoss()


images, labels = next(iter(trainloader))

images = images.view(images.shape[0], -1)

result = model(images)

loss = criterion(result, labels)

print(loss)



loss.backward()



#Exercise


#Creaccion de la red neuronal
model = nn.Sequential(nn.Linear(784, 128), F.relu(), nn.Linear(128, 64), F.relu(), nn.Linear(64, 10), F.log_softmax())


criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)
epochs = 5

#Entrenamiento 
for e in epochs:
	running_loss = 0
	for images, labels in trainloader:
		image = images.view(64, 28*28)
		optimizer.zero_grad()
		output = model(image)
		loss = criterion(output, labels)
		optimizer.step()
		running_loss += loss.item()
	else:
		print(f"Training loss: {running_loss/len(trainloader)}")



#Validacion
%matplotlin inLine

import helper
images, labels = next(iter(trainloader))
img = images[0].view(1,784)
with torch.no_grad():
	logps = model(img)

ps = torch.exp(logps)

helper.view_classify(img.view(1,28,28), ps)

















































