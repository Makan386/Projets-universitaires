
"""TRAORÉ_Makan_ M1MAS_FashionMNIST_classification_202122.ipynb

#Classification de la base FashionMNIST par réseau de neurones
"""


# %matplotlib inline

"""On utilise ``torchvision `` pour télécharger la base de données FashionMNIST"""

import torch
import torchvision
import torchvision.transforms as transforms

"""### 1 :
En utilisant la classe
```
CLASS torchvision.datasets.FashionMNIST
```
documentée ici :

https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST

on définit les objets `trainset`, `trainloader`, `testset`, `testloader` et `classes` en définissant une fonction transform qui ramène les niveaux de gris dans l'intervalle $[-1,1]$.

 Et on Affiche quelques images avec leur classe correspondante.
"""

#Normalisation des données
transform =  transforms.Compose(
           [transforms.ToTensor(), # conversion en tenseur sur pytorch ( entre [0,1])
            transforms.Normalize((0.5), (0.5))] )# pour etre symétrique par rapport à l'origine (niveau de gris entre [-1,1])

# On définit les objets trainset, trainloader, testset, testloader et classes

batch_size = 4 # donne un paquet de 4 images

trainset1 = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform) #trainset
trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=batch_size,
                                          shuffle=True, num_workers=2)      #trainloader

testset1 = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)  # testset
testloader1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size,
                                         shuffle=False, num_workers=2)      #testloader



classes1 = ('T-shirt', 'Trouser', 'Pull', 'Dress','Coat',
           'Sandal', 'Shirt','Sneaker','Bag','Ankle Boot')  #classes

#  Affichage des images avec les classes correspondantes
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     #  on divise par 2 puis on ajoute 0.5 pour  revenir entre [-1,1]
    npimg = img.numpy()    #
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # mettre les canaux dans le bon sens
    plt.show()


# get some random training images
dataiter = iter(trainloader1)
images, labels = next(dataiter)  # donne le prochain bacth , on obtient 4 images

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes1[labels[j]]:5s}' for j in range(batch_size)))

"""### 2 :

Description quantitative de la base de données en répondant aux questions suivantes :

* Quelle est la taille de chaque image ?
* Combien y a-t-il d'images au total pour `trainset` et `testset` ?
* Combien d'images par classe ?
* Combien de classes ?

On pourra proposer un script pour répondre aux questions.

FashionMNIST est une base de données de 60000 images d'entrainement et 10000 images de test, toutes de tailles 28x28 en niveau de gris.\
On dénombre 10 classes , avec 7000 images par classe (On peut le voir sur la matrice de confusion).
"""

# Nombre d'image pour le trainset
print(trainset1) # 60000 images

# Nombre d'image pour le testset
print(testset1)  # 10000 images

# La taille des images

print(images.shape) # 4 = taille du batch, 1 = canal , taille de chaque image = 28x28

"""## Réseau de neurones convolutionnel (CNN)

On souhaite définir un réseau de neurones convolutionnel (CNN) qui s'applique aux images de `FashionMNIST`et effectue la suite d'opérations suivante :

**Partie CNN :**
 * Une convolution 2D de noyau de taille 5x5 avec 8 canaux en sortie, suivie d'une activation  ReLU.
 * Un max-pooling 2D de taille 2x2
 * Une convolution 2D de noyau de taille 5x5 avec 18 canaux en sortie, suivie d'une activation  ReLU.
 * Un max-pooling 2D de taille 2x2

**Partie linéaire :**
 * Une couche linéaire avec dimension de sortie 100, suivie d'une activation  ReLU.
 * Une couche linéaire avec dimension de sortie 60, suivie d'une activation  ReLU.
 * Une couche linéaire finale permettant la classification.

### 3 :


En prenant un batch de taille 4 , les dimensions des tenseurs ainsi que les opérations nécessaire pour passer de la partie CNN à la partie linéaire sont données en commentaire dans le code ci-dessous:

###Q 4 :
On définit une class `CNNet` qui implémente le modèle décrit ci-dessus.

Puis on définit un réseau (une instance de CNNet) cnnet = CNNet() et teste sa fonction `forward` sur un batch du `trainset`.

Afficher quel est le nombre de paramètres du réseau.

https://stackoverflow.com/questions/69652625/pytorch-mat1-and-mat2-cannot-be-multiplied \

pour résoudre le problème de dimension
"""

# On définit la class CNNet

import torch
import torch.nn as nn # networks
import torch.nn.functional as F # fonctions (activation par exemple)


class CNNet(nn.Module): # cnnet herite du réseau

    def __init__(self):
        super(CNNet, self).__init__()
        # 1 input image channel, 8 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=1,out_channels= 8,kernel_size= 5) # Convolution 1
        self.conv2 = nn.Conv2d(8, 18, 5)  # Convolution 2      ( 8= canal d'entrée,18 = canaux de sortie, 5 = taille du noyau )
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4*4*18, 100)  #  Prémière couche linéaire
        self.fc2 = nn.Linear(100, 60)      # Deuxième couche linéaire
        self.fc3 = nn.Linear(60, 10)       # Couche linéaire finale

    def forward(self, x): #forward function
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x)) # x  de taille 28x28x1 à 24x24x8
        x = F.max_pool2d(x,(2,2)) # x de taille 24x24x8 à 12x12x8
        # If the size is a square, you can specify with a single number
        x = F.relu(self.conv2(x)) # x de taille 12x12x8 à 8x8x18
        x= F.max_pool2d(x,2) # x de taille 8x8x18 à 4x4x18
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension (écraser toutes les dimensions sauf celle du batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnnet = CNNet() # On crée une instance de la classe Net (on appelle le constructeur)
print(cnnet)

# On teste la fonction forward du réseau sur un batch du trainset
out1 = cnnet(images)
print(out1)

# images est un  tenseur de taille bxcxwxh où b=4, c=1,w= 28, h= 28
#b= batch (paquet), c= cannaux,w= width(largeur), h= height(hauteur)
# La fonction forward marche bien avec le batch de trainset, on peut donc faire passer nos données dans le réseau

# Le nombre paramètres
params = list(cnnet.parameters()) # liste de tous les paramètres du réseau (poids)
#print(len(params))
nb_param= 0
for param in params :
         print(param.shape)
         nb_param += param.numel()
         print(nb_param)
# Il y a 39396 paramètres

"""##Entraînement du réseau

### 5 :
Un script qui entraîne le réseau `cnnet` sur 10 epochs avec affichage à chaque epochs de l'"epoch loss" (moyenne des loss sur chaque batch considérés dans l'epoch, soit la "running loss" accumulée sur toute l'epoch) et du temps total de l'entraînement.
"""

# Définition de la fonction de perte et de l'optimiseur

import torch.optim as optim
criterion = nn.CrossEntropyLoss() # fonction de perte
#Optimiseur
optimizer = optim.SGD(cnnet.parameters(), lr=0.001, momentum=0.9) # net.parameters() : variables à optimiser
#Gradient stochastique avec momentum:  momentum signifie mon gradient + 0.9 (page 16 cours4)

# Entraînement de cnnet et affichage du temps total de l'entraînement

import time
start = time.time()
for epoch in range(10):  # loop over the dataset multiple times (on fait dix epochs ici)

    running_loss = 0.0 # erreur d'app. a l'instant t
    for i, data in enumerate(trainloader1, 0): # on parcours la donnée d'entrainement
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data # analyse toutes les images et les indices correspondant

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnnet(inputs) #On passe les données dans le réseau
        loss = criterion(outputs, labels) # labels = vraies classes
        loss.backward()
        optimizer.step() # mettre à jour les paramètres

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches (pour tous les 2000 mini-batch on regarde l'erreur accumulée)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


end = time.time()
temps = end - start # temps total d'entrainement en séconde
temps
print('Finished Training')
print(f'Le temps total d entrainement est de: {temps/60} minutes')

"""
### Question 7 :
A l'aide des fonctions de ```sklearn```, On évalue la performance du réseau entraîné (rapport de classification et matrice de confusion).

"""

# Importation des fonctions de sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Précision du réseau cnnet
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs (torch.no_grad())
alllabels = torch.tensor([])
allpred = torch.tensor([])
with torch.no_grad():
    for data in testloader1:
        images, labels = data
        alllabels = torch.cat([alllabels,labels])
        # calculate outputs by running images through the network
        outputs = cnnet(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        allpred= torch.cat([allpred,predicted])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'La Précision du réseau sur les 10000 images de test est: {100 * correct // total} %')

"""La précision de ce réseau  est beaucoup mieux que le hasard, qui donne une précision de 10% . On peut donc dire qu'il a appris quelque chose."""

# Rapport de classification
print(classification_report(alllabels,allpred,target_names =  classes1))

# Matrice de confusion
cm = confusion_matrix(alllabels,allpred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes1)
disp.plot()

"""Dans l'ensemble on a une très bonne prédiction de chaque classe .

La classe ` Trouser ` est la meilleure classe en terme précision de la prédiction contrairement à la classe `shirt` .

## Comparaison avec un réseau à une couche cachée

Pour finir nous allons comparer la performance du réseau CNN avec un réseau simple à une seule couche cachée.

On va utiliser l'architecture suivante :

* Une couche linéaire avec dimension de sortie 400, suivie d'une activation  ReLU.
* Une couche linéaire finale permettant la classification.

### Performance du CNN avec une couche cachée

### 8 :
Définissons une class `MLP` qui implémente le modèle décrit ci-dessus.

On définit un réseau mlp = MLP() et teste sa fonction `forward` sur un batch du `trainset`.

On affichera aussi le nombre de paramètres du réseau `mlp`. 
"""

# Définition de la class MLP

class MLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.fc1 = nn.Linear(28*28,d) # Couche linéaire avec dimension de sortie 400
        self.fc2 = nn.Linear(d,10) # Couche linéaire finale


    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) # Couche 1 suivie d'une ReLU
        x =self.fc2(x)
        return (x)


mlp = MLP(d=400) # Le réseau mlp
print(mlp)

# On teste la fonction forward de mlp sur un batch du trainset.
out2 = mlp(images)
print(out2)

# Afficher le nombre de paramètres du réseau
params = list(mlp.parameters()) # liste de tous les paramètres du réseau (poids)
#print(len(params))
nb_param= 0
for param in params :
         print(param.shape)
         nb_param += param.numel()
         print(nb_param)

 # Il y a 318010 paramètres

a = mlp.fc1.weight # Les poids (tenseur de taille [400,28*28]) pour la prémière couche
b = mlp.fc1.bias # le biais de la prémière couche
print(a.shape)
print(b.shape)

#

"""### 9 :
 Entraînement le réseau `mlp` et évaluation de sa performance.

Comparaison des performances des deux réseaux entraînés.


"""

# Définition de la fonction de perte et de l'optimiseur
import torch.optim as optim

criterion1 = nn.CrossEntropyLoss() # fonction de perte
optimizer = optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9) # momentum(astuce d'accélération)

"""#### On entraîne le réseau mlp"""

start = time.time()
for epoch in range(10):  # loop over the dataset multiple times (on fait dix epochs ici)

    running_loss = 0.0 # erreur d'app. a l'instant t
    for i, data in enumerate(trainloader1, 0): # on parcours la donnée d'entrainement
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data # analyse toutes les images et les indices correspondant

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mlp(inputs)
        loss = criterion1(outputs, labels) # labels = vraies classes
        loss.backward()
        optimizer.step() # mettre à jour les paramètres

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches (pour tous les 2000 mini-batch on regarde l'erreur accumulée)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

end = time.time()
temps = end - start
temps
print('Finished Training')
print(f'Le temps total d entrainement est de: {temps/60} minutes')

# Précision du réseau mlp

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
alllabels1 = torch.tensor([])
allpred1 = torch.tensor([])
with torch.no_grad():
    for data in testloader1:
        images, labels = data
        alllabels1 = torch.cat([alllabels1,labels])
        # calculate outputs by running images through the network
        outputs = mlp(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        allpred1= torch.cat([allpred1,predicted])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'La Précision du réseau sur les 10000 images de test est: {100 * correct // total} %')

""" #### Performance du réseau mlp"""

# Rapport de classification
print(classification_report(alllabels1,allpred1,target_names =  classes1))

# Matrice de confusion
cm1 = confusion_matrix(alllabels1,allpred1)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=classes1)
disp1.plot()

"""Comme pour le réseau cnnet ,le réseau mlp fait beaucoup mieux que le hasard mais est moins précis que ce dernier.
Cependant , il donne une meilleure précision de prédiction de la classe `shirt` et un temps d'entraînement moindre par rapport au réseau cnnet .

# Bibliographie

https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html?msclkid=c3ea2f1abfe811eca2f95440aceef6f4
https://stackoverflow.com/questions/67466531/cnn-pytorch-only-batches-of-spatial-targets-supported-error

https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62

https://deeplizard.com/learn/video/MasG7tZj-hw
"""