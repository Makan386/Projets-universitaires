
"""TRAORE_Makan_M2MAS_DermMNIST_classification_202223.ipynb


# Projet M2MAS "Réseaux de Neurones profonds pour l'Apprentissage"

# Classification of the DermMNIST database

*Reference:*
 - [MedMNIST v2](https://medmnist.com/): A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification, Jiancheng Yang,Rui Shi,Donglai Wei,Zequan Liu,Lin Zhao,Bilian Ke,Hanspeter Pfister,Bingbing Ni, 2021.
Paper [arXiv](https://arxiv.org/pdf/2110.14795.pdf) Code [GitHub](https://github.com/MedMNIST/MedMNIST)
"""

!pip install medmnist

"""Import necessary libraries"""

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import medmnist
print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""Downlaod dataset and define dataloaders"""

info = medmnist.INFO['dermamnist']# info = dictionnaire

for cell in info:
  print(cell)
  print(info[cell])
  print('\n')

# Pour voir que le fichier n'est pas corrompu
# label est aussi un dictionnaire : un dictinnaire de dictionnaire (info[label[1]])
# On écrira la liste de code qui permet d'afficher info (question1)

BATCH_SIZE = 128

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
DataClass = getattr(medmnist, info['python_class'])
train_dataset = DataClass(split='train', transform=data_transform, download=True)
val_dataset = DataClass(split='val', transform=data_transform, download=True)
test_dataset = DataClass(split='test', transform=data_transform, download=True)

print(train_dataset)
print("===================")
print(test_dataset)
print("===================")
print(val_dataset)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_dataset
len(train_dataset)

"""Display function ```montage``` provided by MEDMNIST"""

train_dataset.montage(1)

train_dataset.montage(20)

"""Function to display a list of images (e.g. from a batch) :"""

def imshow(img):
    img = img.clone().detach().to('cpu')*0.5 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images:
print('Images:')
imshow(torchvision.utils.make_grid(images, nrow=12))

# show images for each class:
for k in info['label']:
  print("Class:", k, "Name:", info['label'][k])
  # get some random training images
  # show images
  idx = (labels==int(k)).view(-1)
  print(idx.sum().item(), 'images')
  if idx.sum()>0:
    imshow(torchvision.utils.make_grid(images[idx,:,:,:], nrow=12))

"""## 1  Answer the questions with both a code cell and a text cell

  1. What is the size of an image from the dataset ?

  2. How many images are in each split train/val/test ?

  3. What is the number of classes ?

  4. Are the classes well-balanced in the training set ? (compute the number of images per class in the training set)

### 1
Chaque image est de taille 3x28x28.
"""

print(images.shape) # 128 = taille du batch, 3 = canaux , taille de chaque image = 28x28

"""### 2 .

Il y a 7007 images dans le train, 1003 dans l'ensemble de validation et 2005 dans celui de test .
"""

#print(train_dataset)
print(info['n_samples'])

"""### 3
 Il y a 7 classes .
"""

#4
len(info['label'])

info['label']

"""### 4  Les  classes sont très déséquilibrées."""

dataset_size = len(train_dataset)
classes = list(info['label'].values())
num_classes = len(classes)
img_dict = {}
for i in range(num_classes):
    img_dict[classes[i]] = 0

for i in range(dataset_size):
    img, label = train_dataset[i]
    img_dict[classes[int(label)]] += 1

img_dict

"""# A first neural network for classification:

We will classify the ```DermMNIST``` images using the following CNN architecture.


* CNN part: All 2D convolutions have kernel size 3x3.
  * A 2D convolution with 16 ouput chanels, followed by ReLU.
  * A 2D convolution with 16 ouput chanels, followed by ReLU.
  * A 2D max-pooling with size 2x2
  * A 2D convolution with 64 ouput chanels, followed by ReLU.
  * A 2D convolution with 64 ouput chanels, followed by ReLU.
  * A 2D convolution with 64 ouput chanels, followed by ReLU.
  * A 2D max-pooling with size 2x2
* Fully connected part:
  * A linear layer with output dimension 128, followed by ReLU.
  * A linear layer with output dimension 128, followed by ReLU.
  * A last linear layer

##  2.a:

For each step of the forward function of the CNN, precise the size of the batch of the tensor starting from a tensor of size $b \times  c \times h \times w$ where $c \times h \times w$ are the answers of question 1.1.

### Answer:

## Question 2.b:

Recall the name and the mathematical expression of the training loss that will be used to train the network.

### Answer:

With equation:
$$
\log(\exp(x))=x
$$


criterion( outputs, labels ) : Il s'agit de comprendre la formule mathématique qu'il y a derrière

Ils'agit de la fonction de perte Cross-Entropy définie ci-dessous :


$$
E(\textbf{W})= -\sum_{(x^i,d^i)} \bigg[ \ a_{d^i} - \log \ \Bigg( \ \sum_{k=1}^{K} \exp(a_{k} ) \Bigg) \bigg]
$$

où  $d^i$ est la classe de la donnée d'entrée $x^i$ et $K$ le nombre de classes.

##  3:

1. Define a class ```Net``` that implements in PyTorch the considered architecture for ```DermMNIST``` classification.

2. Check that the forward function is well-defined by applying.

### Answer 3.1
"""

#1 TODO : Define class Net here
# On définit la class Net

import torch
import torch.nn as nn # networks
import torch.nn.functional as F # fonctions (activation par exemple)


class Net(nn.Module): # net herite du réseau

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3,out_channels= 16,kernel_size= 3) # Convolution 1
        self.conv2 = nn.Conv2d(16, 16, 3)  # Convolution 2      ( 16= canal d'entrée,16 = canaux de sortie, 3 = taille du noyau )
        self.conv3 = nn.Conv2d(16, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3*3*64, 128)  #  Prémière couche linéaire
        self.fc2 = nn.Linear(128, 128)      # Deuxième couche linéaire
        self.fc3 = nn.Linear(128, 7)       # Couche linéaire finale

    def forward(self, x): #forward function
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x)) # x  de taille 28x28x3 à 26x26x16
        x = F.relu(self.conv2(x))# x de taille 26x26x16 à 24x24x16
        x = F.max_pool2d(x,(2,2)) #  x de taille 24x24x16 à 12x12x16
        x = F.relu(self.conv3(x)) #  x de taille 12x12x16 à 10x10x64
        x = F.relu(self.conv4(x)) #  x de taille 10x10x64 à 8x8x64
        x = F.relu(self.conv5(x)) #  x de taille 8x8x64 à 6x6x64
        x = F.max_pool2d(x,(2,2)) # x de taille 6x6x64 à 3x3x64
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension (écraser toutes les dimensions sauf celle du batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net() # On crée une instance de la classe Net (on appelle le constructeur)
print(net)

"""### Answer 3.2
 On teste la fonction forward du réseau sur un batch du trainset.
"""

out = net(images)
print(out.shape)
print(images.shape)
out

"""images est un  tenseur de taille bxcxwxh où b=128, c=3,w= 28, h= 28 où
b= batch (paquet), c= cannaux,w= width(largeur), h= height(hauteur). \
La fonction forward marche bien avec le batch de trainset, on peut donc faire passer nos données dans le réseau.

## 4:

The goal of this question is to train a neural network of your class ```model=Net()``` to classify the ```DermMNIST``` dataset.

Define a function ```train``` with the necessary arguments that:
 * runs for ```n_epochs``` epochs
 * uses the optimizer
    ```
    lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ```
 * at each epoch, computes the *epoch loss* (that is the mean of the training loss on each batch of the training set), and the end of each epoch, computes
the classification accuracy on the validation set.
 * print at the end of the training the time spent for training **in minutes and seconds**.
 * displays at the end of the training the two plots of running loss VS epochs and accuracy on validation set VS epochs.
 * displays at the end of the training a classification report and a confusion matrix on the validation set using scikit-learn.

Apply your function to train for ```n_epochs = 6```.
"""

model=Net()
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

from matplotlib import axes
# TODO: training
# model=net
def train(model, criterion, optimizer, num_epochs=6):
    since = time.time()
    losses = []
    accuracy = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0 # Pour calculer la perte
        running_corrects = 0 # pour calculer la précision

            # Iterate over data.
        for inputs, labels in train_loader:

                #.squeeze() returns a tensor with all the dimensions of input of size 1 removed
                inputs = inputs
                lables =  labels.squeeze()
                # Need to sent inputs tensors to the GPU


                #labels = torch.from_numpy(labels )# on transforme labels(numpy.ndarray) en tenseur
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels.squeeze())
                 # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(train_dataset)#La perte sur chaque epoch
        losses.append(epoch_loss)
                # forward
        with torch.no_grad(): # pas de calcul de gradient sur l'ensemble validation
             for inputs, labels in val_loader:
                 inputs = inputs
                 labels = labels.squeeze()

                 outputs = model(inputs)
                 _, preds = torch.max(outputs, 1)

                 running_corrects += torch.sum(preds == labels.squeeze()) # .squeeze() <- .data






        epoch_acc = running_corrects.double() / len(val_dataset)#La précision sur chaque epoch
        accuracy.append(epoch_acc)


        print(' Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))



    # Précision du réseau net
    correct = 0
    total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
    alllabels = torch.tensor([])
    allpred = torch.tensor([])
    alllabels = alllabels.to(device)
    allpred = allpred.to(device)
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            labels = labels.to(device)
            alllabels = torch.cat([alllabels,labels])
        # calculate outputs by running images through the network
            outputs = model(images.to(device))
        # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.squeeze(), 1)
            allpred= torch.cat([allpred.to(device),predicted.to(device)])
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    print(f'La Précision du réseau sur les 2005 images de test est: {100 * correct // total} %')







    fig = plt.figure( figsize=(15, 15))
    ax = fig.add_subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ax.plot(np.arange(num_epochs) ,losses)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(np.arange(num_epochs), torch.Tensor(accuracy))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')


    print(classification_report(alllabels.cpu(), allpred.cpu(), target_names =  None))


    cm = confusion_matrix(alllabels.cpu(),allpred.cpu())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=None)
    disp.plot()



    return model

model = train(model, criterion, optimizer, num_epochs=6)
#print(model)

"""##  5:

Define a new function called ```train_gpu``` that does the same as the ```train``` function above with the training executed on the GPU.

Discuss before all the changes that needs to be done.

Report and discuss the difference of execution times.

### Answer:
"""

model=Net()
model = model.to(device)# On passe le modèle dans le gpu
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# TODO: training

def train_gpu(model, criterion, optimizer, num_epochs=6):
    since = time.time()

    losses = []
    accuracy = []
    #losses = losses.to(device)
    #accuracy = accuracy.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0 # Pour calculer la perte
        running_corrects = 0.0 # pour calculer la précision

            # Iterate over data.
        for inputs, labels in train_loader:

               # On envoie les inputs et les labels vers le GPU
                inputs = inputs.to(device)
                labels =  labels.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)


                loss = criterion(outputs, labels.squeeze())
                 # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(train_dataset)#La perte sur chaque epoch
                epoch_acc = running_corrects / len(val_dataset)#La précision sur chaque epoch


        losses.append(epoch_loss)
               # forward
        with torch.no_grad(): # pas de calcul de gradient sur l'ensemble validation
             for inputs, labels in val_loader:
                 inputs = inputs.to(device)
                 labels = labels.to(device)

                 outputs = model(inputs)
                 _, preds = torch.max(outputs, 1)

                 running_corrects += torch.sum(preds == labels.squeeze()) #




        epoch_acc = running_corrects.double() / len(val_dataset)#La précision sur chaque epoch
        accuracy.append(epoch_acc)


        print(' Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))


     # Précision du réseau net
    correct = 0
    total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
    alllabels = torch.tensor([])
    allpred = torch.tensor([])
    alllabels = alllabels.to(device)
    allpred = allpred.to(device)
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            labels = labels.to(device)
            alllabels = torch.cat([alllabels,labels])
        # calculate outputs by running images through the network
            outputs = model(images.to(device))
        # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.squeeze(), 1)
            allpred= torch.cat([allpred.to(device),predicted.to(device)])
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    print(f'La Précision du réseau sur les 2005 images de test est: {100 * correct // total} %')







    fig = plt.figure( figsize=(15, 15))
    ax = fig.add_subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ax.plot(np.arange(num_epochs) ,losses)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(np.arange(num_epochs), torch.Tensor(accuracy))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')


    print(classification_report(alllabels.cpu(), allpred.cpu(), target_names =  None))


    cm = confusion_matrix(alllabels.cpu(),allpred.cpu())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=None)
    disp.plot()
    return model

model = train_gpu(model, criterion, optimizer, num_epochs=6)

"""La fonction train_gpu est deux fois plus rapide en terme de temps d'exécution que la fonction train.

## 6:

Do a new training on the GPU using 120 epochs (be careful to initialize a new network).

Discuss the performance of the final model.
"""

model = Net()
model = model.to(device)# On passe le modèle dans le gpu
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# TODO: training on GPU with 40 epochs:

model = train_gpu(model, criterion, optimizer, num_epochs=120)

"""Avec 120 epochs la précision n'est plus constante comme en les questions 4 et 5; elle s'améliore puis se dégrade à partir de 80 epochs.

# A second neural network for classification:

##  7:

We will now try to improve the architecture by introducing batch normalization layers within the network.

1.We define a new class ```Net_with_BN``` that adds a 2D batchnormalization layer between each 2D-convolution layer and ReLU activation layer (that is after the 2D convolution and before ReLU).

2. We train a model ```model_with_bn = Net_with_BN()``` using your training function ```train_gpu```.
Hint: You may complete your function with:
```
model_with_bn.train()  # before training
model_with_bn.eval()   # before prediction
```

3. Are the performances similar ? What is the interest of using batchnormalization layers ?



Mettre des bacth normalization entre les conv.

###  7.1
"""

#  Define class Net_with_BN:

import torch
import torch.nn as nn # networks
import torch.nn.functional as F # fonctions (activation par exemple)


class Net_with_BN(nn.Module): # net herite du réseau

    def __init__(self):
        super(Net_with_BN, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3,out_channels= 16,kernel_size= 3) # Convolution 1
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3)  # Convolution 2      ( 16= canal d'entrée,16 = canaux de sortie, 3 = taille du noyau )
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 64, 3)
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.batch4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.batch5 = nn.BatchNorm2d(64)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3*3*64, 128)  #  Prémière couche linéaire
        self.fc2 = nn.Linear(128, 128)      # Deuxième couche linéaire
        self.fc3 = nn.Linear(128, 7)       # Couche linéaire finale

    def forward(self, x): #forward function
        x = F.relu(self.batch1(self.conv1(x)) )
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.batch3(self.conv3(x)))
        x = F.relu(self.batch4(self.conv4(x)))
        x = F.relu(self.batch5(self.conv5(x)))
        x = F.max_pool2d(x,(2,2))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension (écraser toutes les dimensions sauf celle du batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model_with_bn = Net_with_BN() # On crée une instance de la classe Net (on appelle le constructeur)
print(model_with_bn)

model_with_bn = Net_with_BN()

model_with_bn = model_with_bn.to(device)
lr = 0.01
optimizer = optim.SGD(model_with_bn.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

"""###  7.2"""

from numpy.ma.core import append


def train_gpu(model, criterion, optimizer, num_epochs=6):
    since = time.time()

    losses = []
    accuracy = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0 # Pour calculer la perte
        running_corrects = 0.0 # pour calculer la précision

            # Iterate over data.
        model.train()
        for inputs, labels in train_loader:

               # On envoie les inputs et les labels vers le GPU
                inputs = inputs.to(device)
                labels =  labels.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)

                # using an 1D tensor instead of the one-hot code list
                #labels = torch.tensor(labels)
                #labels = torch.argmax(labels,axis=1)
                loss = criterion(outputs, labels.squeeze())
                 # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(train_dataset)#La perte sur chaque epoch
                epoch_acc = running_corrects / len(val_dataset)#La précision sur chaque epoch

        losses.append(epoch_loss)
                # forward
        model.eval()
        with torch.no_grad(): # pas de calcul de gradient sur l'ensemble validation
             for inputs, labels in val_loader:
                 inputs = inputs.to(device)
                 labels = labels.to(device)

                 outputs = model(inputs)
                 _, preds = torch.max(outputs, 1)

                 running_corrects += torch.sum(preds == labels.squeeze()) #




        epoch_acc = running_corrects.double() / len(val_dataset)#La précision sur chaque epoch
        accuracy.append(epoch_acc)

        print(' Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))



    # Précision du réseau
    correct = 0
    total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
    alllabels = torch.tensor([])
    allpred = torch.tensor([])
    alllabels = alllabels.to(device)
    allpred = allpred.to(device)
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            labels = labels.to(device)
            alllabels = torch.cat([alllabels,labels])
        # calculate outputs by running images through the network
            outputs = model(images.to(device))
        # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.squeeze(), 1)
            allpred= torch.cat([allpred.to(device),predicted.to(device)])
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    print(f'La Précision du réseau sur les 2005 images de test est: {100 * correct // total} %')







    fig = plt.figure( figsize=(15, 15))
    ax = fig.add_subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ax.plot(np.arange(num_epochs) ,losses)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(np.arange(num_epochs), torch.Tensor(accuracy))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')


    print(classification_report(alllabels.cpu(), allpred.cpu(), target_names =  None))


    cm = confusion_matrix(alllabels.cpu(),allpred.cpu())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=None)
    disp.plot()



    return model

# Train a model_with_bn = Net_with_BN():
model_with_bn = train_gpu(model_with_bn, criterion, optimizer, num_epochs=120)

"""### 7.3
Le batch normalization accélère l'entrainement. On remarque aussi qu'à un certains nombres d'epochs la précision ne se dégrade plus.

## 8:
What is the proper way to determine which model is the best between the trained ```model``` from class ```Net()``` and the trained ```model_with_bn``` from class ```model_with_bn = Net_with_BN()``` ?


### Answer:
On compare les deux modèles en utilisant les matrices de classifications et de confusions sur l'ensemble test.
"""

# Précision du réseau model
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
alllabels = torch.tensor([])
allpred = torch.tensor([])
alllabels = alllabels.to(device)
allpred = allpred.to(device)
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        labels = labels.to(device)
        alllabels = torch.cat([alllabels,labels])
        # calculate outputs by running images through the network
        outputs = model(images.to(device))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.squeeze(), 1)
        allpred= torch.cat([allpred.to(device),predicted.to(device)])
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

print(f'La Précision du réseau sur les 2005 images de test est: {100 * correct // total} %')

print(classification_report(alllabels.cpu(), allpred.cpu(), target_names =  None))

cm = confusion_matrix(alllabels.cpu(),allpred.cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=None)
disp.plot()

# Précision du réseau model_with_bn
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
alllabels1 = torch.tensor([])
allpred1 = torch.tensor([])
alllabels1 = alllabels1.to(device)
allpred1 = allpred1.to(device)

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        alllabels1 = torch.cat([alllabels1,labels])
        # calculate outputs by running images through the network
        outputs = model_with_bn(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.squeeze(), 1)
        allpred1= torch.cat([allpred1.to(device),predicted.to(device)])
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

print(f'La Précision du réseau sur les 2005 images de test est: {100 * correct // total} %')

print(classification_report(alllabels1.cpu(),allpred1.cpu(),target_names =  None))

cm = confusion_matrix(alllabels1.cpu(),allpred1.cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=None)
disp.plot()

"""### Answer:
Le modèle ```model_with_bn ```  est un  peu plus précis que ```model```.

# Bibliographie

https://arxiv.org/pdf/1502.03167v3.pdf
"""