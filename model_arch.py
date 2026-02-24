import torch.nn as nn
import torch.nn.functional as F
import torch

class AiDetectorCNN(nn.Module):
    def __init__(self):
        super(AiDetectorCNN, self).__init__()
        
        # Blocs de convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Utilisation de LeakyReLU pour éviter les "neurones morts"
        self.activation = nn.LeakyReLU(0.1)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(512 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Passage dans les couches avec LeakyReLU
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = self.pool(self.activation(self.bn4(self.conv4(x))))
        x = self.pool(self.activation(self.bn5(self.conv5(x))))
        
        x = x.view(x.size(0), -1) # Flatten dynamique
        
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        
        # IMPORTANT : On ne met PAS de Sigmoid ici si tu utilises BCEWithLogitsLoss
        # Si tu veux des probabilités (0 à 1) après, tu feras torch.sigmoid(output) 
        # uniquement lors de l'inférence (test).
        x = self.fc2(x)
        return x