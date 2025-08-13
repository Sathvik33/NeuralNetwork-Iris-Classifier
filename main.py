from sklearn.datasets import load_iris
import torch
import numpy as np
import pandas as pd
import torch.nn  as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os,joblib
from joblib import dump, load

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

da=load_iris()

x=da.data
y=da.target


scaler=StandardScaler()
x=scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test  = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)


class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel,self).__init__()
        self.layer1=nn.Linear(4,20)
        self.layer2=nn.Linear(20,10)
        self.layer3=nn.Linear(10,3)

    def forward(self,x):
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        x=self.layer3(x)

        return x
        
        
model=IrisModel()

loss=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs=model(x_train)
    loss_value=loss(outputs,y_train)
    loss_value.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss_value.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred=model(x_test)
    y_p_c=torch.argmax(y_pred,dim=1)
    acc=(y_p_c==y_test).float().mean()

    print(f"Accuracy: {acc*100:.1f}")



torch.save(model.state_dict(), os.path.join(MODEL_DIR, "iris_model.pth"))
dump(scaler, os.path.join(MODEL_DIR, "scaler.save"))