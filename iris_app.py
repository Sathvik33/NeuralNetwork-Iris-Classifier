# Save this as iris_app.py
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import load
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(4, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


model = IrisModel()

# Load trained model weights
model_path = os.path.join(MODEL_DIR, "iris_model.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

# Load scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.save")
scaler = load(scaler_path)


st.title("Iris Flower Classifier")

sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.0)
sepal_width  = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
petal_width  = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
  
    x_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    scaler = StandardScaler()
    from sklearn.datasets import load_iris
    iris = load_iris()
    scaler.fit(iris.data)
    x_input_scaled = scaler.transform(x_input)
    
    x_input_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        output = model(x_input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Species: **{species[predicted_class]}**")
