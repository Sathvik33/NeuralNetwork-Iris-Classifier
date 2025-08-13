# NeuralNetwork-Bloom 🌸

**NeuralNetwork-Bloom** is a simple and interactive project that classifies iris flowers (and potentially other flowers) using a **Neural Network** built in **PyTorch**. It includes a **trained model**, **StandardScaler**, and a **Streamlit UI** for real-time predictions.  

---

## **Features**

- Neural Network classifier for iris flowers (Setosa, Versicolor, Virginica)  
- Data preprocessing using **StandardScaler**  
- Interactive **Streamlit UI** for entering flower measurements and getting predictions  
- Fully portable with saved model and scaler in `Model/` folder  
- Easy to extend for other datasets or flower classification tasks  

---

## **Project Structure**

├─ Model/ # Saved model and scaler
│ ├─ iris_model.pth
│ └─ scaler.save
├─ train_model.py # Script to train and save the model
├─ iris_app.py # Streamlit UI script
├─ requirements.txt # Python dependencies
└─ README.md

---
Training the Model
If you want to retrain the model:

bash


python train_model.py


This script will:

Load the Iris dataset

Preprocess the features using StandardScaler

Train a Neural Network (4 → 20 → 10 → 3) using PyTorch

Save the trained model as Model/iris_model.pth

Save the scaler as Model/scaler.save

Running the UI
bash
Copy code
streamlit run iris_app.py
The app will open in your browser.

Enter Sepal length, Sepal width, Petal length, Petal width.

Click Predict to see the predicted flower species.

A green success box (st.success) will display the prediction.

Example Input Values
Species	Sepal Length	Sepal Width	Petal Length	Petal Width
Setosa	5.1	3.5	1.4	0.2
Versicolor	6.0	2.9	4.5	1.5
Virginica	6.5	3.0	5.5	2.0

Technologies Used
Python 3.10+

PyTorch (Neural Network)

Scikit-learn (Dataset, preprocessing)

Joblib (Save/load scaler)

Streamlit (Interactive UI)

Usage
Train the model (optional if model already exists in Model/)

Run the Streamlit app

Enter measurements and click Predict

View the predicted flower species

Contributing
Feel free to submit pull requests or issues.

You can extend the project to classify other flowers or plants.

Add additional visualizations or deploy the app online.
yaml
Copy code
