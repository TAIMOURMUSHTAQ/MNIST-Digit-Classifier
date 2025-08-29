# MNIST Handwritten Digit Recognition
A machine learning project that implements and compares various classification algorithms for recognizing handwritten digits from the famous MNIST dataset. This project demonstrates end-to-end workflow from data loading and preprocessing to model training, evaluation, and deployment.
The system achieves high accuracy in classifying handwritten digits (0-9) using traditional machine learning approaches, providing a solid foundation for understanding computer vision and pattern recognition concepts.

Key Features:
•	Multiple ML algorithm implementations (Random Forest, Logistic Regression, K-Neighbors)
•	Comprehensive model evaluation with visualization
•	Model persistence for deployment
•	Clean, reproducible code structure

Technologies Used: Python, Scikit-learn, NumPy, Matplotlib, TensorFlow/Keras
Dataset: MNIST (Modified National Institute of Standards and Technology) database of handwritten digits

## 📖 Overview
This project demonstrates the complete workflow of a machine learning system for digit recognition:
•	Data loading and visualization
•	Preprocessing and normalization
•	Model training with multiple algorithms
•	Performance evaluation and comparison
•	Model persistence for deployment
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. This project serves as an excellent introduction to image classification and computer vision concepts.

## ✨ Features
1.	**Multiple Algorithms**: Implementation of Random Forest, Logistic Regression, and K-Neighbors classifiers
2.	**Comprehensive Evaluation**: Accuracy scores, confusion matrices, and classification reports
3.	**Visualization**: Sample digit images with actual vs predicted labels
4.	**Model Persistence**: Saved model for deployment using joblib
5.	**Clean Code**: Well-structured and documented Jupyter notebook


## 🚀 Quick Start
### Installation
1. Clone the repository:
```bash
git clone https://github.com/TAIMOURMUSHTAQ /MNIST-Digit-Classifier.git
cd MNIST-Digit-Classifier
2.	Install required dependencies:
pip install -r requirements.txt
Usage
1.	Open and run the Jupyter notebook:
jupyter notebook MNIST_Handwritten_Digit_Recognition.ipynb
2.	Alternatively, run the Python script directly:
python mnist_digit_recognition.py

Requirements
The main requirements are:
•	Python 3.8+
•	numpy
•	matplotlib
•	scikit-learn
•	tensorflow (for loading MNIST dataset)
•	joblib

Install all requirements using:
pip install numpy matplotlib scikit-learn tensorflow joblib


📊 Results
The Random Forest classifier achieves approximately 97% accuracy on the test set. Sample results:
Model	Accuracy
Random Forest	~97%
Logistic Regression	~92%
K-Neighbors	~97%
Sample Prediction:


🏗️ Project Structure
MNIST-Digit-Classifier/
│
├── MNIST_Handwritten_Digit_Recognition.ipynb  # Main Jupyter notebook
├── mnist_digit_recognition.py                 # Python script version
├── mnist_digit_classifier.pkl                 # Saved model (after running)
├── requirements.txt                           # Project dependencies
└── README.md                                  # Project documentation


🔧 Implementation Details
Data Preprocessing
•	Images flattened from 28×28 to 784-dimensional vectors
•	Pixel values normalized to [0, 1] range by dividing by 255
•	Train-test split maintained from original MNIST dataset
Model Training
•	Random Forest: 100 estimators with default parameters
•	Logistic Regression: Multi-class classification with default parameters
•	K-Neighbors: K-nearest neighbors classifier
Evaluation Metrics
•	Accuracy score
•	Confusion matrix
•	Classification report (precision, recall, f1-score)


🎯 Usage Examples
Load and Use the Trained Model
import joblib
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the saved model
model = joblib.load("mnist_digit_classifier.pkl")
# Load and preprocess new data
(_, _), (X_test, y_test) = mnist.load_data()
X_test_flat = X_test.reshape(-1, 28*28) / 255.0
# Make predictions
predictions = model.predict(X_test_flat)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
Visualize Predictions
import matplotlib.pyplot as plt
# Visualize a test sample with prediction
index = 42  # Change to visualize different samples
plt.imshow(X_test[index], cmap='gray')
plt.title(f"Actual: {y_test[index]} | Predicted: {predictions[index]}")
plt.axis('off')
plt.show()


🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
1.	Fork the project
2.	Create your feature branch (git checkout -b feature/AmazingFeature)
3.	Commit your changes (git commit -m 'Add some AmazingFeature')
4.	Push to the branch (git push origin feature/AmazingFeature)
5.	Open a Pull Request

📧 Author Taimour Mushtaq 🎓 BSCS Student at Federal Urdu University of Arts,Science and Technology, Islamabad Pakistan 🔗 https://www.linkedin.com/in/taimourmushtaq/ |https://github.com/TAIMOURMUSHTAQ

