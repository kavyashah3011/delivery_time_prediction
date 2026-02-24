# 🍔 Food Delivery Time Prediction System

A Machine Learning powered web application that predicts food delivery time based on distance, preparation time, weather conditions, traffic level, and vehicle type.

🔗 **Live Demo:**  
[🚀 Try the App](https://huggingface.co/spaces/kavya3011/Delivery-Time-Prediction)

---

## 📌 Project Overview

This project uses a **Linear Regression model** built with Scikit-Learn and deployed using **FastAPI**.

Users enter delivery conditions, and the system predicts the estimated delivery time in minutes.

---

## 🧠 Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- HTML / CSS / JavaScript
- Hugging Face Spaces (Deployment)

---

## ⚙️ Features

- 📊 Real-time delivery time prediction
- 📈 Model performance metrics (MAE, MSE, R²)
- 🌐 Public deployment
- 🎨 Clean responsive UI
- 🚀 Production-ready backend

---

## 🏗️ Project Structure

project/<br>
│<br>
├── app.py<br>
├── Food_Delivery_Times.csv<br>
├── requirements.txt<br>
├── Dockerfile<br>
│<br>
├── templates/<br>
│ └── index.html<br>
│<br>
└── static/<br>
├── css/style.css<br>
└── js/script.js<br>


---

## 📊 Machine Learning Model

- Algorithm: **Linear Regression**
- Feature Scaling: StandardScaler
- Encoding: One-Hot Encoding
- Train-Test Split: 80-20
- Evaluation Metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

---

## 🚀 How It Works

User Input<br>
↓<br>
Frontend (HTML + JS)<br>
↓<br>
FastAPI Backend<br>
↓<br>
ML Model Prediction<br>
↓<br>
Return JSON Response<br>
↓<br>
Display Result
