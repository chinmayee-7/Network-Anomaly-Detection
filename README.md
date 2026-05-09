# Network Anomaly Detection System

**A machine learning based web application that detects anomalies in network traffic using three unsupervised learning models.**

## About the Project

- **Network intrusion detection is a critical part of cybersecurity.**\
- **This project uses the KDD Cup 99 dataset to train three anomaly detection models that can classify network traffic as Normal or Anomaly.**

## Models Used

- **Isolation Forest** - detects anomalies by isolating data points in a tree structure
- **One Class SVM** - learns a boundary around normal traffic and flags anything outside
- **AutoEncoder** - reconstructs normal traffic and flags high reconstruction error as anomaly

## Dataset

**KDD Cup 99 Dataset - a standard benchmark dataset for network intrusion detection containing various types of network attacks like neptune, normal, smurf, portsweep, satan etc.**

## Project Structure
```
├── app.py                  → Streamlit web application
├── final_project.ipynb     → Model training notebook
├── requirements.txt        → Required libraries
├── IF_model.pkl            → Trained Isolation Forest model
├── SVM_model.pkl           → Trained One Class SVM model
├── autoencoder.pkl         → Trained AutoEncoder model
├── scaler.pkl              → Fitted Standard Scaler
├── pca.pkl                 → Fitted PCA
├── threshold.pkl           → AutoEncoder threshold value
├── rare_services.pkl       → Rare services list from training
├── constant_columns.pkl    → Constant columns removed during training
└── feature_columns.pkl     → Feature column order from training
```

## Steps to Run
pip install -r requirements.txt

### Run the app
streamlit run app.py


## How It Works

1. User enters network traffic details in the web form
2. Input goes through the same preprocessing as training data
3. All 3 models predict Normal or Anomaly
4. Results are displayed with color coding

## Tech Stack

- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Joblib