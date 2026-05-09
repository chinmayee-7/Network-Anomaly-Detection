import streamlit as st
import pandas as pd
import joblib
import numpy as np

rare = joblib.load('rare_services.pkl')
const = joblib.load('constant_columns.pkl')
columns = joblib.load('feature_columns.pkl')

scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
threshold = joblib.load('threshold.pkl')

IF_model = joblib.load('IF_model.pkl')
autoencoder = joblib.load('autoencoder.pkl')
SVM_model = joblib.load('SVM_model.pkl')


def preprocessing(data):
    df = data.copy()

    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

    df['service'] = df['service'].apply(lambda x: 'others' if x in rare else x)

    df.drop(columns=[c for c in const if c in df.columns], inplace=True)

    df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

    for c in columns:
        if c not in df.columns:
            df[c] = 0
    df = df[columns]

    df = scaler.transform(df)
    df = pca.transform(df)

    return df


def predict(data):

    trained = preprocessing(data)

    isolation_pred = IF_model.predict(trained)
    isolation_pred = ['Normal' if pred == 1 else 'Anomaly' for pred in isolation_pred]

    svm_pred = SVM_model.predict(trained)
    svm_pred = ['Normal' if pred == 1 else 'Anomaly' for pred in svm_pred]

    reconstruct = autoencoder.predict(trained)
    err = np.mean((trained - reconstruct) ** 2, axis = 1)
    autoencoder_pred = ['Normal' if e < threshold else 'Anomaly' for e in err]

    predictions = pd.DataFrame({
        'ISOLATION FOREST' : isolation_pred,
        'ONE CLASS SVM' : svm_pred,
        'AUTOENCODER' : autoencoder_pred
    })

    return predictions

st.title("NETWORK ANOMALY DETECTION")
st.write("Fill the network traffic details and click on predict button")

c1, c2, c3 = st.columns(3)

with c1:
    duration = st.number_input("Duration", value = 0)
    src_bytes = st.number_input("Source Bytes", value = 181)
    dst_bytes = st.number_input("Destination Bytes", value = 5450)
    count = st.number_input("Count", value = 8)
    srv_count = st.number_input("Srv Count", value = 8)
    logged_in = st.selectbox("Logged In", [0, 1])

with c2:
    protocol_type = st.selectbox("Protocol Type", ['tcp', 'udp', 'icmp'])
    service = st.selectbox("Service", ['http', 'ftp', 'smtp', 'ssh', 'dns', 'ftp_data', 'others'])
    flag = st.selectbox("Flag", ['SF', 'S0', 'REJ', 'RSTO', 'SH'])
    land = st.selectbox("Land", [0, 1])
    wrong_fragment = st.number_input("Wrong Fragment", value = 0)
    urgent = st.number_input("Urgent", value = 0)

with c3:
    serror_rate = st.slider("Serror Rate", 0.0, 1.0, 0.0)
    rerror_rate = st.slider("Rerror Rate", 0.0, 1.0, 0.0)
    same_srv_rate = st.slider("Same Srv Rate", 0.0, 1.0, 1.0)
    diff_srv_rate = st.slider("Diff Srv Rate", 0.0, 1.0, 0.0)
    dst_host_count = st.number_input("Dst Host Count", value = 9)
    hot = st.number_input("Hot", value = 0)


if st.button("Predict"):

    inp = pd.DataFrame({
        'duration': [duration],
        'protocol_type': [protocol_type],
        'service': [service],
        'flag': [flag],
        'src_bytes': [src_bytes],
        'dst_bytes': [dst_bytes],
        'land': [land],
        'wrong_fragment': [wrong_fragment],
        'urgent': [urgent],
        'hot': [hot],
        'logged_in': [logged_in],
        'count': [count],
        'srv_count': [srv_count],
        'serror_rate': [serror_rate],
        'rerror_rate': [rerror_rate],
        'same_srv_rate': [same_srv_rate],
        'diff_srv_rate': [diff_srv_rate],
        'dst_host_count': [dst_host_count]
    })
    result = predict(inp)

    st.subheader("Prediction")

    for col in result.columns:
        op = result[col][0]
        if op == 'Anomaly':
            st.error(f"{col} : Anomaly")
        else:
            st.success(f"{col} : Normal")