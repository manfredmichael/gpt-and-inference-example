import pickle
import numpy as np


all_data = []

def dump_data(temperature, humidity, aqi):
    global all_data
    all_data.append([temperature, humidity, aqi])
    all_data = all_data[-200:]
    return all_data

for i in range(10_000):
    dump_data(21, 0.98, 200)

def load_model(filename):
    # Load the model from the file
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_temperature(data, scaler):
    data = np.array(data)
    data = data[:,0]
    data = scaler.transform([data])
    return data

def preprocess_humidity(data, scaler):
    data = np.array(data)
    data = data[:,1]
    data = scaler.transform([data])
    return data

temp_scaler = load_model('temp_scaler.pkl')
temp_model = load_model('temp_model.pkl')

# Temperature inference
temp_data = preprocess_temperature(all_data, temp_scaler)
predictions = temp_model.predict(temp_data)
predictions = predictions[0][0]
print(predictions)

# Humidity 