import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Configure database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'health_records.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # To suppress a warning
db = SQLAlchemy(app)
ma = Marshmallow(app)

# Define IoT device data model
class IoTDeviceData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), unique=True)
    data = db.Column(db.String(1000))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __init__(self, device_id, data):
        self.device_id = device_id
        self.data = data

# Define health record data model
class HealthRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(100), unique=True)
    health_data = db.Column(db.String(1000))

    def __init__(self, patient_id, health_data):
        self.patient_id = patient_id
        self.health_data = health_data

# Define machine learning model
class MachineLearningModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# Define API endpoints
@app.route('/api/iot_data', methods=['POST'])
def receive_iot_data():
    data = request.get_json()
    device_id = data['device_id']
    iot_data = data['data']
    iot_device_data = IoTDeviceData(device_id=device_id, data=json.dumps(iot_data))
    db.session.add(iot_device_data)
    db.session.commit()
    return jsonify({'message': 'IoT data received successfully'})

@app.route('/api/health_records', methods=['GET'])
def get_health_records():
    health_records = HealthRecord.query.all()
    output = []
    for health_record in health_records:
        health_data = json.loads(health_record.health_data)
        output.append({'patient_id': health_record.patient_id, 'health_data': health_data})
    return jsonify({'health_records': output})

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    health_data = data['health_data']
    X = pd.DataFrame([health_data])
    prediction = ml_model.predict(X)
    return jsonify({'prediction': int(prediction[0])})

# Initialize and create the database
@app.before_first_request
def create_tables():
    db.create_all()

# Train machine learning model
def train_ml_model():
    iot_data = IoTDeviceData.query.all()
    X = []
    y = []
    for iot_device_data in iot_data:
        data = json.loads(iot_device_data.data)
        X.append(data)
        y.append(iot_device_data.device_id)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    ml_model = MachineLearningModel()
    ml_model.train(X, y)
    return ml_model

# Initialize machine learning model
ml_model = train_ml_model()

if __name__ == '__main__':
    app.run(debug=True)
