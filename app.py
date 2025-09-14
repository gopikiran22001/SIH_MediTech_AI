from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models
doctor_clf, doctor_tfidf = joblib.load("doctor_recommendation.joblib")
risk_clf = joblib.load("risk_assessment.joblib")
symptom_pipeline = joblib.load("symptom_analyzer.joblib")

@app.route('/recommend-doctor', methods=['POST'])
def recommend_doctor():
    app.logger.info("POST /recommend-doctor")
    data = request.json
    symptoms = data.get('symptoms', '')
    
    # Transform symptoms and predict
    symptoms_vector = doctor_tfidf.transform([symptoms])
    prediction = doctor_clf.predict(symptoms_vector)[0]
    confidence = doctor_clf.predict_proba(symptoms_vector).max()
    
    return jsonify({
        'specialization': prediction,
        'confidence': float(confidence)
    })

@app.route('/assess-risk', methods=['POST'])
def assess_risk():
    app.logger.info("POST /assess-risk")
    data = request.json
    age = data.get('age', 0)
    chest_pain = 1 if data.get('chest_pain', False) else 0
    shortness_of_breath = 1 if data.get('shortness_of_breath', False) else 0
    
    # Predict risk
    features = np.array([[age, chest_pain, shortness_of_breath]])
    risk_score = risk_clf.predict_proba(features)[0][1]
    risk_level = "high" if risk_score > 0.5 else "low"
    
    return jsonify({
        'risk_level': risk_level,
        'risk_score': float(risk_score)
    })

@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    app.logger.info("POST /analyze-symptoms")
    data = request.json
    symptoms = data.get('symptoms', '')
    
    # Predict condition
    prediction = symptom_pipeline.predict([symptoms])[0]
    confidence = symptom_pipeline.predict_proba([symptoms]).max()
    
    return jsonify({
        'condition': prediction,
        'confidence': float(confidence)
    })

@app.route('/health', methods=['GET'])
def health():
    app.logger.info("GET /health")
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':

    port=8000
  
    # from unicorn import serve
    # print(f"Starting unicorn server on port {port}")
    # serve(app, host='0.0.0.0', port=port)
    
    from waitress import serve
    print(f"Starting waitress server on port {port}")
    serve(app, host='0.0.0.0', port=port)