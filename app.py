from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CORRECTED MODEL LOADING ---
# Load all the necessary files. Note the use of separate files for models and their label encoders.
try:
    # Doctor Recommender: a pipeline and its label encoder
    doctor_pipeline = joblib.load('doctor_recommender_model.joblib')
    doctor_le = joblib.load('specialist_label_encoder.joblib')
    app.logger.info("Doctor Recommender model loaded successfully.")

    # Risk Assessor: a single model file
    risk_model = joblib.load('risk_assessor_model.joblib')
    app.logger.info("Risk Assessor model loaded successfully.")

    # Symptom Analyzer: a pipeline and its label encoder
    symptom_pipeline = joblib.load('symptom_analyzer_model.joblib')
    symptom_le = joblib.load('symptom_label_encoder.joblib')
    app.logger.info("Symptom Analyzer model loaded successfully.")

except FileNotFoundError as e:
    app.logger.error(f"Error loading model files: {e}")
    app.logger.error("Please ensure all .joblib files are in the same directory as the Flask app.")
    # Exit or handle the error appropriately if models are essential
    exit()


@app.route("/health", methods=["GET"])
def health_check():
    """A simple health check endpoint."""
    app.logger.info("GET /health")
    return jsonify({"status": "healthy", "message": "All models loaded."}), 200

@app.route('/recommend-doctor', methods=['POST'])
def recommend_doctor():
    """Recommends a doctor specialization based on symptoms."""
    app.logger.info("POST /recommend-doctor")
    data = request.json
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return jsonify({"error": "Symptom text cannot be empty."}), 400

    # The pipeline handles both text transformation and prediction.
    pred_index = doctor_pipeline.predict([symptoms])[0]
    probabilities = doctor_pipeline.predict_proba([symptoms])[0]
    
    # Convert the numerical prediction back to a string label.
    specialization_name = doctor_le.inverse_transform([pred_index])[0]
    confidence = probabilities.max()

    return jsonify({
        'specialization': specialization_name,
        'confidence': float(confidence)
    })

@app.route('/assess-risk', methods=['POST'])
def assess_risk():
    """Assesses health risk based on structured data."""
    app.logger.info("POST /assess-risk")
    data = request.json
    
    # Safely get data, providing default values.
    age = data.get('age', 30) # Default to a neutral age
    chest_pain = 1 if data.get('chest_pain', False) else 0
    shortness_of_breath = 1 if data.get('shortness_of_breath', False) else 0

    # Prepare features for the model
    features = np.array([[age, chest_pain, shortness_of_breath]])

    # --- CORRECTED LOGIC FOR MULTI-CLASS OUTPUT ---
    # Predict the risk level (e.g., 'low', 'moderate', 'high')
    risk_level = risk_model.predict(features)[0]
    
    # Get the probabilities for all classes
    probabilities = risk_model.predict_proba(features)[0]
    
    # Find the confidence score for the predicted class
    class_index = list(risk_model.classes_).index(risk_level)
    risk_score = probabilities[class_index]
    
    return jsonify({
        'risk_level': risk_level,
        'risk_score': float(risk_score)
    })

@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    """Analyzes symptoms to predict a likely condition."""
    app.logger.info("POST /analyze-symptoms")
    data = request.json
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return jsonify({"error": "Symptom text cannot be empty."}), 400

    # The pipeline handles text transformation and prediction.
    pred_index = symptom_pipeline.predict([symptoms])[0]
    probabilities = symptom_pipeline.predict_proba([symptoms])[0]
    
    # Convert the numerical prediction back to a string label.
    condition_name = symptom_le.inverse_transform([pred_index])[0]
    confidence = probabilities.max()

    return jsonify({
        'condition': condition_name,
        'confidence': float(confidence)
    })

@app.route('/health', methods=['GET'])

def health():

    app.logger.info("GET /health")

    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = 8000
    
    # Waitress is a production-ready WSGI server for Windows and Unix.
    from waitress import serve
    print(f"✅ Starting Waitress server on http://0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port)
