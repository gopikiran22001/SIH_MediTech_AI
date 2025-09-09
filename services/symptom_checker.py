from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from typing import List, Dict, Optional

class SymptomChecker:
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.classifier = None
        self.symptom_database = self._load_symptom_database()
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            # Use DistilBERT for lightweight symptom classification
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True
            )
            print("Symptom checker model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.classifier = None
    
    def _load_symptom_database(self):
        # Simplified symptom-condition mapping
        return {
            "fever": ["common cold", "flu", "infection", "covid-19"],
            "cough": ["common cold", "flu", "bronchitis", "pneumonia", "covid-19"],
            "headache": ["tension headache", "migraine", "flu", "dehydration"],
            "sore throat": ["common cold", "strep throat", "flu"],
            "fatigue": ["flu", "anemia", "depression", "sleep disorder"],
            "nausea": ["food poisoning", "gastritis", "pregnancy", "migraine"],
            "chest pain": ["heart disease", "anxiety", "muscle strain", "acid reflux"],
            "shortness of breath": ["asthma", "heart disease", "anxiety", "covid-19"],
            "dizziness": ["low blood pressure", "dehydration", "inner ear problem"],
            "abdominal pain": ["gastritis", "appendicitis", "food poisoning", "ulcer"]
        }
    
    def analyze(self, symptoms: List[str], age: Optional[int] = None, 
                gender: Optional[str] = None, medical_history: List[str] = []) -> Dict:
        if not self.classifier:
            return self._fallback_analysis(symptoms, age, gender, medical_history)
        
        try:
            # Combine symptoms into text for analysis
            symptom_text = " ".join(symptoms).lower()
            
            # Get possible conditions based on symptom database (optimized)
            possible_conditions = set()
            symptom_set = {symptom.lower() for symptom in symptoms}
            
            # Pre-compute symptom database keys for faster lookup
            db_keys = list(self.symptom_database.keys())
            
            for symptom_lower in symptom_set:
                # Use any() for early termination and better performance
                matching_keys = [key for key in db_keys if key in symptom_lower or symptom_lower in key]
                for key in matching_keys:
                    possible_conditions.update(self.symptom_database[key])
            
            # Calculate severity score
            severity_score = self._calculate_severity(symptoms, age, medical_history)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(symptoms, severity_score)
            
            return {
                "symptoms": symptoms,
                "possible_conditions": list(possible_conditions)[:5],  # Top 5
                "severity_score": severity_score,
                "severity_level": self._get_severity_level(severity_score),
                "recommendations": recommendations,
                "disclaimer": "This is not a medical diagnosis. Please consult a healthcare professional."
            }
            
        except Exception as e:
            return self._fallback_analysis(symptoms, age, gender, medical_history)
    
    def _fallback_analysis(self, symptoms: List[str], age: Optional[int], 
                          gender: Optional[str], medical_history: List[str]) -> Dict:
        # Simple rule-based fallback
        possible_conditions = set()
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for key, conditions in self.symptom_database.items():
                if key in symptom_lower:
                    possible_conditions.update(conditions)
        
        severity_score = len(symptoms) * 0.2  # Simple scoring
        
        return {
            "symptoms": symptoms,
            "possible_conditions": list(possible_conditions)[:5],
            "severity_score": min(severity_score, 1.0),
            "severity_level": self._get_severity_level(severity_score),
            "recommendations": self._generate_recommendations(symptoms, severity_score),
            "disclaimer": "This is not a medical diagnosis. Please consult a healthcare professional."
        }
    
    def _calculate_severity(self, symptoms: List[str], age: Optional[int], 
                           medical_history: List[str]) -> float:
        base_score = len(symptoms) * 0.15
        
        # High-risk symptoms
        high_risk_symptoms = ["chest pain", "shortness of breath", "severe headache", 
                             "high fever", "difficulty breathing"]
        for symptom in symptoms:
            if any(risk in symptom.lower() for risk in high_risk_symptoms):
                base_score += 0.3
        
        # Age factor
        if age and age > 65:
            base_score += 0.2
        elif age and age < 5:
            base_score += 0.15
        
        # Medical history factor
        if medical_history:
            base_score += len(medical_history) * 0.1
        
        return min(base_score, 1.0)
    
    def _get_severity_level(self, score: float) -> str:
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        else:
            return "High"
    
    def _generate_recommendations(self, symptoms: List[str], severity_score: float) -> List[str]:
        recommendations = []
        
        if severity_score > 0.7:
            recommendations.append("Seek immediate medical attention")
            recommendations.append("Consider visiting emergency room")
        elif severity_score > 0.4:
            recommendations.append("Schedule appointment with doctor within 24-48 hours")
            recommendations.append("Monitor symptoms closely")
        else:
            recommendations.append("Rest and stay hydrated")
            recommendations.append("Monitor symptoms for 24-48 hours")
        
        # Symptom-specific recommendations
        symptom_text = " ".join(symptoms).lower()
        if "fever" in symptom_text:
            recommendations.append("Take temperature regularly")
            recommendations.append("Use fever reducers if needed")
        
        if "cough" in symptom_text:
            recommendations.append("Stay hydrated")
            recommendations.append("Use humidifier if available")
        
        return recommendations
    
    def is_ready(self) -> bool:
        return self.classifier is not None