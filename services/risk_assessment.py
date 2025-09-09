import re
from typing import Dict, List, Optional
from transformers import pipeline
import torch

class RiskAssessmentService:
    def __init__(self):
        self.classifier = None
        self.high_risk_keywords = [
            'chest pain', 'difficulty breathing', 'severe headache', 'unconscious',
            'bleeding', 'stroke', 'heart attack', 'seizure', 'severe pain',
            'can\'t breathe', 'choking', 'overdose', 'suicide'
        ]
        self.medium_risk_keywords = [
            'fever', 'vomiting', 'diarrhea', 'infection', 'injury', 'fracture',
            'allergic reaction', 'rash', 'swelling', 'persistent cough'
        ]
    
    def _load_model(self):
        """Lazy load the classification model"""
        if self.classifier is None:
            try:
                # Use tiny BERT for memory efficiency
                self.classifier = pipeline(
                    "text-classification",
                    model="prajjwal1/bert-tiny",
                    device=-1,  # CPU only
                    torch_dtype=torch.float32
                )
            except:
                # Fallback to rule-based if model fails
                self.classifier = "rule_based"
    
    def assess_risk(self, symptoms: str, age: Optional[int] = None, gender: Optional[str] = None) -> Dict:
        """Assess risk level based on symptoms"""
        symptoms_lower = symptoms.lower()
        
        # Rule-based assessment for critical symptoms
        if any(keyword in symptoms_lower for keyword in self.high_risk_keywords):
            return {
                "risk_level": "high",
                "confidence": 0.9,
                "recommendations": [
                    "Seek immediate medical attention",
                    "Call emergency services if symptoms worsen",
                    "Do not delay treatment"
                ]
            }
        
        if any(keyword in symptoms_lower for keyword in self.medium_risk_keywords):
            risk_level = "medium"
            confidence = 0.7
        else:
            risk_level = "low"
            confidence = 0.6
        
        # Age-based risk adjustment
        if age and age > 65:
            if risk_level == "low":
                risk_level = "medium"
            confidence += 0.1
        
        # Try ML classification as secondary assessment
        try:
            self._load_model()
            if self.classifier != "rule_based":
                ml_result = self.classifier(symptoms)
                # Combine rule-based and ML results
                if ml_result[0]['label'] == 'NEGATIVE' and ml_result[0]['score'] > 0.8:
                    confidence = min(confidence + 0.1, 0.95)
        except:
            pass  # Continue with rule-based assessment
        
        recommendations = self._get_recommendations(risk_level)
        
        return {
            "risk_level": risk_level,
            "confidence": round(confidence, 2),
            "recommendations": recommendations
        }
    
    def _get_recommendations(self, risk_level: str) -> List[str]:
        """Get recommendations based on risk level"""
        if risk_level == "high":
            return [
                "Seek immediate medical attention",
                "Call emergency services if symptoms worsen",
                "Do not delay treatment"
            ]
        elif risk_level == "medium":
            return [
                "Consult a doctor within 24 hours",
                "Monitor symptoms closely",
                "Seek immediate help if symptoms worsen"
            ]
        else:
            return [
                "Monitor symptoms",
                "Consider consulting a doctor if symptoms persist",
                "Maintain good hygiene and rest"
            ]