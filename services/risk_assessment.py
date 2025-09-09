import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional
import json

class RiskAssessment:
    def __init__(self):
        self.risk_factors = self._load_risk_factors()
        self.models = self._initialize_models()
    
    def _load_risk_factors(self) -> Dict:
        return {
            "diabetes": {
                "age_risk": {">45": 0.3, "35-45": 0.2, "<35": 0.1},
                "symptoms": {
                    "frequent urination": 0.4,
                    "excessive thirst": 0.4,
                    "unexplained weight loss": 0.3,
                    "fatigue": 0.2,
                    "blurred vision": 0.3
                },
                "lifestyle": {
                    "obesity": 0.4,
                    "sedentary": 0.3,
                    "poor_diet": 0.2
                }
            },
            "hypertension": {
                "age_risk": {">60": 0.4, "40-60": 0.3, "<40": 0.1},
                "symptoms": {
                    "headache": 0.2,
                    "dizziness": 0.3,
                    "chest pain": 0.4,
                    "shortness of breath": 0.3
                },
                "lifestyle": {
                    "high_sodium": 0.3,
                    "stress": 0.2,
                    "smoking": 0.4
                }
            },
            "heart_disease": {
                "age_risk": {">65": 0.5, "45-65": 0.3, "<45": 0.1},
                "symptoms": {
                    "chest pain": 0.5,
                    "shortness of breath": 0.4,
                    "fatigue": 0.2,
                    "irregular heartbeat": 0.4
                },
                "lifestyle": {
                    "smoking": 0.5,
                    "high_cholesterol": 0.4,
                    "diabetes": 0.3
                }
            }
        }
    
    def _initialize_models(self) -> Dict:
        # In a real implementation, these would be trained models
        # For now, we'll use rule-based assessment
        return {
            "diabetes": None,
            "hypertension": None,
            "heart_disease": None
        }
    
    def assess_chronic_disease_risk(self, age: int, gender: str, symptoms: List[str], 
                                   medical_history: List[str], 
                                   lifestyle_factors: Dict = None) -> Dict:
        if lifestyle_factors is None:
            lifestyle_factors = {}
        results = {}
        
        for disease, factors in self.risk_factors.items():
            risk_score = self._calculate_disease_risk(
                disease, age, gender, symptoms, medical_history, lifestyle_factors
            )
            results[disease] = {
                "risk_score": risk_score,
                "risk_level": self._get_risk_level(risk_score),
                "recommendations": self._get_recommendations(disease, risk_score)
            }
        
        # Overall assessment
        overall_risk = np.mean([results[disease]["risk_score"] for disease in results])
        
        return {
            "individual_risks": results,
            "overall_risk_score": overall_risk,
            "overall_risk_level": self._get_risk_level(overall_risk),
            "general_recommendations": self._get_general_recommendations(overall_risk),
            "disclaimer": "This assessment is for informational purposes only. Consult a healthcare professional for proper diagnosis."
        }
    
    def _calculate_disease_risk(self, disease: str, age: int, gender: str, 
                               symptoms: List[str], medical_history: List[str],
                               lifestyle_factors: Dict) -> float:
        factors = self.risk_factors[disease]
        risk_score = 0.0
        
        # Age factor
        age_risks = factors["age_risk"]
        if age > 65:
            risk_score += age_risks.get(">65", age_risks.get(">60", 0))
        elif age > 45:
            risk_score += age_risks.get("45-65", age_risks.get("35-45", 0))
        else:
            risk_score += age_risks.get("<45", age_risks.get("<35", 0))
        
        # Symptom factor
        symptom_risks = factors["symptoms"]
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for risk_symptom, score in symptom_risks.items():
                if risk_symptom in symptom_lower:
                    risk_score += score
        
        # Medical history factor
        for condition in medical_history:
            condition_lower = condition.lower()
            if disease in condition_lower or condition_lower in disease:
                risk_score += 0.5
        
        # Lifestyle factors
        lifestyle_risks = factors.get("lifestyle", {})
        for factor, value in lifestyle_factors.items():
            if factor in lifestyle_risks and value:
                risk_score += lifestyle_risks[factor]
        
        # Gender-specific adjustments
        if disease == "heart_disease" and gender.lower() == "male":
            risk_score += 0.1
        elif disease == "diabetes" and gender.lower() == "female":
            risk_score += 0.05
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    RISK_THRESHOLDS = {'low': 0.3, 'moderate': 0.6}
    
    def _get_risk_level(self, score: float) -> str:
        if score < self.RISK_THRESHOLDS['low']:
            return "Low"
        elif score < self.RISK_THRESHOLDS['moderate']:
            return "Moderate"
        else:
            return "High"
    
    def _get_recommendations(self, disease: str, risk_score: float) -> List[str]:
        recommendations = []
        risk_level = self._get_risk_level(risk_score)
        
        if risk_level == "High":
            recommendations.append(f"High risk for {disease} detected. Consult a healthcare provider immediately.")
            recommendations.append("Consider comprehensive health screening.")
        elif risk_level == "Moderate":
            recommendations.append(f"Moderate risk for {disease}. Schedule regular check-ups.")
            recommendations.append("Monitor symptoms and lifestyle factors.")
        else:
            recommendations.append(f"Low risk for {disease}. Maintain healthy lifestyle.")
        
        # Disease-specific recommendations
        if disease == "diabetes":
            recommendations.extend([
                "Monitor blood sugar levels",
                "Maintain healthy weight",
                "Exercise regularly",
                "Follow balanced diet"
            ])
        elif disease == "hypertension":
            recommendations.extend([
                "Monitor blood pressure regularly",
                "Reduce sodium intake",
                "Manage stress levels",
                "Maintain healthy weight"
            ])
        elif disease == "heart_disease":
            recommendations.extend([
                "Regular cardiovascular exercise",
                "Heart-healthy diet",
                "Quit smoking if applicable",
                "Manage cholesterol levels"
            ])
        
        return recommendations
    
    _GENERAL_RECOMMENDATIONS = [
        "Maintain regular exercise routine",
        "Follow balanced, nutritious diet",
        "Get adequate sleep (7-9 hours)",
        "Manage stress through relaxation techniques",
        "Avoid smoking and limit alcohol consumption",
        "Stay hydrated",
        "Regular health check-ups"
    ]
    
    def _get_general_recommendations(self, overall_risk: float) -> List[str]:
        recommendations = self._GENERAL_RECOMMENDATIONS.copy()
        
        if overall_risk > 0.5:
            recommendations.insert(0, "Schedule comprehensive health evaluation with healthcare provider")
        
        return recommendations
    
    def is_ready(self) -> bool:
        return len(self.risk_factors) > 0