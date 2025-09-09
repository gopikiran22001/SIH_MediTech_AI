import re
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SymptomCheckerService:
    def __init__(self):
        self.symptom_condition_map = {
            # Respiratory
            'cough fever shortness breath': {'condition': 'Respiratory Infection', 'specialty': 'Pulmonology', 'severity': 'medium'},
            'chest pain breathing difficulty': {'condition': 'Respiratory Distress', 'specialty': 'Emergency Medicine', 'severity': 'high'},
            'runny nose sneezing': {'condition': 'Common Cold', 'specialty': 'General Medicine', 'severity': 'low'},
            
            # Cardiovascular
            'chest pain heart palpitations': {'condition': 'Cardiac Issue', 'specialty': 'Cardiology', 'severity': 'high'},
            'high blood pressure headache': {'condition': 'Hypertension', 'specialty': 'Cardiology', 'severity': 'medium'},
            
            # Gastrointestinal
            'nausea vomiting diarrhea': {'condition': 'Gastroenteritis', 'specialty': 'Gastroenterology', 'severity': 'medium'},
            'stomach pain bloating': {'condition': 'Digestive Issue', 'specialty': 'Gastroenterology', 'severity': 'low'},
            
            # Neurological
            'headache dizziness confusion': {'condition': 'Neurological Symptoms', 'specialty': 'Neurology', 'severity': 'medium'},
            'severe headache vision problems': {'condition': 'Migraine/Neurological', 'specialty': 'Neurology', 'severity': 'high'},
            
            # Musculoskeletal
            'joint pain swelling stiffness': {'condition': 'Arthritis/Joint Issue', 'specialty': 'Rheumatology', 'severity': 'medium'},
            'back pain muscle ache': {'condition': 'Musculoskeletal Pain', 'specialty': 'Orthopedics', 'severity': 'low'},
            
            # Dermatological
            'rash itching skin irritation': {'condition': 'Skin Condition', 'specialty': 'Dermatology', 'severity': 'low'},
            'severe rash swelling face': {'condition': 'Allergic Reaction', 'specialty': 'Emergency Medicine', 'severity': 'high'},
            
            # General
            'fever fatigue body ache': {'condition': 'Viral Infection', 'specialty': 'General Medicine', 'severity': 'medium'},
            'weight loss fatigue': {'condition': 'General Health Concern', 'specialty': 'Internal Medicine', 'severity': 'medium'}
        }
        
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.condition_vectors = None
        self._prepare_vectors()
    
    def _prepare_vectors(self):
        """Prepare TF-IDF vectors for symptom matching"""
        symptom_texts = list(self.symptom_condition_map.keys())
        self.condition_vectors = self.vectorizer.fit_transform(symptom_texts)
    
    def analyze_symptoms(self, symptoms: str) -> Dict:
        """Analyze symptoms and return possible conditions"""
        # Clean and preprocess symptoms
        symptoms_clean = re.sub(r'[^\w\s]', '', symptoms.lower())
        
        # Vectorize input symptoms
        symptom_vector = self.vectorizer.transform([symptoms_clean])
        
        # Calculate similarity with known conditions
        similarities = cosine_similarity(symptom_vector, self.condition_vectors).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:3]
        
        conditions = []
        symptom_keys = list(self.symptom_condition_map.keys())
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                condition_info = self.symptom_condition_map[symptom_keys[idx]]
                conditions.append({
                    'condition': condition_info['condition'],
                    'specialty': condition_info['specialty'],
                    'severity': condition_info['severity'],
                    'confidence': round(similarities[idx], 2)
                })
        
        # If no good matches, provide general advice
        if not conditions:
            conditions.append({
                'condition': 'General Health Concern',
                'specialty': 'General Medicine',
                'severity': 'low',
                'confidence': 0.5
            })
        
        recommendations = self._get_recommendations(conditions[0] if conditions else None)
        
        return {
            'conditions': conditions,
            'recommendations': recommendations
        }
    
    def _get_recommendations(self, primary_condition: Dict) -> List[str]:
        """Get recommendations based on primary condition"""
        if not primary_condition:
            return ["Consult a general practitioner for proper diagnosis"]
        
        severity = primary_condition.get('severity', 'low')
        specialty = primary_condition.get('specialty', 'General Medicine')
        
        if severity == 'high':
            return [
                "Seek immediate medical attention",
                f"Consider visiting emergency department",
                "Do not delay treatment"
            ]
        elif severity == 'medium':
            return [
                f"Consult a {specialty} specialist",
                "Schedule appointment within 1-2 days",
                "Monitor symptoms closely"
            ]
        else:
            return [
                f"Consider consulting {specialty}",
                "Monitor symptoms for changes",
                "Maintain good health practices"
            ]