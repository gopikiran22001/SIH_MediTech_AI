from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

# Lazy imports to save memory
app = FastAPI(title="Healthcare AI Service", version="1.0.0")

# Global model storage
models = {}

class SymptomRequest(BaseModel):
    symptoms: str
    age: Optional[int] = None
    gender: Optional[str] = None

class RiskResponse(BaseModel):
    risk_level: str
    confidence: float
    recommendations: List[str]

class ConditionResponse(BaseModel):
    conditions: List[dict]
    recommendations: List[str]

class DoctorRequest(BaseModel):
    condition: str
    location: Optional[str] = None

class PrescriptionRequest(BaseModel):
    medicines: List[str]
    location: Optional[str] = None

class TranslationRequest(BaseModel):
    text: str
    target_language: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    pass

@app.get("/")
async def root():
    return {"message": "Healthcare AI Service", "status": "running"}

@app.post("/risk-assessment", response_model=RiskResponse)
async def assess_risk(request: SymptomRequest):
    """Assess patient risk level based on symptoms"""
    from services.risk_assessment import RiskAssessmentService
    
    if 'risk_service' not in models:
        models['risk_service'] = RiskAssessmentService()
    
    result = models['risk_service'].assess_risk(
        request.symptoms, 
        request.age, 
        request.gender
    )
    return result

@app.post("/symptom-analysis", response_model=ConditionResponse)
async def analyze_symptoms(request: SymptomRequest):
    """Analyze symptoms and suggest possible conditions"""
    from services.symptom_checker import SymptomCheckerService
    
    if 'symptom_service' not in models:
        models['symptom_service'] = SymptomCheckerService()
    
    result = models['symptom_service'].analyze_symptoms(request.symptoms)
    return result

@app.post("/find-doctor")
async def find_doctor(request: DoctorRequest):
    """Find relevant doctors based on condition"""
    from services.doctor_mapping import DoctorMappingService
    
    if 'doctor_service' not in models:
        models['doctor_service'] = DoctorMappingService()
    
    result = await models['doctor_service'].find_doctors(
        request.condition, 
        request.location
    )
    return result

@app.post("/find-pharmacy")
async def find_pharmacy(request: PrescriptionRequest):
    """Find pharmacies with required medicines"""
    from services.pharmacy_mapping import PharmacyMappingService
    
    if 'pharmacy_service' not in models:
        models['pharmacy_service'] = PharmacyMappingService()
    
    result = models['pharmacy_service'].find_pharmacies(
        request.medicines, 
        request.location
    )
    return result

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Translate text to target language"""
    from services.translation_service import TranslationService
    
    if 'translation_service' not in models:
        models['translation_service'] = TranslationService()
    
    result = models['translation_service'].translate(
        request.text, 
        request.target_language
    )
    return result

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text"""
    from services.speech_service import SpeechService
    
    if 'speech_service' not in models:
        models['speech_service'] = SpeechService()
    
    result = await models['speech_service'].transcribe_audio(audio)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)