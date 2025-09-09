from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from services.symptom_checker import SymptomChecker
from services.translation_service import TranslationService
from services.risk_assessment import RiskAssessment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MediTech AI Service", version="1.0.0")

# Initialize services with error handling
try:
    symptom_checker = SymptomChecker()
    logger.info("Symptom checker initialized successfully")
except Exception as e:
    logger.warning(f"Symptom checker initialization failed: {e}")
    symptom_checker = None

try:
    translation_service = TranslationService()
    logger.info("Translation service initialized successfully")
except Exception as e:
    logger.warning(f"Translation service initialization failed: {e}")
    translation_service = None

try:
    risk_assessment = RiskAssessment()
    logger.info("Risk assessment service initialized successfully")
except Exception as e:
    logger.warning(f"Risk assessment initialization failed: {e}")
    risk_assessment = None

class SymptomRequest(BaseModel):
    symptoms: List[str]
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[List[str]] = []

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"

class RiskAssessmentRequest(BaseModel):
    age: int
    gender: str
    symptoms: List[str]
    medical_history: List[str]
    lifestyle_factors: Optional[dict] = {}

@app.get("/")
async def root():
    return {"message": "MediTech AI Service is running"}

@app.post("/analyze-symptoms")
async def analyze_symptoms(request: SymptomRequest):
    if not symptom_checker:
        raise HTTPException(status_code=503, detail="Symptom checker service unavailable")
    try:
        result = symptom_checker.analyze(
            symptoms=request.symptoms,
            age=request.age,
            gender=request.gender,
            medical_history=request.medical_history
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    if not translation_service:
        raise HTTPException(status_code=503, detail="Translation service unavailable")
    try:
        result = translation_service.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception:
        raise HTTPException(status_code=500, detail="Translation failed")

@app.post("/assess-risk")
async def assess_chronic_risk(request: RiskAssessmentRequest):
    if not risk_assessment:
        raise HTTPException(status_code=503, detail="Risk assessment service unavailable")
    try:
        result = risk_assessment.assess_chronic_disease_risk(
            age=request.age,
            gender=request.gender,
            symptoms=request.symptoms,
            medical_history=request.medical_history,
            lifestyle_factors=request.lifestyle_factors
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception:
        raise HTTPException(status_code=500, detail="Risk assessment failed")

@app.get("/health")
async def health_check():
    services = {}
    
    try:
        services["symptom_checker"] = symptom_checker.is_ready() if symptom_checker else False
    except Exception:
        services["symptom_checker"] = False
        
    try:
        services["translation"] = translation_service.is_ready() if translation_service else False
    except Exception:
        services["translation"] = False
        
    try:
        services["risk_assessment"] = risk_assessment.is_ready() if risk_assessment else False
    except Exception:
        services["risk_assessment"] = False
    
    return {
        "status": "healthy" if any(services.values()) else "degraded",
        "services": services
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)