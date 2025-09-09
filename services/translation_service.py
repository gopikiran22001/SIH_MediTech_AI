from deep_translator import GoogleTranslator
from typing import Dict, Optional

class TranslationService:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'ur': 'Urdu'
        }
        
        # Medical term translations for common healthcare phrases
        self.medical_phrases = {
            'en': {
                'symptoms': 'symptoms',
                'pain': 'pain',
                'fever': 'fever',
                'headache': 'headache',
                'doctor': 'doctor',
                'medicine': 'medicine',
                'hospital': 'hospital',
                'emergency': 'emergency'
            },
            'hi': {
                'symptoms': 'लक्षण',
                'pain': 'दर्द',
                'fever': 'बुखार',
                'headache': 'सिरदर्द',
                'doctor': 'डॉक्टर',
                'medicine': 'दवा',
                'hospital': 'अस्पताल',
                'emergency': 'आपातकाल'
            },
            'es': {
                'symptoms': 'síntomas',
                'pain': 'dolor',
                'fever': 'fiebre',
                'headache': 'dolor de cabeza',
                'doctor': 'médico',
                'medicine': 'medicina',
                'hospital': 'hospital',
                'emergency': 'emergencia'
            }
        }
    
    def translate(self, text: str, target_language: str, source_language: str = 'auto') -> Dict:
        """Translate text to target language"""
        try:
            # Validate target language
            if target_language not in self.supported_languages:
                return {
                    'error': f'Unsupported target language: {target_language}',
                    'supported_languages': self.supported_languages
                }
            
            # Check if translation is needed
            if source_language == target_language:
                return {
                    'original_text': text,
                    'translated_text': text,
                    'source_language': source_language,
                    'target_language': target_language,
                    'confidence': 1.0
                }
            
            # Perform translation
            translator = GoogleTranslator(source=source_language, target=target_language)
            translated_text = translator.translate(text)
            
            # Detect source language if auto
            detected_language = source_language
            if source_language == 'auto':
                try:
                    detected_language = GoogleTranslator(source='auto', target='en').detect(text)
                except:
                    detected_language = 'unknown'
            
            return {
                'original_text': text,
                'translated_text': translated_text,
                'source_language': detected_language,
                'target_language': target_language,
                'confidence': 0.9  # GoogleTranslator doesn't provide confidence scores
            }
            
        except Exception as e:
            return {
                'error': f'Translation failed: {str(e)}',
                'original_text': text,
                'fallback_translation': self._get_fallback_translation(text, target_language)
            }
    
    def _get_fallback_translation(self, text: str, target_language: str) -> Optional[str]:
        """Provide fallback translation for common medical terms"""
        text_lower = text.lower().strip()
        
        if target_language in self.medical_phrases:
            target_phrases = self.medical_phrases[target_language]
            for en_term, translated_term in target_phrases.items():
                if en_term in text_lower:
                    return translated_term
        
        return None
    
    def get_supported_languages(self) -> Dict:
        """Get list of supported languages"""
        return self.supported_languages
    
    def translate_medical_terms(self, terms: list, target_language: str) -> Dict:
        """Translate multiple medical terms"""
        translations = {}
        
        for term in terms:
            result = self.translate(term, target_language)
            if 'error' not in result:
                translations[term] = result['translated_text']
            else:
                translations[term] = term  # Keep original if translation fails
        
        return {
            'translations': translations,
            'target_language': target_language,
            'total_terms': len(terms)
        }
    
    def detect_language(self, text: str) -> Dict:
        """Detect the language of input text"""
        try:
            detected = GoogleTranslator(source='auto', target='en').detect(text)
            language_name = self.supported_languages.get(detected, 'Unknown')
            
            return {
                'text': text,
                'detected_language': detected,
                'language_name': language_name,
                'confidence': 0.8
            }
        except Exception as e:
            return {
                'error': f'Language detection failed: {str(e)}',
                'text': text
            }