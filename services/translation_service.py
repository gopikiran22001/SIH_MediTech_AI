from deep_translator import GoogleTranslator
from gtts import gTTS
import io
import base64
from typing import Dict, Optional

# Handle speech_recognition import for Python 3.13 compatibility
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

class TranslationService:
    def __init__(self):
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
        else:
            self.recognizer = None
        self.supported_languages = {
            'english': 'en',
            'hindi': 'hi',
            'punjabi': 'pa'
        }
    
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> Dict:
        try:
            # Convert language names to codes
            target_code = self.supported_languages.get(target_lang.lower(), target_lang)
            
            # Perform translation using deep-translator
            translator = GoogleTranslator(source='auto', target=target_code)
            translated_text = translator.translate(text)
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": "auto",
                "target_language": target_code,
                "confidence": None
            }
        except ConnectionError:
            return {
                "error": "Translation service unavailable",
                "original_text": text,
                "translated_text": text
            }
        except ValueError as e:
            return {
                "error": f"Invalid input: {str(e)}",
                "original_text": text,
                "translated_text": text
            }
        except Exception:
            return {
                "error": "Translation failed",
                "original_text": text,
                "translated_text": text
            }
    
    def text_to_speech(self, text: str, language: str = "en") -> Dict:
        audio_buffer = None
        try:
            # Convert language name to code
            lang_code = self.supported_languages.get(language.lower(), language)
            
            # Generate speech
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            # Convert to base64 for API response
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
            
            return {
                "text": text,
                "language": lang_code,
                "audio_base64": audio_base64,
                "format": "mp3"
            }
        except Exception as e:
            return {
                "error": f"Text-to-speech failed: {str(e)}",
                "text": text
            }
        finally:
            if audio_buffer:
                audio_buffer.close()
    
    def speech_to_text(self, audio_data: bytes, language: str = "en") -> Dict:
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
            return {
                "error": "Speech recognition not available (Python 3.13 compatibility issue)",
                "recognized_text": ""
            }
            
        audio_buffer = None
        try:
            # Convert language name to code
            lang_code = self.supported_languages.get(language.lower(), language)
            
            # Convert audio data to AudioFile
            audio_buffer = io.BytesIO(audio_data)
            audio_file = sr.AudioFile(audio_buffer)
            
            with audio_file as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio, language=lang_code)
            
            return {
                "recognized_text": text,
                "language": lang_code,
                "confidence": None  # Google API doesn't provide confidence
            }
        except sr.UnknownValueError:
            return {
                "error": "Could not understand audio",
                "recognized_text": ""
            }
        except sr.RequestError as e:
            return {
                "error": f"Speech recognition service error: {str(e)}",
                "recognized_text": ""
            }
        except Exception as e:
            return {
                "error": f"Speech-to-text failed: {str(e)}",
                "recognized_text": ""
            }
        finally:
            if audio_buffer:
                audio_buffer.close()
    
    def get_supported_languages(self) -> Dict:
        return {
            "supported_languages": self.supported_languages,
            "default_source": "auto",
            "default_target": "en"
        }
    
    def detect_language(self, text: str) -> Dict:
        try:
            from deep_translator import single_detection
            detected_lang = single_detection(text, api_key=None)
            return {
                "text": text,
                "detected_language": detected_lang,
                "confidence": None
            }
        except Exception as e:
            return {
                "error": f"Language detection failed: {str(e)}",
                "text": text
            }
    
    def is_ready(self) -> bool:
        try:
            # Test translation service
            translator = GoogleTranslator(source='en', target='hi')
            test_result = translator.translate("test")
            return test_result is not None
        except Exception:
            return False