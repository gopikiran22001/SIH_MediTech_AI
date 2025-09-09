import json
import wave
import tempfile
import os
from typing import Dict
from fastapi import UploadFile
import vosk

class SpeechService:
    def __init__(self):
        self.model = None
        self.model_path = None
        
    def _load_model(self):
        """Lazy load Vosk model"""
        if self.model is None:
            try:
                # Try to load a small Vosk model
                # In production, download vosk-model-small-en-us-0.15
                model_paths = [
                    "vosk-model-small-en-us-0.15",
                    "vosk-model-en-us-0.22-lgraph",
                    "model"  # fallback
                ]
                
                for path in model_paths:
                    if os.path.exists(path):
                        self.model = vosk.Model(path)
                        break
                
                if self.model is None:
                    # Create a dummy response if no model found
                    return False
                    
            except Exception as e:
                print(f"Failed to load Vosk model: {e}")
                return False
        return True
    
    async def transcribe_audio(self, audio_file: UploadFile) -> Dict:
        """Transcribe audio file to text"""
        try:
            # Validate file type
            if not audio_file.content_type.startswith('audio/'):
                return {
                    'error': 'Invalid file type. Please upload an audio file.',
                    'supported_formats': ['wav', 'mp3', 'ogg', 'flac']
                }
            
            # Load model
            if not self._load_model():
                return {
                    'error': 'Speech recognition model not available',
                    'fallback_message': 'Please type your symptoms instead'
                }
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Process audio file
                transcription = self._process_audio_file(temp_file_path)
                
                return {
                    'filename': audio_file.filename,
                    'transcription': transcription,
                    'confidence': 0.8,  # Vosk doesn't provide detailed confidence
                    'language': 'en'
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            return {
                'error': f'Transcription failed: {str(e)}',
                'fallback_message': 'Please type your symptoms instead'
            }
    
    def _process_audio_file(self, file_path: str) -> str:
        """Process audio file and return transcription"""
        try:
            # Open audio file
            wf = wave.open(file_path, 'rb')
            
            # Check audio format
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != 'NONE':
                return "Audio format not supported. Please use mono 16-bit WAV format."
            
            # Create recognizer
            rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            transcription_parts = []
            
            # Process audio in chunks
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                    
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get('text'):
                        transcription_parts.append(result['text'])
            
            # Get final result
            final_result = json.loads(rec.FinalResult())
            if final_result.get('text'):
                transcription_parts.append(final_result['text'])
            
            wf.close()
            
            # Combine all parts
            full_transcription = ' '.join(transcription_parts).strip()
            
            if not full_transcription:
                return "No speech detected in audio file."
            
            return full_transcription
            
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    
    def get_supported_formats(self) -> Dict:
        """Get supported audio formats"""
        return {
            'supported_formats': ['wav', 'mp3', 'ogg', 'flac'],
            'recommended_format': 'wav',
            'sample_rate': '16000 Hz',
            'channels': 'mono',
            'bit_depth': '16-bit'
        }
    
    def convert_text_to_speech_instructions(self, text: str) -> Dict:
        """Provide instructions for text-to-speech (placeholder)"""
        return {
            'text': text,
            'message': 'Text-to-speech conversion not implemented in this lightweight version',
            'alternative': 'Use device built-in text-to-speech functionality',
            'instructions': [
                'Copy the text below',
                'Use your device\'s accessibility features',
                'Enable text-to-speech in system settings'
            ]
        }