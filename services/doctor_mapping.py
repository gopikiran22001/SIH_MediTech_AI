from typing import Dict, List, Optional
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class DoctorMappingService:
    def __init__(self):
        self.backend_url = os.getenv('BACKEND_URL', 'http://localhost:3000')
        self.api_key = os.getenv('API_KEY')
        self.doctors_cache = None
        
        # Specialty mapping for conditions
        self.condition_specialty_map = {
            'cardiac': 'Cardiology',
            'heart': 'Cardiology',
            'chest pain': 'Cardiology',
            'respiratory': 'Pulmonology',
            'breathing': 'Pulmonology',
            'cough': 'Pulmonology',
            'neurological': 'Neurology',
            'headache': 'Neurology',
            'migraine': 'Neurology',
            'skin': 'Dermatology',
            'rash': 'Dermatology',
            'joint': 'Orthopedics',
            'bone': 'Orthopedics',
            'fracture': 'Orthopedics',
            'stomach': 'Gastroenterology',
            'digestive': 'Gastroenterology',
            'nausea': 'Gastroenterology',
            'emergency': 'Emergency Medicine',
            'urgent': 'Emergency Medicine'
        }
    
    async def find_doctors(self, condition: str, location: Optional[str] = None) -> Dict:
        """Find relevant doctors based on condition and location"""
        try:
            # Determine required specialty
            specialty = self._map_condition_to_specialty(condition.lower())
            
            # Fetch doctors from backend API
            doctors = await self._fetch_doctors_from_api(specialty)
            
            if not doctors:
                return {
                    'error': 'No doctors found',
                    'recommended_specialty': specialty,
                    'doctors': [],
                    'message': 'Unable to fetch doctors from server'
                }
            
            # Filter by location if provided
            if location:
                location_filtered = [
                    doc for doc in doctors 
                    if location.lower() in (doc.get('address', '') or '').lower()
                ]
                if location_filtered:
                    doctors = location_filtered
            
            # Sort by experience
            doctors.sort(key=lambda x: x.get('experience', 0), reverse=True)
            
            # Limit to top 5 doctors
            top_doctors = doctors[:5]
            
            return {
                'recommended_specialty': specialty,
                'total_found': len(doctors),
                'doctors': top_doctors,
                'message': f"Found {len(top_doctors)} {specialty} doctors"
            }
            
        except Exception as e:
            return {
                'error': f'Failed to fetch doctors: {str(e)}',
                'recommended_specialty': specialty if 'specialty' in locals() else 'General Medicine',
                'doctors': [],
                'message': 'Service temporarily unavailable'
            }
    
    def _map_condition_to_specialty(self, condition: str) -> str:
        """Map condition keywords to medical specialty"""
        for keyword, specialty in self.condition_specialty_map.items():
            if keyword in condition:
                return specialty
        
        # Default to General Medicine if no specific specialty found
        return 'General Medicine'
    
    async def _fetch_doctors_from_api(self, specialty: Optional[str] = None) -> List[Dict]:
        """Fetch doctors from backend API"""
        try:
            if not self.api_key:
                print("API key not configured")
                return []
                
            url = f"{self.backend_url}/api/users/doctors/public"
            headers = {'x-api-key': self.api_key}
            params = {}
            if specialty:
                params['specialization'] = specialty
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('doctors', [])
            else:
                print(f"API returned status {response.status_code}: {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch doctors from API: {e}")
            return []
    
    async def get_all_specialties(self) -> List[str]:
        """Get list of all available specialties"""
        try:
            doctors = await self._fetch_doctors_from_api()
            specialties = list(set(doc.get('specialization', '') for doc in doctors if doc.get('specialization')))
            return sorted(specialties)
        except:
            return ['General Medicine', 'Cardiology', 'Neurology', 'Dermatology']