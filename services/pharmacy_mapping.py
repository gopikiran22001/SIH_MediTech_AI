from typing import Dict, List, Optional
import random

class PharmacyMappingService:
    def __init__(self):
        # Sample pharmacy database with medicine inventory
        self.pharmacies = [
            {
                "id": 1,
                "name": "HealthPlus Pharmacy",
                "location": "New York, NY",
                "address": "123 Main St, New York, NY 10001",
                "phone": "+1-555-0101",
                "rating": 4.5,
                "distance": 0.5,
                "medicines": [
                    "paracetamol", "ibuprofen", "amoxicillin", "aspirin", 
                    "metformin", "lisinopril", "atorvastatin", "omeprazole"
                ]
            },
            {
                "id": 2,
                "name": "MediCare Drugstore",
                "location": "New York, NY",
                "address": "456 Broadway, New York, NY 10002",
                "phone": "+1-555-0102",
                "rating": 4.3,
                "distance": 1.2,
                "medicines": [
                    "paracetamol", "aspirin", "cough syrup", "insulin",
                    "metformin", "amlodipine", "simvastatin"
                ]
            },
            {
                "id": 3,
                "name": "QuickMeds Pharmacy",
                "location": "California, CA",
                "address": "789 Oak Ave, Los Angeles, CA 90001",
                "phone": "+1-555-0103",
                "rating": 4.7,
                "distance": 0.8,
                "medicines": [
                    "ibuprofen", "amoxicillin", "lisinopril", "atorvastatin",
                    "omeprazole", "levothyroxine", "gabapentin"
                ]
            },
            {
                "id": 4,
                "name": "Family Pharmacy",
                "location": "Texas, TX",
                "address": "321 Pine St, Houston, TX 77001",
                "phone": "+1-555-0104",
                "rating": 4.2,
                "distance": 2.1,
                "medicines": [
                    "paracetamol", "ibuprofen", "aspirin", "metformin",
                    "lisinopril", "amlodipine", "prednisone"
                ]
            },
            {
                "id": 5,
                "name": "Express Pharmacy",
                "location": "Florida, FL",
                "address": "654 Palm Blvd, Miami, FL 33101",
                "phone": "+1-555-0105",
                "rating": 4.6,
                "distance": 1.5,
                "medicines": [
                    "amoxicillin", "atorvastatin", "omeprazole", "insulin",
                    "levothyroxine", "gabapentin", "sertraline"
                ]
            }
        ]
        
        # Common medicine name mappings (generic to brand names)
        self.medicine_aliases = {
            "acetaminophen": "paracetamol",
            "tylenol": "paracetamol",
            "advil": "ibuprofen",
            "motrin": "ibuprofen",
            "bayer": "aspirin",
            "glucophage": "metformin",
            "prinivil": "lisinopril",
            "lipitor": "atorvastatin",
            "prilosec": "omeprazole"
        }
    
    def find_pharmacies(self, medicines: List[str], location: Optional[str] = None) -> Dict:
        """Find pharmacies with required medicines"""
        # Normalize medicine names
        normalized_medicines = [self._normalize_medicine_name(med.lower().strip()) for med in medicines]
        
        # Filter pharmacies by location if provided
        relevant_pharmacies = self.pharmacies
        if location:
            relevant_pharmacies = [
                pharmacy for pharmacy in self.pharmacies
                if location.lower() in pharmacy['location'].lower()
            ]
        
        # Calculate availability score for each pharmacy
        pharmacy_scores = []
        for pharmacy in relevant_pharmacies:
            available_medicines = []
            unavailable_medicines = []
            
            for medicine in normalized_medicines:
                if medicine in pharmacy['medicines']:
                    available_medicines.append(medicine)
                else:
                    unavailable_medicines.append(medicine)
            
            availability_score = len(available_medicines) / len(normalized_medicines) if normalized_medicines else 0
            
            pharmacy_scores.append({
                **pharmacy,
                'available_medicines': available_medicines,
                'unavailable_medicines': unavailable_medicines,
                'availability_score': round(availability_score, 2),
                'total_requested': len(normalized_medicines),
                'total_available': len(available_medicines)
            })
        
        # Sort by availability score, then by rating, then by distance
        pharmacy_scores.sort(
            key=lambda x: (x['availability_score'], x['rating'], -x['distance']), 
            reverse=True
        )
        
        # Filter pharmacies with at least one medicine available
        available_pharmacies = [p for p in pharmacy_scores if p['availability_score'] > 0]
        
        return {
            'requested_medicines': medicines,
            'total_pharmacies_found': len(available_pharmacies),
            'pharmacies': available_pharmacies[:10],  # Limit to top 10
            'message': f"Found {len(available_pharmacies)} pharmacies with requested medicines"
        }
    
    def _normalize_medicine_name(self, medicine: str) -> str:
        """Normalize medicine name using aliases"""
        medicine_clean = medicine.lower().strip()
        return self.medicine_aliases.get(medicine_clean, medicine_clean)
    
    def get_pharmacy_by_id(self, pharmacy_id: int) -> Optional[Dict]:
        """Get specific pharmacy by ID"""
        for pharmacy in self.pharmacies:
            if pharmacy['id'] == pharmacy_id:
                return pharmacy
        return None
    
    def check_medicine_availability(self, medicine: str, location: Optional[str] = None) -> Dict:
        """Check availability of a specific medicine"""
        normalized_medicine = self._normalize_medicine_name(medicine.lower().strip())
        
        relevant_pharmacies = self.pharmacies
        if location:
            relevant_pharmacies = [
                pharmacy for pharmacy in self.pharmacies
                if location.lower() in pharmacy['location'].lower()
            ]
        
        available_at = []
        for pharmacy in relevant_pharmacies:
            if normalized_medicine in pharmacy['medicines']:
                available_at.append({
                    'pharmacy_name': pharmacy['name'],
                    'location': pharmacy['location'],
                    'phone': pharmacy['phone'],
                    'distance': pharmacy['distance'],
                    'rating': pharmacy['rating']
                })
        
        return {
            'medicine': medicine,
            'normalized_name': normalized_medicine,
            'available_at': available_at,
            'total_locations': len(available_at),
            'message': f"Medicine '{medicine}' available at {len(available_at)} locations"
        }