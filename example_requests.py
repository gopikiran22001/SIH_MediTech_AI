import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_risk_assessment():
    """Test risk assessment endpoint"""
    print("Testing Risk Assessment...")
    
    data = {
        "symptoms": "severe chest pain and difficulty breathing",
        "age": 45,
        "gender": "male"
    }
    
    response = requests.post(f"{BASE_URL}/risk-assessment", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_symptom_analysis():
    """Test symptom analysis endpoint"""
    print("Testing Symptom Analysis...")
    
    data = {
        "symptoms": "headache, fever, and nausea for 2 days"
    }
    
    response = requests.post(f"{BASE_URL}/symptom-analysis", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_doctor_mapping():
    """Test doctor mapping endpoint"""
    print("Testing Doctor Mapping...")
    
    data = {
        "condition": "heart problems",
        "location": "New York"
    }
    
    response = requests.post(f"{BASE_URL}/find-doctor", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)
    
    # Test different conditions
    test_conditions = [
        {"condition": "respiratory infection", "location": None},
        {"condition": "skin rash", "location": "California"},
        {"condition": "joint pain", "location": None}
    ]
    
    for test_case in test_conditions:
        print(f"Testing condition: {test_case['condition']}")
        response = requests.post(f"{BASE_URL}/find-doctor", json=test_case)
        result = response.json()
        print(f"Specialty: {result.get('recommended_specialty', 'N/A')}")
        print(f"Doctors found: {result.get('total_found', 0)}")
        print("-" * 30)

def test_pharmacy_mapping():
    """Test pharmacy mapping endpoint"""
    print("Testing Pharmacy Mapping...")
    
    data = {
        "medicines": ["paracetamol", "ibuprofen", "amoxicillin"],
        "location": "New York"
    }
    
    response = requests.post(f"{BASE_URL}/find-pharmacy", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_translation():
    """Test translation endpoint"""
    print("Testing Translation...")
    
    data = {
        "text": "I have a severe headache and fever",
        "target_language": "hi"
    }
    
    response = requests.post(f"{BASE_URL}/translate", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_health_check():
    """Test health check endpoint"""
    print("Testing Health Check...")
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

if __name__ == "__main__":
    print("Healthcare AI Service - Example Requests")
    print("=" * 50)
    
    try:
        test_health_check()
        test_risk_assessment()
        test_symptom_analysis()
        test_doctor_mapping()
        test_pharmacy_mapping()
        test_translation()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running with: python main.py")
    except Exception as e:
        print(f"Error: {e}")