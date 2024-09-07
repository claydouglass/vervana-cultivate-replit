import os
from openai import OpenAI
import base64
import requests
import csv
from io import StringIO
from datetime import datetime
from models import db, EnvironmentalData

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def analyze_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this plant image. Focus on plant health, growth stage, and any visible issues or stress indicators."},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    },
                ],
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

def get_chatbot_response(user_input):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for plant cultivation."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content

def fetch_and_store_historical_data():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQbCXrjL4mZ7mJ6VRkz-eI0cYyz1Cvq7P5uMV-j6Kd_Yf-UUTVUnYdUY1COWyT3Qm93j655CznFm-1g/pub?output=csv"
    response = requests.get(url)
    if response.status_code == 200:
        csv_content = StringIO(response.text)
        csv_reader = csv.DictReader(csv_content)
        
        for row in csv_reader:
            timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            environmental_data = EnvironmentalData(
                timestamp=timestamp,
                temperature=float(row['temperature']),
                humidity=float(row['humidity']),
                co2_level=float(row['co2_level'])
            )
            db.session.add(environmental_data)
        
        db.session.commit()
        return True
    else:
        return False

def fetch_base_recipe():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-OvMFLTAzu2aSDIEF1ma2YADXDYN-ZjjPfMBB-PMvpbisTVVADFDnyJrLGYn-YJtYei8vQLu0-U9Z/pub?gid=149803876&single=true&output=csv"
    response = requests.get(url)
    if response.status_code == 200:
        csv_content = StringIO(response.text)
        csv_reader = csv.DictReader(csv_content)
        base_recipe = next(csv_reader)  # Assuming the first row contains the base recipe
        return {k: float(v) if v.replace('.', '').isdigit() else v for k, v in base_recipe.items()}
    else:
        return None

def compare_and_suggest_adjustments(current_data, base_recipe):
    adjustments = {}
    for key in current_data.keys():
        if key in base_recipe and isinstance(base_recipe[key], float):
            difference = base_recipe[key] - current_data[key]
            if abs(difference) > 0.1 * base_recipe[key]:  # 10% threshold for adjustment
                adjustments[key] = difference
    return adjustments

def get_adjustment_recommendations():
    current_data = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
    if not current_data:
        return "No current environmental data available."
    
    base_recipe = fetch_base_recipe()
    if not base_recipe:
        return "Unable to fetch base recipe data."
    
    current_env = {
        'temperature': current_data.temperature,
        'humidity': current_data.humidity,
        'co2_level': current_data.co2_level
    }
    
    adjustments = compare_and_suggest_adjustments(current_env, base_recipe)
    
    if not adjustments:
        return "All parameters are within acceptable ranges. No adjustments needed."
    
    recommendations = "Recommended adjustments:\n"
    for key, value in adjustments.items():
        recommendations += f"- {key.capitalize()}: {'Increase' if value > 0 else 'Decrease'} by {abs(value):.2f}\n"
    
    return recommendations
