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
