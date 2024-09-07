import os
from openai import OpenAI
import base64
import requests
import csv
from io import StringIO
from datetime import datetime, timedelta
from models import db, EnvironmentalData, BatchData, NutrientData
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import joblib

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def analyze_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this plant image in detail. Focus on the following aspects:\n1. Plant health (excellent, good, fair, poor)\n2. Growth stage (seedling, vegetative, flowering, fruiting)\n3. Leaf color and condition\n4. Stem strength and structure\n5. Any visible pests or diseases\n6. Signs of nutrient deficiencies or excesses\n7. Overall plant vigor\n8. Estimated yield potential (low, medium, high)\n9. Recommendations for improvement\nProvide a detailed analysis for each aspect, including specific observations and scientific explanations where applicable."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

def get_chatbot_response(user_input):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert assistant for plant cultivation, specializing in advanced plant health analysis and yield prediction. Provide detailed, scientific answers to cultivation questions, incorporating the latest research and best practices in agriculture and horticulture."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=800
    )
    return response.choices[0].message.content

# Keep all other existing functions...

def predict_yield(batch_id):
    # Existing code...
    
    # Add more advanced features
    def calculate_growing_degree_days(temp_data):
        base_temp = 10  # Adjust based on the specific crop
        return sum(max(0, (temp - base_temp)) for temp in temp_data)

    def calculate_vapor_pressure_deficit(temp_data, humidity_data):
        return [calculate_vpd(temp, humidity) for temp, humidity in zip(temp_data, humidity_data)]

    # Fetch all batch data
    batches = BatchData.query.all()
    
    # Prepare data for training
    X = []
    y = []
    for batch in batches:
        # Extract features from terpene profile
        terpenes = batch.terpene_profile.split(',')
        terpene_values = [float(t.split(':')[1]) for t in terpenes]
        
        # Get environmental data for the batch
        env_data = EnvironmentalData.query.filter(
            EnvironmentalData.timestamp >= batch.harvest_date - timedelta(days=90),
            EnvironmentalData.timestamp <= batch.harvest_date
        ).all()
        
        # Get nutrient data for the batch
        nutrient_data = NutrientData.query.filter(
            NutrientData.timestamp >= batch.harvest_date - timedelta(days=90),
            NutrientData.timestamp <= batch.harvest_date
        ).all()
        
        if env_data and nutrient_data:
            temp_data = [d.temperature for d in env_data]
            humidity_data = [d.humidity for d in env_data]
            co2_data = [d.co2_level for d in env_data]
            light_data = [d.light_intensity for d in env_data]
            
            avg_temp = np.mean(temp_data)
            avg_humidity = np.mean(humidity_data)
            avg_co2 = np.mean(co2_data)
            avg_light_intensity = np.mean(light_data)
            
            avg_n = np.mean([d.nitrogen_level for d in nutrient_data])
            avg_p = np.mean([d.phosphorus_level for d in nutrient_data])
            avg_k = np.mean([d.potassium_level for d in nutrient_data])
            
            # Calculate additional features
            growing_degree_days = calculate_growing_degree_days(temp_data)
            vpd = np.mean(calculate_vapor_pressure_deficit(temp_data, humidity_data))
            dli = calculate_dli(avg_light_intensity)
            
            # Combine all features
            features = [batch.thc_level] + terpene_values + [
                avg_temp, avg_humidity, avg_co2, avg_light_intensity,
                avg_n, avg_p, avg_k, growing_degree_days, vpd, dli,
                np.std(temp_data), np.std(humidity_data), np.std(co2_data),
                max(temp_data) - min(temp_data),  # Temperature range
                max(humidity_data) - min(humidity_data),  # Humidity range
                max(co2_data) - min(co2_data)  # CO2 range
            ]
            X.append(features)
            y.append(batch.yield_amount)
    
    if not X or not y:
        return "Insufficient data for yield prediction"
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save the model and scaler
    joblib.dump(model, 'yield_prediction_model.joblib')
    joblib.dump(scaler, 'yield_prediction_scaler.joblib')
    
    # Predict yield for the given batch_id
    target_batch = BatchData.query.filter_by(batch_id=batch_id).first()
    if not target_batch:
        return "Batch not found"
    
    # Get environmental and nutrient data for the target batch
    target_env_data = EnvironmentalData.query.filter(
        EnvironmentalData.timestamp >= target_batch.harvest_date - timedelta(days=90),
        EnvironmentalData.timestamp <= target_batch.harvest_date
    ).all()
    
    target_nutrient_data = NutrientData.query.filter(
        NutrientData.timestamp >= target_batch.harvest_date - timedelta(days=90),
        NutrientData.timestamp <= target_batch.harvest_date
    ).all()
    
    if not target_env_data or not target_nutrient_data:
        return "Insufficient environmental or nutrient data for yield prediction"
    
    temp_data = [d.temperature for d in target_env_data]
    humidity_data = [d.humidity for d in target_env_data]
    co2_data = [d.co2_level for d in target_env_data]
    light_data = [d.light_intensity for d in target_env_data]
    
    avg_temp = np.mean(temp_data)
    avg_humidity = np.mean(humidity_data)
    avg_co2 = np.mean(co2_data)
    avg_light_intensity = np.mean(light_data)
    
    avg_n = np.mean([d.nitrogen_level for d in target_nutrient_data])
    avg_p = np.mean([d.phosphorus_level for d in target_nutrient_data])
    avg_k = np.mean([d.potassium_level for d in target_nutrient_data])
    
    growing_degree_days = calculate_growing_degree_days(temp_data)
    vpd = np.mean(calculate_vapor_pressure_deficit(temp_data, humidity_data))
    dli = calculate_dli(avg_light_intensity)
    
    target_features = [target_batch.thc_level] + \
                      [float(t.split(':')[1]) for t in target_batch.terpene_profile.split(',')] + \
                      [avg_temp, avg_humidity, avg_co2, avg_light_intensity,
                       avg_n, avg_p, avg_k, growing_degree_days, vpd, dli,
                       np.std(temp_data), np.std(humidity_data), np.std(co2_data),
                       max(temp_data) - min(temp_data),
                       max(humidity_data) - min(humidity_data),
                       max(co2_data) - min(co2_data)]
    
    # Scale the target features
    target_features_scaled = scaler.transform([target_features])
    
    predicted_yield = model.predict(target_features_scaled)[0]
    
    feature_importance = analyze_feature_importance(model, X, y)
    
    return {
        "predicted_yield": f"{predicted_yield:.2f} units",
        "model_performance": {
            "mse": mse,
            "r2": r2
        },
        "feature_importance": feature_importance
    }

def analyze_plant_health_trends():
    # Existing code...
    
    # Add more advanced analysis
    def calculate_stress_index(temp_trend, humidity_trend, co2_trend):
        temp_stress = sum(1 for t in temp_trend if t < 15 or t > 30)
        humidity_stress = sum(1 for h in humidity_trend if h < 30 or h > 70)
        co2_stress = sum(1 for c in co2_trend if c < 800 or c > 1500)
        return (temp_stress + humidity_stress + co2_stress) / len(temp_trend)

    # Existing code for fetching data and calculating trends...

    stress_index = calculate_stress_index(temp_trend, humidity_trend, co2_trend)
    
    analysis += f"\nPlant Stress Index: {stress_index:.2f}\n"
    if stress_index > 0.3:
        analysis += "Warning: Plants are experiencing significant stress. Immediate action recommended.\n"
    elif stress_index > 0.1:
        analysis += "Caution: Moderate stress levels detected. Monitor closely and consider adjustments.\n"
    else:
        analysis += "Good: Plants are experiencing low stress levels.\n"

    # Add spectral analysis for cyclical patterns
    from scipy.fft import fft

    def detect_cyclical_patterns(data):
        fft_result = fft(data)
        frequencies = np.fft.fftfreq(len(data), 1)
        dominant_freq = frequencies[np.argmax(np.abs(fft_result))]
        if dominant_freq > 0:
            cycle_length = 1 / dominant_freq
            return f"Detected cyclical pattern with approximate length of {cycle_length:.1f} days"
        return "No significant cyclical patterns detected"

    analysis += "\nCyclical Pattern Analysis:\n"
    analysis += f"Temperature: {detect_cyclical_patterns(temp_trend)}\n"
    analysis += f"Humidity: {detect_cyclical_patterns(humidity_trend)}\n"
    analysis += f"CO2 Levels: {detect_cyclical_patterns(co2_trend)}\n"

    return analysis

# Keep all other existing functions...

def calculate_plant_health_score(temp, humidity, co2, light, n, p, k):
    # Existing code...
    
    # Add more sophisticated scoring
    def calculate_interaction_score(params):
        # Consider interactions between parameters
        vpd_score = 100 - abs(calculate_vpd(temp, humidity) - 1.0) * 20  # Optimal VPD around 1.0 kPa
        nutrient_balance_score = 100 - (abs(n/p - 10) + abs(n/k - 1)) * 5  # Ideal N:P:K ratio
        return (vpd_score + nutrient_balance_score) / 2

    interaction_score = calculate_interaction_score({'temp': temp, 'humidity': humidity, 'n': n, 'p': p, 'k': k})
    
    # Calculate weighted average (you can adjust weights based on importance)
    total_score = (temp_score * 1.5 + humidity_score * 1.5 + co2_score * 1.2 + 
                   light_score * 1.2 + n_score + p_score + k_score + interaction_score * 2) / 10.4
    
    return round(total_score, 2)

# New function for advanced pest and disease prediction
def predict_pest_disease_risk():
    recent_env_data = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).limit(168).all()  # Last 7 days
    
    if not recent_env_data:
        return "Insufficient data for pest and disease risk prediction"
    
    temp_data = [d.temperature for d in recent_env_data]
    humidity_data = [d.humidity for d in recent_env_data]
    
    avg_temp = np.mean(temp_data)
    avg_humidity = np.mean(humidity_data)
    temp_variance = np.var(temp_data)
    humidity_variance = np.var(humidity_data)
    
    # Simple risk model (can be expanded with more sophisticated machine learning models)
    fungal_risk = (avg_humidity > 70) * 0.6 + (avg_temp > 20 and avg_temp < 30) * 0.4
    insect_risk = (avg_temp > 25) * 0.5 + (humidity_variance > 100) * 0.3 + (temp_variance > 25) * 0.2
    
    risk_assessment = "Pest and Disease Risk Assessment:\n"
    risk_assessment += f"Fungal Disease Risk: {'High' if fungal_risk > 0.6 else 'Moderate' if fungal_risk > 0.3 else 'Low'}\n"
    risk_assessment += f"Insect Pest Risk: {'High' if insect_risk > 0.6 else 'Moderate' if insect_risk > 0.3 else 'Low'}\n"
    
    if fungal_risk > 0.6:
        risk_assessment += "- Consider preventive fungicide application and improve air circulation\n"
    if insect_risk > 0.6:
        risk_assessment += "- Implement regular scouting for insect pests and consider preventive measures\n"
    
    return risk_assessment

# New function for advanced nutrient optimization
def optimize_nutrient_plan(batch_id):
    batch = BatchData.query.filter_by(batch_id=batch_id).first()
    if not batch:
        return "Batch not found"
    
    nutrient_data = NutrientData.query.filter(
        NutrientData.timestamp >= batch.harvest_date - timedelta(days=90),
        NutrientData.timestamp <= batch.harvest_date
    ).all()
    
    if not nutrient_data:
        return "Insufficient nutrient data for optimization"
    
    current_n = np.mean([d.nitrogen_level for d in nutrient_data])
    current_p = np.mean([d.phosphorus_level for d in nutrient_data])
    current_k = np.mean([d.potassium_level for d in nutrient_data])
    
    # Ideal ratios (can be adjusted based on specific crop requirements)
    ideal_n_p_ratio = 10  # N:P ratio
    ideal_n_k_ratio = 1   # N:K ratio
    
    optimization_plan = "Nutrient Optimization Plan:\n"
    optimization_plan += f"Current N:P:K Ratio - {current_n:.1f}:{current_p:.1f}:{current_k:.1f}\n"
    
    if current_n / current_p < ideal_n_p_ratio:
        optimization_plan += "- Increase nitrogen or decrease phosphorus\n"
    elif current_n / current_p > ideal_n_p_ratio:
        optimization_plan += "- Decrease nitrogen or increase phosphorus\n"
    
    if current_n / current_k < ideal_n_k_ratio:
        optimization_plan += "- Increase nitrogen or decrease potassium\n"
    elif current_n / current_k > ideal_n_k_ratio:
        optimization_plan += "- Decrease nitrogen or increase potassium\n"
    
    # Calculate recommended adjustments
    target_n = max(current_n, (current_p * ideal_n_p_ratio + current_k * ideal_n_k_ratio) / 2)
    target_p = target_n / ideal_n_p_ratio
    target_k = target_n / ideal_n_k_ratio
    
    optimization_plan += f"\nRecommended N:P:K Ratio - {target_n:.1f}:{target_p:.1f}:{target_k:.1f}\n"
    optimization_plan += f"Adjust N by {target_n - current_n:.1f} ppm\n"
    optimization_plan += f"Adjust P by {target_p - current_p:.1f} ppm\n"
    optimization_plan += f"Adjust K by {target_k - current_k:.1f} ppm\n"
    
    return optimization_plan
