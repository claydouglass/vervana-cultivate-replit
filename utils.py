import os
from dotenv import load_dotenv
from openai import OpenAI
import base64
import requests
import csv
from io import StringIO
from datetime import datetime, timedelta
from models import db, EnvironmentalData, BatchData, NutrientData, CultivationSchedule, ProcessingSchedule
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import joblib
import pandas as pd
import math

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def batch_exists(batch_id):
    """
    Check if a batch with the given batch_id exists in the database.
    
    :param batch_id: The batch_id to check
    :return: True if the batch exists, False otherwise
    """
    return db.session.query(BatchData.query.filter_by(batch_id=batch_id).exists()).scalar()

def fetch_csv_data(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return pd.read_csv(StringIO(response.text))

# Add this constant at the top of the file
ENVIRONMENTAL_DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-OvMFLTAzu2aSDIEF1ma2YADXDYN-ZjjPfMBB-PMvpbisTVVADFDnyJrLGYn-YJtYei8vQLu0-U9Z/pub?gid=149803876&single=true&output=csv"

# Update the fetch_and_store_historical_data function
def fetch_and_store_historical_data(start_date, end_date):
    df = fetch_csv_data(ENVIRONMENTAL_DATA_URL)
    
    # Convert ct_tm to datetime
    df['ct_tm'] = pd.to_datetime(df['ct_tm'], format='%m/%d/%y %H:%M')
    mask = (df['ct_tm'] >= start_date) & (df['ct_tm'] <= end_date)
    df_filtered = df.loc[mask]
    
    # Calculate light duration
    df_filtered['light_duration'] = df_filtered['light_status'].rolling(window=24, min_periods=1).sum()
    
    # Convert temperature from Fahrenheit to Celsius
    df_filtered['tp'] = (df_filtered['tp'] - 32) * 5/9
    
    for _, row in df_filtered.iterrows():
        env_data = EnvironmentalData(
            timestamp=row['ct_tm'],
            temperature=row['tp'],  # Now in Celsius
            humidity=row['hy'],
            co2_level=row['co2'],
            vpd=row['vpd'],
            light_duration=row['light_duration'],
            is_day=row['light_status'] == 1
        )
        db.session.add(env_data)
    
    db.session.commit()
    return f"Historical data from {start_date} to {end_date} fetched and stored successfully"

def import_recipes():
    """
    Import recipes from the provided Google Sheets CSV.
    """
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSHZY7IWvVORybjlct01p2jWvaqYRQ6Nva5BEqFThwHFC7tPY6E18gLQ985Wmz9hvn1oz2K15lLePE/pub?output=csv"
    df = fetch_csv_data(url)
    
    recipes = []
    for _, row in df.iterrows():
        recipe = {
            'stage': row['Stage'],
            'N(ppm)_day_avg': row['N (ppm)'],
            'P(ppm)_day_avg': row['P (ppm)'],
            'K(ppm)_day_avg': row['K (ppm)'],
            'Ca(ppm)_day_avg': row['Ca (ppm)'],
            'Mg(ppm)_day_avg': row['Mg (ppm)'],
            'S(ppm)_day_avg': row['S (ppm)'],
            'Fe(ppm)_day_avg': row['Fe (ppm)'],
            'Zn(ppm)_day_avg': row['Zn (ppm)'],
            'Mn(ppm)_day_avg': row['Mn (ppm)'],
            'pH_min': row['pH Min'],
            'pH_max': row['pH Max'],
            'EC_min': row['EC Min (mS/cm)'],
            'EC_max': row['EC Max (mS/cm)']
        }
        recipes.append(recipe)
    
    return recipes

def import_environmental_recipes():
    """
    Import environmental recipes from the provided Google Sheets CSV.
    """
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-OvMFLTAzu2aSDIEF1ma2YADXDYN-ZjjPfMBB-PMvpbisTVVADFDnyJrLGYn-YJtYei8vQLu0-U9Z/pub?gid=149803876&single=true&output=csv"
    df = fetch_csv_data(url)
    
    recipes = []
    for _, row in df.iterrows():
        recipe = {
            'stage': row['stage'],
            'temp_day_avg': row['temp(c)_day_avg'],
            'temp_day_high': row['temp(c)_day_high'],
            'temp_day_low': row['temp(c)_day_low'],
            'temp_night_avg': row['temp(c)_night_avg'],
            'temp_night_high': row['temp(c)_night_high'],
            'temp_night_low': row['temp(c)_night_low'],
            'humid_day_avg': row['humid(%rh)_day_avg'],
            'humid_day_high': row['humid(%rh)_day_high'],
            'humid_day_low': row['humid(%rh)_day_low'],
            'humid_night_avg': row['humid(%rh)_night_avg'],
            'humid_night_high': row['humid(%rh)_night_high'],
            'humid_night_low': row['humid(%rh)_night_low'],
            'co2_day_avg': row['co2(ppm)_day_avg'],
            'co2_day_high': row['co2(ppm)_day_high'],
            'co2_day_low': row['co2(ppm)_day_low'],
            'co2_night_high': row['co2(ppm)_night_high'],
            'co2_night_low': row['co2(ppm)_night_low'],
            'vpd_high': row['vpd_high'],
            'vpd_low': row['vpd_low'],
            'light_duration': row['light_duration'],
            'light_ppfd': row['light_ppfd']
        }
        recipes.append(recipe)
    
    return recipes

def get_vpd_bounds(growth_phase):
    """Get VPD bounds based on the current growth phase from environmental recipes."""
    recipes = import_environmental_recipes()
    recipe = next((r for r in recipes if r['stage'] == growth_phase), None)
    if recipe:
        return float(recipe['vpd_low']), float(recipe['vpd_high'])
    else:
        # Return a default range if the growth phase is not found
        return 0.8, 1.2

def import_cultivation_schedule():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSugT9wd7xG_wyn3tbFE2MSOiNCAj_HhmvEIWrIhBEjSXzCwF-1LOrzLzvW6NLZyL8PjVC-8O5uCzGi/pub?output=csv"
    df = fetch_csv_data(url)
    
    schedule = []
    for _, row in df.iterrows():
        if row['Phase'] != 'Processing':  # Only include cultivation tasks
            day_schedule = {
                'Phase': row['Phase'],
                'Day': row['Day'],
                'Tasks': [task for task in row.index[2:] if row[task] == 'X']
            }
            schedule.append(day_schedule)
    
    return schedule

def import_processing_schedule():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSugT9wd7xG_wyn3tbFE2MSOiNCAj_HhmvEIWrIhBEjSXzCwF-1LOrzLzvW6NLZyL8PjVC-8O5uCzGi/pub?output=csv"
    df = fetch_csv_data(url)
    
    schedule = []
    for _, row in df.iterrows():
        if row['Phase'] == 'Processing':  # Only include processing tasks
            day_schedule = {
                'Phase': row['Phase'],
                'Day': row['Day'],
                'Tasks': [task for task in row.index[2:] if row[task] == 'X']
            }
            schedule.append(day_schedule)
    
    return schedule

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

def predict_yield(batch_id):
    def calculate_growing_degree_days(temp_data):
        base_temp = 10  # This is now in Celsius (equivalent to about 50°F)
        return sum(max(0, (temp - base_temp)) for temp in temp_data)

    batches = BatchData.query.all()
    
    X = []
    y = []
    for batch in batches:
        terpenes = batch.terpene_profile.split(',')
        terpene_values = [float(t.split(':')[1]) for t in terpenes]
        
        env_data = EnvironmentalData.query.filter(
            EnvironmentalData.timestamp >= batch.harvest_date - timedelta(days=90),
            EnvironmentalData.timestamp <= batch.harvest_date
        ).all()
        
        nutrient_data = NutrientData.query.filter(
            NutrientData.timestamp >= batch.harvest_date - timedelta(days=90),
            NutrientData.timestamp <= batch.harvest_date
        ).all()
        
        if env_data and nutrient_data:
            temp_data = [d.temperature for d in env_data]
            humidity_data = [d.humidity for d in env_data]
            co2_data = [d.co2_level for d in env_data]
            
            avg_temp = np.mean(temp_data)
            avg_humidity = np.mean(humidity_data)
            avg_co2 = np.mean(co2_data)
            
            avg_n = np.mean([d.nitrogen_level for d in nutrient_data])
            avg_p = np.mean([d.phosphorus_level for d in nutrient_data])
            avg_k = np.mean([d.potassium_level for d in nutrient_data])
            
            growing_degree_days = calculate_growing_degree_days(temp_data)
            vpd = np.mean(calculate_vapor_pressure_deficit(temp_data, humidity_data))
            dli = calculate_dli(avg_light_intensity)
            
            features = [batch.thc_level] + terpene_values + [
                avg_temp, avg_humidity, avg_co2,
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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    joblib.dump(model, 'yield_prediction_model.joblib')
    joblib.dump(scaler, 'yield_prediction_scaler.joblib')
    
    target_batch = BatchData.query.filter_by(batch_id=batch_id).first()
    if not target_batch:
        return "Batch not found"
    
    target_env_data = EnvironmentalData.query.filter(
        EnvironmentalData.timestamp >= target_batch.harvest_date - timedelta(days=90),
        EnvironmentalData.timestamp <= target_batch.harvest_date
    ).all()
    
    target_nutrient_data = NutrientData.query.filter(
        EnvironmentalData.timestamp >= target_batch.harvest_date - timedelta(days=90),
        EnvironmentalData.timestamp <= target_batch.harvest_date
    ).all()
    
    if not target_env_data or not target_nutrient_data:
        return "Insufficient environmental or nutrient data for yield prediction"
    
    temp_data = [d.temperature for d in target_env_data]
    humidity_data = [d.humidity for d in target_env_data]
    co2_data = [d.co2_level for d in target_env_data]
    
    avg_temp = np.mean(temp_data)
    avg_humidity = np.mean(humidity_data)
    avg_co2 = np.mean(co2_data)
    
    avg_n = np.mean([d.nitrogen_level for d in target_nutrient_data])
    avg_p = np.mean([d.phosphorus_level for d in target_nutrient_data])
    avg_k = np.mean([d.potassium_level for d in target_nutrient_data])
    
    growing_degree_days = calculate_growing_degree_days(temp_data)
    vpd = np.mean(calculate_vapor_pressure_deficit(temp_data, humidity_data))
    dli = calculate_dli(avg_light_intensity)
    
    target_features = [target_batch.thc_level] + \
                      [float(t.split(':')[1]) for t in target_batch.terpene_profile.split(',')] + \
                      [avg_temp, avg_humidity, avg_co2,
                       avg_n, avg_p, avg_k, growing_degree_days, vpd, dli,
                       np.std(temp_data), np.std(humidity_data), np.std(co2_data),
                       max(temp_data) - min(temp_data),
                       max(humidity_data) - min(humidity_data),
                       max(co2_data) - min(co2_data)]
    
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
    def calculate_stress_index(temp_trend, humidity_trend, co2_trend):
        temp_stress = sum(1 for t in temp_trend if t < 10 or t > 30)  # Adjusted for Celsius
        humidity_stress = sum(1 for h in humidity_trend if h < 30 or h > 70)
        co2_stress = sum(1 for c in co2_trend if c < 800 or c > 1500)
        return (temp_stress + humidity_stress + co2_stress) / len(temp_trend)

    recent_env_data = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).limit(168).all()
    
    if not recent_env_data:
        return "Insufficient data for plant health analysis"
    
    temp_trend = [d.temperature for d in recent_env_data]
    humidity_trend = [d.humidity for d in recent_env_data]
    co2_trend = [d.co2_level for d in recent_env_data]
    
    stress_index = calculate_stress_index(temp_trend, humidity_trend, co2_trend)
    
    analysis = f"Plant Stress Index: {stress_index:.2f}\n"
    if stress_index > 0.3:
        analysis += "Warning: Plants are experiencing significant stress. Immediate action recommended.\n"
    elif stress_index > 0.1:
        analysis += "Caution: Moderate stress levels detected. Monitor closely and consider adjustments.\n"
    else:
        analysis += "Good: Plants are experiencing low stress levels.\n"

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

def calculate_plant_health_score(temp, humidity, co2, n, p, k):
    def calculate_interaction_score(params):
        vpd_score = 100 - abs(calculate_vpd(temp, humidity) - 1.0) * 20  # Optimal VPD around 1.0 kPa
        nutrient_balance_score = 100 - (abs(n/p - 10) + abs(n/k - 1)) * 5  # Ideal N:P:K ratio
        return (vpd_score + nutrient_balance_score) / 2

    interaction_score = calculate_interaction_score({'temp': temp, 'humidity': humidity, 'n': n, 'p': p, 'k': k})
    
    total_score = (temp_score * 1.5 + humidity_score * 1.5 + co2_score * 1.2 + 
                   n_score + p_score + k_score + interaction_score * 2) / 10.4
    
    return round(total_score, 2)

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
    
    fungal_risk = (avg_humidity > 70) * 0.6 + (avg_temp > 10 and avg_temp < 30) * 0.4
    insect_risk = (avg_temp > 25) * 0.5 + (humidity_variance > 100) * 0.3 + (temp_variance > 25) * 0.2
    
    risk_assessment = "Pest and Disease Risk Assessment:\n"
    risk_assessment += f"Fungal Disease Risk: {'High' if fungal_risk > 0.6 else 'Moderate' if fungal_risk > 0.3 else 'Low'}\n"
    risk_assessment += f"Insect Pest Risk: {'High' if insect_risk > 0.6 else 'Moderate' if insect_risk > 0.3 else 'Low'}\n"
    
    if fungal_risk > 0.6:
        risk_assessment += "- Consider preventive fungicide application and improve air circulation\n"
    if insect_risk > 0.6:
        risk_assessment += "- Implement regular scouting for insect pests and consider preventive measures\n"
    
    return risk_assessment

def optimize_nutrient_plan(batch_id):
    batch = BatchData.query.filter_by(batch_id=batch_id).first()
    if not batch:
        return "Batch not found"
    
    recipes = import_recipes()
    
    stages = [
        ("Veg Week 1-2", batch.veg_week_1_2_start, batch.veg_week_1_2_end),
        ("Veg Week 3", batch.veg_week_3_start, batch.veg_week_3_end),
        ("Flower Week 1-3", batch.flower_week_1_3_start, batch.flower_week_1_3_end),
        ("Flower Week 4-6.5", batch.flower_week_4_6_5_start, batch.flower_week_4_6_5_end),
        ("Flower Week 6.5-8.5", batch.flower_week_6_5_8_5_start, batch.flower_week_6_5_8_5_end),
        ("Flower Week 8.5+ Harvest", batch.flower_week_8_5_plus_start, batch.flower_week_8_5_plus_end)
    ]
    
    current_stage = None
    for stage, start, end in reversed(stages):
        if start and start <= datetime.now():
            if not end or end >= datetime.now():
                current_stage = stage
                break
            elif end < datetime.now():
                current_stage = stage
                break
    
    if not current_stage:
        return "Unable to determine current growth stage"
    
    current_recipe = next((r for r in recipes if r['stage'] == current_stage), None)
    
    if not current_recipe:
        return f"No recipe found for stage: {current_stage}"
    
    latest_nutrient = NutrientData.query.order_by(NutrientData.timestamp.desc()).first()
    
    if not latest_nutrient:
        return "No nutrient data available"
    
    recommendations = []
    for nutrient, target in [('nitrogen', 'N'), ('phosphorus', 'P'), ('potassium', 'K')]:
        current_level = getattr(latest_nutrient, f"{nutrient}_level")
        target_level = float(current_recipe[f'{target}(ppm)_day_avg'])
        
        if current_level < target_level * 0.9:
            recommendations.append(f"Increase {nutrient} to reach {target_level} ppm")
        elif current_level > target_level * 1.1:
            recommendations.append(f"Decrease {nutrient} to reach {target_level} ppm")
    
    latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
    if latest_env:
        if latest_env.ph < float(current_recipe['pH_min']):
            recommendations.append(f"Increase pH to reach minimum of {current_recipe['pH_min']}")
        elif latest_env.ph > float(current_recipe['pH_max']):
            recommendations.append(f"Decrease pH to reach maximum of {current_recipe['pH_max']}")
        
        if latest_env.ec < float(current_recipe['EC_min']):
            recommendations.append(f"Increase EC to reach minimum of {current_recipe['EC_min']} mS/cm")
        elif latest_env.ec > float(current_recipe['EC_max']):
            recommendations.append(f"Decrease EC to reach maximum of {current_recipe['EC_max']} mS/cm")
    
    if not recommendations:
        return "Current nutrient levels, pH, and EC are within optimal range"
    
    return "\n".join(recommendations)

def get_adjustment_recommendations():
    recent_data = db.session.query(EnvironmentalData, NutrientData).filter(
        EnvironmentalData.timestamp == NutrientData.timestamp
    ).order_by(EnvironmentalData.timestamp.desc()).limit(24).all()  # Last 24 hours of data

    if not recent_data:
        return "Insufficient data for recommendations"

    recommendations = []

    avg_temp = sum(data.EnvironmentalData.temperature for data in recent_data) / len(recent_data)
    if avg_temp < 10:
        recommendations.append("Consider increasing temperature")
    elif avg_temp > 30:
        recommendations.append("Consider decreasing temperature")

    avg_humidity = sum(data.EnvironmentalData.humidity for data in recent_data) / len(recent_data)
    if avg_humidity < 40:
        recommendations.append("Consider increasing humidity")
    elif avg_humidity > 60:
        recommendations.append("Consider decreasing humidity")

    avg_nitrogen = sum(data.NutrientData.nitrogen_level for data in recent_data) / len(recent_data)
    if avg_nitrogen < 100:
        recommendations.append("Consider increasing nitrogen levels")
    elif avg_nitrogen > 200:
        recommendations.append("Consider decreasing nitrogen levels")

    if not recommendations:
        return "No adjustments recommended at this time"
    
    return "\n".join(recommendations)

def get_current_growth_phase(batch_id):
       """Determine the current growth phase based on the batch data."""
       batch = BatchData.query.filter_by(batch_id=batch_id).first()
       if not batch:
           return "Batch not found"

       current_date = datetime.now()
       
       if batch.veg_week_1_2_start and current_date < batch.veg_week_1_2_end:
           return "Veg Week 1-2"
       elif batch.veg_week_3_start and current_date < batch.veg_week_3_end:
           return "Veg Week 3"
       elif batch.flower_week_1_3_start and current_date < batch.flower_week_1_3_end:
           return "Flower Week 1-3"
       elif batch.flower_week_4_6_5_start and current_date < batch.flower_week_4_6_5_end:
           return "Flower Week 4-6.5"
       elif batch.flower_week_6_5_8_5_start and current_date < batch.flower_week_6_5_8_5_end:
           return "Flower Week 6.5-8.5"
       elif batch.flower_week_8_5_plus_start and batch.flower_week_8_5_plus_end and current_date < batch.flower_week_8_5_plus_end:
           return "Flower Week 8.5+ Harvest"
       elif batch.harvest_date and current_date >= batch.harvest_date:
           return "Post-Harvest"
       else:
           return "Unable to determine current growth stage"

def compare_batch_performance(batch_id1, batch_id2):
    batch1 = BatchData.query.filter_by(batch_id=batch_id1).first()
    batch2 = BatchData.query.filter_by(batch_id=batch_id2).first()
    
    if not batch1 or not batch2:
        return "One or both batches not found"
    
    def get_completed_stages(batch):
        completed_stages = []
        if batch.veg_week_1_2_end:
            completed_stages.append(("Veg Week 1-2", batch.veg_week_1_2_start, batch.veg_week_1_2_end))
        if batch.veg_week_3_end:
            completed_stages.append(("Veg Week 3", batch.veg_week_3_start, batch.veg_week_3_end))
        if batch.flower_week_1_3_end:
            completed_stages.append(("Flower Week 1-3", batch.flower_week_1_3_start, batch.flower_week_1_3_end))
        if batch.flower_week_4_6_5_end:
            completed_stages.append(("Flower Week 4-6.5", batch.flower_week_4_6_5_start, batch.flower_week_4_6_5_end))
        if batch.flower_week_6_5_8_5_end:
            completed_stages.append(("Flower Week 6.5-8.5", batch.flower_week_6_5_8_5_start, batch.flower_week_6_5_8_5_end))
        if batch.flower_week_8_5_plus_end:
            completed_stages.append(("Flower Week 8.5+", batch.flower_week_8_5_plus_start, batch.flower_week_8_5_plus_end))
        return completed_stages

    completed_stages1 = get_completed_stages(batch1)
    completed_stages2 = get_completed_stages(batch2)
    
    common_stages = [stage for stage in completed_stages1 if stage[0] in [s[0] for s in completed_stages2]]
    
    comparison = []
    for stage, start1, end1 in common_stages:
        start2, end2 = next((s[1], s[2]) for s in completed_stages2 if s[0] == stage)
        
        env_data1 = EnvironmentalData.query.filter(
            EnvironmentalData.timestamp >= start1,
            EnvironmentalData.timestamp <= end1
        ).order_by(EnvironmentalData.timestamp).all()
        
        env_data2 = EnvironmentalData.query.filter(
            EnvironmentalData.timestamp >= start2,
            EnvironmentalData.timestamp <= end2
        ).order_by(EnvironmentalData.timestamp).all()
        
        nutrient_data1 = NutrientData.query.filter(
            NutrientData.timestamp >= start1,
            NutrientData.timestamp <= end1
        ).order_by(NutrientData.timestamp).all()
        
        nutrient_data2 = NutrientData.query.filter(
            NutrientData.timestamp >= start2,
            NutrientData.timestamp <= end2
        ).order_by(NutrientData.timestamp).all()
        
        stage_comparison = {
            "stage": stage,
            "environmental_comparison": compare_environmental_data(env_data1, env_data2),
            "nutrient_comparison": compare_nutrient_data(nutrient_data1, nutrient_data2)
        }
        comparison.append(stage_comparison)
    
    final_comparison = {}
    if batch1.yield_amount is not None and batch2.yield_amount is not None:
        final_comparison["yield_difference"] = batch1.yield_amount - batch2.yield_amount
    if batch1.thc_level is not None and batch2.thc_level is not None:
        final_comparison["thc_level_difference"] = batch1.thc_level - batch2.thc_level
    if batch1.terpene_profile and batch2.terpene_profile:
        final_comparison["terpene_profile_difference"] = compare_terpene_profiles(batch1.terpene_profile, batch2.terpene_profile)
    
    return {
        "stage_comparisons": comparison,
        "final_comparison": final_comparison
    }

def compare_environmental_data(data1, data2):
    return {
        "avg_temperature_diff": np.mean([d1.temperature for d1 in data1]) - np.mean([d2.temperature for d2 in data2]),
        "avg_humidity_diff": np.mean([d1.humidity for d1 in data1]) - np.mean([d2.humidity for d2 in data2]),
        "avg_co2_diff": np.mean([d1.co2_level for d1 in data1]) - np.mean([d2.co2_level for d2 in data2]),
        "avg_light_intensity_diff": np.mean([d1.light_intensity for d1 in data1]) - np.mean([d2.light_intensity for d2 in data2])
    }

def compare_nutrient_data(data1, data2):
    return {
        "avg_nitrogen_diff": np.mean([d1.nitrogen_level for d1 in data1]) - np.mean([d2.nitrogen_level for d2 in data2]),
        "avg_phosphorus_diff": np.mean([d1.phosphorus_level for d1 in data1]) - np.mean([d2.phosphorus_level for d2 in data2]),
        "avg_potassium_diff": np.mean([d1.potassium_level for d1 in data1]) - np.mean([d2.potassium_level for d2 in data2])
    }

def compare_terpene_profiles(profile1, profile2):
    terpenes1 = dict(item.split(':') for item in profile1.split(','))
    terpenes2 = dict(item.split(':') for item in profile2.split(','))
    
    all_terpenes = set(terpenes1.keys()) | set(terpenes2.keys())
    
    differences = {}
    for terpene in all_terpenes:
        value1 = float(terpenes1.get(terpene, 0))
        value2 = float(terpenes2.get(terpene, 0))
        differences[terpene] = value1 - value2
    
    return differences

def early_warning_system():
    recent_env_data = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).limit(168).all()  # Last 7 days
    recent_nutrient_data = NutrientData.query.order_by(NutrientData.timestamp.desc()).limit(168).all()  # Last 7 days

    if not recent_env_data or not recent_nutrient_data:
        return "Insufficient data for early warning analysis"

    warnings = []

    for i in range(1, len(recent_env_data)):
        temp_change = abs(recent_env_data[i].temperature - recent_env_data[i-1].temperature)
        humidity_change = abs(recent_env_data[i].humidity - recent_env_data[i-1].humidity)
        co2_change = abs(recent_env_data[i].co2_level - recent_env_data[i-1].co2_level)

        if temp_change > 5:
            warnings.append(f"Rapid temperature change detected at {recent_env_data[i].timestamp}")
        if humidity_change > 10:
            warnings.append(f"Rapid humidity change detected at {recent_env_data[i].timestamp}")
        if co2_change > 200:
            warnings.append(f"Rapid CO2 level change detected at {recent_env_data[i].timestamp}")

    for data in recent_nutrient_data:
        n_p_ratio = data.nitrogen_level / data.phosphorus_level if data.phosphorus_level else 0
        n_k_ratio = data.nitrogen_level / data.potassium_level if data.potassium_level else 0

        if n_p_ratio > 12 or n_p_ratio < 8:
            warnings.append(f"N:P ratio out of optimal range at {data.timestamp}")
        if n_k_ratio > 1.2 or n_k_ratio < 0.8:
            warnings.append(f"N:K ratio out of optimal range at {data.timestamp}")

    pest_disease_risk = predict_pest_disease_risk()
    if "High" in pest_disease_risk:
        warnings.append("High risk of pest or disease detected")

    if not warnings:
        return "No early warnings detected"
    else:
        return "\n".join(warnings)

def schedule_cultivation_tasks(batch_id, current_day):
    cultivation_schedule = import_cultivation_schedule()
    
    day_schedule = next((day for day in cultivation_schedule if day['Day'] == current_day), None)
    
    if not day_schedule:
        return f"No cultivation schedule found for day {current_day}"
    
    for task in day_schedule['Tasks']:
        new_task = CultivationSchedule(
            batch_id=batch_id,
            day=current_day,
            task=task
        )
        db.session.add(new_task)
    
    db.session.commit()
    return f"Scheduled {len(day_schedule['Tasks'])} cultivation tasks for batch {batch_id} on day {current_day}"

def schedule_processing_tasks(batch_id, current_day):
    processing_schedule = import_processing_schedule()
    
    day_schedule = next((day for day in processing_schedule if day['Day'] == current_day), None)
    
    if not day_schedule:
        return f"No processing schedule found for day {current_day}"
    
    for task in day_schedule['Tasks']:
        new_task = ProcessingSchedule(
            batch_id=batch_id,
            day=current_day,
            task=task
        )
        db.session.add(new_task)
    
    db.session.commit()
    return f"Scheduled {len(day_schedule['Tasks'])} processing tasks for batch {batch_id} on day {current_day}"

def get_cultivation_schedule(batch_id, current_day):
    cultivation_schedule = import_cultivation_schedule()
    
    day_schedule = next((day for day in cultivation_schedule if day['Day'] == current_day), None)
    
    if not day_schedule:
        return f"No cultivation schedule found for day {current_day}"
    
    scheduled_tasks = CultivationSchedule.query.filter_by(batch_id=batch_id, day=current_day).all()
    
    schedule = {
        'Phase': day_schedule['Phase'],
        'Day': day_schedule['Day'],
        'Tasks': [
            {
                'task': task,
                'scheduled': any(st.task == task for st in scheduled_tasks),
                'completed': any(st.task == task and st.completed for st in scheduled_tasks)
            }
            for task in day_schedule['Tasks']
        ]
    }
    
    return schedule

def get_processing_schedule(batch_id, current_day):
    processing_schedule = import_processing_schedule()
    
    day_schedule = next((day for day in processing_schedule if day['Day'] == current_day), None)
    
    if not day_schedule:
        return f"No processing schedule found for day {current_day}"
    
    scheduled_tasks = ProcessingSchedule.query.filter_by(batch_id=batch_id, day=current_day).all()
    
    schedule = {
        'Phase': day_schedule['Phase'],
        'Day': day_schedule['Day'],
        'Tasks': [
            {
                'task': task,
                'scheduled': any(st.task == task for st in scheduled_tasks),
                'completed': any(st.task == task and st.completed for st in scheduled_tasks)
            }
            for task in day_schedule['Tasks']
        ]
    }
    
    return schedule

def optimize_environment_and_nutrients(batch_id):
    current_phase = get_current_growth_phase(batch_id)
    env_recipes = import_environmental_recipes()
    current_recipe = next((r for r in env_recipes if r['stage'] == current_phase), None)
    
    if not current_recipe:
        return "Unable to find environmental recipe for current growth phase"
    
    latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
    
    env_recommendations = []
    if latest_env:
        is_day = latest_env.is_day
        
        # Temperature
        temp_low = current_recipe['temp_day_low'] if is_day else current_recipe['temp_night_low']
        temp_high = current_recipe['temp_day_high'] if is_day else current_recipe['temp_night_high']
        if latest_env.temperature < float(temp_low):
            env_recommendations.append(f"Increase temperature to at least {temp_low}°C")
        elif latest_env.temperature > float(temp_high):
            env_recommendations.append(f"Decrease temperature to at most {temp_high}°C")
        
        # Humidity
        humid_low = current_recipe['humid_day_low'] if is_day else current_recipe['humid_night_low']
        humid_high = current_recipe['humid_day_high'] if is_day else current_recipe['humid_night_high']
        if latest_env.humidity < float(humid_low):
            env_recommendations.append(f"Increase humidity to at least {humid_low}%")
        elif latest_env.humidity > float(humid_high):
            env_recommendations.append(f"Decrease humidity to at most {humid_high}%")
        
        # CO2
        co2_low = current_recipe['co2_day_low'] if is_day else current_recipe['co2_night_low']
        co2_high = current_recipe['co2_day_high'] if is_day else current_recipe['co2_night_high']
        if latest_env.co2_level < float(co2_low):
            env_recommendations.append(f"Increase CO2 level to at least {co2_low} ppm")
        elif latest_env.co2_level > float(co2_high):
            env_recommendations.append(f"Decrease CO2 level to at most {co2_high} ppm")
        
        # VPD
        if latest_env.vpd < float(current_recipe['vpd_low']):
            env_recommendations.append(f"Increase VPD to at least {current_recipe['vpd_low']} kPa")
        elif latest_env.vpd > float(current_recipe['vpd_high']):
            env_recommendations.append(f"Decrease VPD to at most {current_recipe['vpd_high']} kPa")
    
    nutrient_recommendations = optimize_nutrient_plan(batch_id)
    
    return {
        "environmental_recommendations": env_recommendations if env_recommendations else "No environmental adjustments needed",
        "nutrient_recommendations": nutrient_recommendations
    }
