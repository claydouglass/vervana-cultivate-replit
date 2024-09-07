from flask import Flask, render_template, request, jsonify
from flask_migrate import Migrate
from models import db, EnvironmentalData, BatchData, NutrientData, CultivationSchedule, ProcessingSchedule
from config import Config
from utils import (
    analyze_image,
    get_chatbot_response,
    fetch_and_store_historical_data,
    get_adjustment_recommendations,
    predict_yield,
    analyze_plant_health_trends,
    compare_batch_performance,
    early_warning_system,
    predict_pest_disease_risk,
    optimize_nutrient_plan,
    import_recipes,
    import_cultivation_schedule,
    import_processing_schedule,
    schedule_cultivation_tasks,
    schedule_processing_tasks,
    get_cultivation_schedule,
    get_processing_schedule,
    optimize_environment_and_nutrients,
    calculate_vpd,
    get_current_growth_phase,
    get_vpd_bounds,
    get_recent_environmental_data
)
from datetime import datetime, timedelta
import os
import uuid

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
migrate = Migrate(app, db)

# Ensure UPLOAD_FOLDER is set and the directory exists
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/environmental_data')
def get_environmental_data():
    data = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.asc()).all()
    
    return jsonify([{
        'temperature': d.temperature,
        'humidity': d.humidity,
        'co2_level': d.co2_level,
        'light_intensity': d.light_intensity,
        'timestamp': d.timestamp.isoformat()
    } for d in data])

@app.route('/api/image_analysis', methods=['POST'])
def image_analysis():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if image and allowed_file(image.filename):
        filename = str(uuid.uuid4()) + '.' + image.filename.rsplit('.', 1)[1].lower()
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        analysis_result = analyze_image(image_path)
        os.remove(image_path)  # Clean up the temporary file
        return jsonify({'analysis': analysis_result})
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    response = get_chatbot_response(user_input)
    return jsonify({'response': response})

@app.route('/batch_data', methods=['GET', 'POST'])
def batch_data():
    if request.method == 'POST':
        new_batch = BatchData(
            batch_number=int(request.form['batch_number']),
            batch_id=request.form['batch_id'],
            veg_week_1_2_start=datetime.strptime(request.form['veg_week_1_2_start'], '%Y-%m-%d'),
            veg_week_1_2_end=datetime.strptime(request.form['veg_week_1_2_end'], '%Y-%m-%d'),
            veg_week_3_start=datetime.strptime(request.form['veg_week_3_start'], '%Y-%m-%d'),
            veg_week_3_end=datetime.strptime(request.form['veg_week_3_end'], '%Y-%m-%d'),
            flower_week_1_3_start=datetime.strptime(request.form['flower_week_1_3_start'], '%Y-%m-%d'),
            flower_week_1_3_end=datetime.strptime(request.form['flower_week_1_3_end'], '%Y-%m-%d'),
            flower_week_4_6_5_start=datetime.strptime(request.form['flower_week_4_6_5_start'], '%Y-%m-%d'),
            flower_week_4_6_5_end=datetime.strptime(request.form['flower_week_4_6_5_end'], '%Y-%m-%d'),
            flower_week_6_5_8_5_start=datetime.strptime(request.form['flower_week_6_5_8_5_start'], '%Y-%m-%d'),
            flower_week_6_5_8_5_end=datetime.strptime(request.form['flower_week_6_5_8_5_end'], '%Y-%m-%d'),
            flower_week_8_5_plus_start=datetime.strptime(request.form.get('flower_week_8_5_plus_start', ''), '%Y-%m-%d') if request.form.get('flower_week_8_5_plus_start') else None,
            flower_week_8_5_plus_end=datetime.strptime(request.form.get('flower_week_8_5_plus_end', ''), '%Y-%m-%d') if request.form.get('flower_week_8_5_plus_end') else None,
            harvest_date=datetime.strptime(request.form['harvest_date'], '%Y-%m-%d'),
            drying_start=datetime.strptime(request.form['drying_start'], '%Y-%m-%d'),
            drying_end=datetime.strptime(request.form['drying_end'], '%Y-%m-%d'),
            curing_start=datetime.strptime(request.form['curing_start'], '%Y-%m-%d'),
            curing_end=datetime.strptime(request.form['curing_end'], '%Y-%m-%d'),
            yield_amount=float(request.form['yield_amount']),
            thc_level=float(request.form['thc_level']),
            terpene_profile=request.form['terpene_profile']
        )
        db.session.add(new_batch)
        db.session.commit()
    batches = BatchData.query.all()
    return render_template('batch_data.html', batches=batches)

@app.route('/api/adjust_environment', methods=['GET'])
def adjust_environment():
    recommendations = get_adjustment_recommendations()
    return jsonify({'recommendations': recommendations})

@app.route('/api/predict_yield/<batch_id>', methods=['GET'])
def get_yield_prediction(batch_id):
    prediction_result = predict_yield(batch_id)
    if isinstance(prediction_result, str):
        return jsonify({'error': prediction_result}), 400
    
    return jsonify({
        'prediction': prediction_result['predicted_yield'],
        'model_performance': prediction_result['model_performance'],
        'feature_importance': prediction_result['feature_importance']
    })

@app.route('/api/plant_health_trends', methods=['GET'])
def get_plant_health_trends():
    trends = analyze_plant_health_trends()
    return jsonify({'trends': trends})

@app.route('/api/compare_batch/<batch_id>', methods=['GET'])
def compare_batch(batch_id):
    comparison = compare_batch_performance(batch_id)
    return jsonify({'comparison': comparison})

@app.route('/api/early_warning', methods=['GET'])
def get_early_warning():
    warnings = early_warning_system()
    return jsonify({'warnings': warnings})

@app.route('/api/pest_disease_risk', methods=['GET'])
def get_pest_disease_risk():
    risk_assessment = predict_pest_disease_risk()
    return jsonify({'risk_assessment': risk_assessment})

@app.route('/api/optimize_nutrients/<batch_id>', methods=['GET'])
def get_nutrient_optimization(batch_id):
    optimization_plan = optimize_nutrient_plan(batch_id)
    return jsonify({'optimization_plan': optimization_plan})

@app.route('/recommendations')
def recommendations():
    with app.app_context():
        return jsonify({"recommendations": get_adjustment_recommendations()})

@app.route('/api/schedule_cultivation/<batch_id>/<int:day>', methods=['POST'])
def schedule_cultivation(batch_id, day):
    result = schedule_cultivation_tasks(batch_id, day)
    return jsonify({'result': result})

@app.route('/api/schedule_processing/<batch_id>/<int:day>', methods=['POST'])
def schedule_processing(batch_id, day):
    result = schedule_processing_tasks(batch_id, day)
    return jsonify({'result': result})

@app.route('/api/cultivation_schedule/<batch_id>/<int:day>', methods=['GET'])
def get_cultivation_schedule_api(batch_id, day):
    schedule = get_cultivation_schedule(batch_id, day)
    return jsonify({'schedule': schedule})

@app.route('/api/processing_schedule/<batch_id>/<int:day>', methods=['GET'])
def get_processing_schedule_api(batch_id, day):
    schedule = get_processing_schedule(batch_id, day)
    return jsonify({'schedule': schedule})

@app.route('/api/optimize_environment_and_nutrients/<batch_id>', methods=['GET'])
def optimize_environment_and_nutrients_api(batch_id):
    optimization = optimize_environment_and_nutrients(batch_id)
    return jsonify(optimization)

@app.route('/api/dashboard_data')
def dashboard_data():
    current_phase = get_current_growth_phase()
    vpd_bounds = get_vpd_bounds(current_phase)
    env_data = get_recent_environmental_data()
    
    return jsonify({
        'current_phase': current_phase,
        'vpd_bounds': vpd_bounds,
        'environmental_data': env_data
    })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if EnvironmentalData.query.first() is None:
            fetch_and_store_historical_data()
    app.run(host="0.0.0.0", port=5000)
