from flask import Flask, render_template, request, jsonify
from models import db, EnvironmentalData, BatchData
from config import Config
from utils import analyze_image, get_chatbot_response, fetch_and_store_historical_data
from datetime import datetime, timedelta
import os
import uuid

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

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
            batch_id=request.form['batch_id'],
            harvest_date=datetime.strptime(request.form['harvest_date'], '%Y-%m-%d'),
            thc_level=float(request.form['thc_level']),
            terpene_profile=request.form['terpene_profile']
        )
        db.session.add(new_batch)
        db.session.commit()
    batches = BatchData.query.all()
    return render_template('batch_data.html', batches=batches)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if EnvironmentalData.query.first() is None:
            fetch_and_store_historical_data()
    app.run(host="0.0.0.0", port=5000)
