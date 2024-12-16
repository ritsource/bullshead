import random
from flask import Blueprint, jsonify, request

# Initialize blueprint
main_bp = Blueprint('main', __name__)

# Basic routes
@main_bp.route('/')
def home():
    return jsonify({"message": "Welcome to the API"})

@main_bp.route('/health')
def health_check():
    return jsonify({"status": "healthy"})

# API v1 routes
@main_bp.route('/api/v1/example', methods=['GET', 'POST'])
def example_endpoint():
    if request.method == 'GET':
        return jsonify({"message": "This is a GET request"})
    elif request.method == 'POST':
        data = request.get_json()
        return jsonify({
            "message": "Received POST request",
            "data": data
        })

@main_bp.route('/api/v1/trade/status', methods=['GET'])
def get_trade_status():
    data = request.get_json()
    
    # Validate required fields
    if not all(key in data for key in ['token', 'price', 'ask', 'time']):
        return jsonify({
            "error": "Missing required fields. Please provide token, price, ask and time"
        }), 400
    
    # Generate trade recommendation    
    decision = random.choice(['buy', 'sell'])
    confidence = random.uniform(0.6, 0.95)
    
    return jsonify({
        "token": data['token'],
        "recommendation": decision,
        "confidence": round(confidence, 2),
        "timestamp": data['time']
    })