from flask import Blueprint, request, jsonify
from app.services.model_service import ModelService

predict_bp = Blueprint('predict', __name__)
model_service = ModelService()

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = model_service.get_prediction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400