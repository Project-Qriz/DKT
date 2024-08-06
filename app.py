from flask import Flask, request, jsonify
import logging
import traceback
from dkt import load_model, predict, UserActivityDataset, load_activities, DKTModel, train, save_model
from torch.utils.data import DataLoader
import torch

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_num_q():
    activities = load_activities()
    all_skill_ids = set(activity['skill_id'] for activity in activities)
    return len(all_skill_ids)

num_q = get_num_q()
model_path = 'dkt_model.pth'

try:
    model = load_model(model_path, num_q)
    if model is None:
        activities = load_activities()
        train_dataset = UserActivityDataset(activities, max_seq_len=100, num_q=num_q)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        model = DKTModel(num_q=num_q, emb_size=128, hidden_size=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train(model, train_loader, optimizer)
        save_model(model, num_q, model_path)
except Exception as e:
    print(f"Failed to load or train model: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        data = request.json
        logger.debug(f"Received data structure: {type(data)}")
        logger.debug(f"Activities structure: {type(data['activities'])}")
        logger.debug(f"First activity: {data['activities'][0]}")
        logger.debug(f"Received data: {data}")  # 받은 데이터 로깅
        
        user_id = data['user_id']
        activities = data['activities']
        
        logger.debug(f"User ID: {user_id}")
        logger.debug(f"Activities: {activities}")
        
        # activities가 리스트인지 확인
        if not isinstance(activities, list):
            raise ValueError("activities must be a list")

        # activities의 각 항목이 올바른 형식인지 확인
        for i, activity in enumerate(activities):
            if not isinstance(activity, dict) or 'question_id' not in activity or 'correct' not in activity or 'time_spent' not in activity:
                raise ValueError(f"Invalid activity format at index {i}: {activity}")

        dataset = UserActivityDataset(activities, user_id, max_seq_len=100, num_q=num_q)
        predictions = predict(model, dataset, num_q, user_id)
        logger.debug(f"Predictions: {predictions}")
        
        return jsonify({'user_id': user_id, 'predictions': predictions[user_id].tolist()})
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())  # 상세한 오류 트레이스백 로깅
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

if __name__ == '__main__':
    logger.info("Starting the application...")
    num_q = get_num_q()
    logger.info(f"Number of questions: {num_q}")
    model_path = 'dkt_model.pth'

    try:
        logger.info("Attempting to load the model...")
        model = load_model(model_path, num_q)
        if model is None:
            logger.info("Model not found. Training a new model...")
            activities = load_activities()
            train_dataset = UserActivityDataset(activities, max_seq_len=100, num_q=num_q)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            model = DKTModel(num_q=num_q, emb_size=128, hidden_size=256)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train(model, train_loader, optimizer)
            save_model(model, num_q, model_path)
            logger.info("New model trained and saved.")
        else:
            logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load or train model: {e}")
        logger.error(traceback.format_exc())
        exit(1)

    logger.info("Starting the Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)