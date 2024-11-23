import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import torch.nn.functional as F
import pymysql
from db_update import update_predictions
from datetime import datetime
import os
import numpy as np

# 데이터베이스 연결 함수
def connect_db():
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='1234',
            db='qriz',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except Exception as e:
        print(f"Database connection failed due to {e}")
        return None

# 데이터 로드 함수
def load_activities():
    conn = connect_db()
    if conn is not None:
        try:
            with conn.cursor() as cursor:
                sql = """
                SELECT ua.*, activity_id, ua.user_id, ua.question_id, ua.checked, ua.time_spent, 
                       q.skill_id, q.difficulty, q.answer
                FROM user_activity ua
                JOIN question q ON ua.question_id = q.question_id
                WHERE q.category = 2
                """
                cursor.execute(sql)
                activities = cursor.fetchall()
                return activities
        except Exception as e:
            print(f"Failed to load data due to {e}")
        finally:
            conn.close()
    return []

# DKT 모델 정의
class DKTModel(Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKTModel, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.interaction_emb = Embedding(num_q * 2 + 1, emb_size)  # +1 for the padding index
        self.time_emb = Linear(1, emb_size)
        self.lstm = LSTM(emb_size, hidden_size, batch_first=True)
        self.fc = Linear(hidden_size, num_q)
        self.dropout = Dropout(0.5)

    def forward(self, questions, responses, times_spent):
        interactions = questions + self.num_q * responses
        x = self.interaction_emb(interactions)
        time_emb = self.time_emb(times_spent.unsqueeze(-1))
        x = x + time_emb
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.dropout(x)
        y = torch.sigmoid(x)
        return y

# 사용자 활동 데이터셋 클래스
class UserActivityDataset(Dataset):
    def __init__(self, activities, user_id, max_seq_len, num_q):
        self.activities = activities
        self.max_seq_len = max_seq_len
        self.num_q = num_q
        self.user_id = user_id  # user_id를 인스턴스 변수로 저장

    def __len__(self):
        return 1  # 단일 사용자 데이터셋이므로 길이는 1

    def __getitem__(self, idx):
        user_activities = self.activities  # 이미 단일 사용자의 활동만 포함되어 있다고 가정
        
        q_ids = [activity['question_id'] for activity in user_activities]
        responses = [1 if activity['correct'] == 1 else 0 for activity in user_activities]
        times_spent = [activity['time_spent'] for activity in user_activities]
        
        seq_len = len(q_ids)

        # Ensure indices are within the embedding range
        q_ids = [min(q_id, self.num_q - 1) for q_id in q_ids]

        # Padding 처리
        if seq_len < self.max_seq_len:
            q_ids += [self.num_q * 2] * (self.max_seq_len - seq_len)  # Use out-of-bound index for padding
            responses += [0] * (self.max_seq_len - seq_len)
            times_spent += [0] * (self.max_seq_len - seq_len)
        else:
            q_ids = q_ids[:self.max_seq_len]
            responses = responses[:self.max_seq_len]
            times_spent = times_spent[:self.max_seq_len]

        return torch.tensor(q_ids, dtype=torch.long), torch.tensor(responses, dtype=torch.long), torch.tensor(times_spent, dtype=torch.float), self.user_id

# 모델 훈련 함수
def train_model(train_loader, num_q, num_epochs=10):
    model = DKTModel(num_q=num_q, emb_size=128, hidden_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for questions, responses, times_spent, _ in train_loader:
            optimizer.zero_grad()
            predictions = model(questions, responses, times_spent)
            predictions = predictions.view(-1, predictions.size(-1))
            responses = responses.view(-1)
            predictions = predictions.gather(1, responses.view(-1, 1)).squeeze(1)
            loss = F.binary_cross_entropy(predictions, responses.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')
    
    return model

# 모델 훈련 함수
def train(model, train_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for questions, responses, times_spent, _ in train_loader:
            optimizer.zero_grad()
            predictions = model(questions, responses, times_spent)
            predictions = predictions.view(-1, predictions.size(-1))  # (batch_size * seq_len, num_q)
            responses = responses.view(-1)  # (batch_size * seq_len)
            predictions = predictions.gather(1, responses.view(-1, 1)).squeeze(1)  # gather predictions for the actual questions
            loss = F.binary_cross_entropy(predictions, responses.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')


# 예측 함수
def predict(model, dataset, num_q, user_id):
    model.eval()
    predictions = {}
    with torch.no_grad():
        questions, responses, times_spent, _ = dataset[0]  # 단일 사용자 데이터셋이므로 인덱스 0
        output = model(questions.unsqueeze(0), responses.unsqueeze(0), times_spent.unsqueeze(0))
        preds = output.squeeze(0).detach().numpy()
        predictions[user_id] = preds
    return predictions

# 모델 저장 함수
def save_model(model, num_q, model_path='dkt_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_q': num_q
    }, model_path)
    print(f"Model saved with num_q = {num_q}")

# 모델 로드 함수
def load_model(model_path, current_num_q):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        saved_num_q = checkpoint.get('num_q')
        
        if saved_num_q is None:
            print("The saved model doesn't have num_q information. Training new model...")
            return None
        
        if saved_num_q != current_num_q:
            print(f"Mismatch in num_q: saved={saved_num_q}, current={current_num_q}")
            print("Training new model...")
            return None
        
        model = DKTModel(num_q=current_num_q, emb_size=128, hidden_size=256)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded with num_q = {current_num_q}")
        return model
    else:
        print("No saved model found. Training new model...")
        return None

# 메인 실행 함수
def main():
    activities = load_activities()
    
    # 데이터 전처리 추가
    all_skill_ids = set(activity['skill_id'] for activity in activities)
    skill_id_map = {skill_id: i for i, skill_id in enumerate(sorted(all_skill_ids))}
    
    for activity in activities:
        activity['skill_id'] = skill_id_map[activity['skill_id']]
    
    num_q = len(skill_id_map)
    
    max_seq_len = 100
    train_dataset = UserActivityDataset(activities, max_seq_len, num_q)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model_path = 'dkt_model.pth'
    
    model = load_model(model_path, num_q)
    if model is None:
        model = DKTModel(num_q=num_q, emb_size=128, hidden_size=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train(model, train_loader, optimizer)
        save_model(model, num_q, model_path)
    
    # 예측 및 결과 업데이트
    predictions = predict(model, train_dataset, num_q)
    for user_id, predict_accuracy in predictions.items():
        user_activities = [activity for activity in activities if activity['user_id'] == user_id]
        skill_ids = set(activity['skill_id'] for activity in user_activities)

        for skill_id in skill_ids:
            difficulties = set(activity['difficulty'] for activity in user_activities if activity['skill_id'] == skill_id)
            
            for difficulty in difficulties:
                current_accuracy = calculate_current_accuracy(user_id, skill_id, difficulty, activities)
                skill_predict_accuracies = [predict_accuracy[activity['skill_id']] for activity in user_activities if activity['skill_id'] == skill_id and activity['difficulty'] == difficulty]
                
                if skill_predict_accuracies:
                    avg_predict_accuracy = np.mean(skill_predict_accuracies)
                    
                    try:
                        print(f'Updating user_id: {user_id}, predict_accuracy: {avg_predict_accuracy}, current_accuracy: {current_accuracy}, difficulty: {difficulty}, skill_id: {skill_id}')
                        update_skill_level(user_id, skill_id, avg_predict_accuracy, current_accuracy, difficulty)
                    except Exception as e:
                        print(f"Error processing skill_id {skill_id} for user_id {user_id}: {e}")

    
if __name__ == "__main__":
    main()

# difficulty와 skill_id를 가져오는 함수
def get_question_difficulty_and_skill_id(question_id):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT difficulty, skill_id FROM question WHERE question_id = %s"
            cursor.execute(sql, (question_id,))
            result = cursor.fetchone()
            if result:
                return result['difficulty'], result['skill_id']
            else:
                return None, None
    except Exception as e:
        print(f"Failed to get question difficulty and skill_id due to {e}")
        return None, None
    finally:
        conn.close()

# SkillLevel 업데이트 함수
def update_skill_level(user_id, skill_id, predict_accuracy, current_accuracy, difficulty):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            # 먼저 해당 user_id와 skill_id에 대한 행이 존재하는지 확인합니다.
            cursor.execute("SELECT 1 FROM skill WHERE skill_id = %s", (skill_id,))
            if cursor.fetchone() is None:
                print(f"Skill ID {skill_id} does not exist. Skipping update.")
                return
            
            cursor.execute("SELECT 1 FROM skill_level WHERE user_id = %s AND skill_id = %s AND difficulty = %s", (user_id, skill_id, difficulty))
            exists = cursor.fetchone()
            
            if exists:
                sql = """
                UPDATE skill_level
                SET predict_accuracy = %s, current_accuracy = %s, last_updated = %s
                WHERE user_id = %s AND skill_id = %s AND difficulty = %s
                """
                cursor.execute(sql, (predict_accuracy, current_accuracy, datetime.now(), user_id, skill_id, difficulty))
            else:
                sql = """
                INSERT INTO skill_level (user_id, skill_id, predict_accuracy, current_accuracy, difficulty, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (user_id, skill_id, predict_accuracy, current_accuracy, difficulty, datetime.now()))
            
            conn.commit()
    except Exception as e:
        print(f"Failed to update skill level due to {e}")
    finally:
        conn.close()

# correction 값 변환 함수
def correction_to_int(correction):
    return 1 if correction == b'\x01' else 0

# current_accuracy 계산 함수
def calculate_current_accuracy(user_id, skill_id, difficulty, activities):
    user_activities = [activity for activity in activities if activity['user_id'] == user_id and activity['skill_id'] == skill_id and activity['difficulty'] == difficulty]
    print(f"User {user_id}, Skill {skill_id}, Difficulty {difficulty}: Total activities = {len(user_activities)}")
    if len(user_activities) == 0:
        return 0.0
    correct_answers = sum(correction_to_int(activity['correction']) for activity in user_activities)  # correction 값을 변환하여 합산
    total_activities = len(user_activities)
    accuracy = correct_answers / total_activities if total_activities > 0 else 0.0
    print(f"User {user_id}, Skill {skill_id}, Difficulty {difficulty}: Correct answers = {correct_answers}, Accuracy = {accuracy}")
    return accuracy