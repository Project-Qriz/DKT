import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import torch.nn.functional as F
import pymysql
from db_update import update_predictions  # db_update 모듈에서 함수 임포트
from datetime import datetime
import os
import numpy as np  # numpy 모듈 임포트

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
                # category가 2인 question_id만 선택
                sql = """
                SELECT ua.*, q.skill_id, q.difficulty
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
    def __init__(self, activities, max_seq_len, num_q):
        self.activities = activities
        self.max_seq_len = max_seq_len
        self.num_q = num_q
        self.user_ids = list(set(activity['user_id'] for activity in activities))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        user_activities = [activity for activity in self.activities if activity['user_id'] == user_id]
        
        q_ids = [activity['question_id'] for activity in user_activities]
        responses = [1 if activity['correction'] else 0 for activity in user_activities]
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

        return torch.tensor(q_ids, dtype=torch.long), torch.tensor(responses, dtype=torch.long), torch.tensor(times_spent, dtype=torch.float), user_id

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
def predict(model, dataset, num_q):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for questions, responses, times_spent, user_id in DataLoader(dataset, batch_size=1):
            output = model(questions, responses, times_spent)
            preds = output.squeeze(0).detach().numpy()
            predictions[user_id.item()] = preds  # user_id를 스칼라로 변환하여 사용
    return predictions

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

# current_accuracy 계산 함수
def calculate_current_accuracy(user_id, skill_id, difficulty, activities):
    user_activities = [activity for activity in activities if activity['user_id'] == user_id and activity['skill_id'] == skill_id and activity['difficulty'] == difficulty]
    if len(user_activities) == 0:
        return 0.0
    correct_answers = sum(1 if activity['correction'] else 0 for activity in user_activities)
    return correct_answers / len(user_activities)

# 데이터 로드 및 모델 설정
activities = load_activities()
num_q = 100
max_seq_len = 100
train_dataset = UserActivityDataset(activities, max_seq_len, num_q)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = DKTModel(num_q=num_q, emb_size=128, hidden_size=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 저장 경로
model_path = 'dkt_model.pth'

# 모델이 저장되어 있는 경우 로드
if (os.path.exists(model_path)):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from disk.")
else:
    # 모델 훈련 시작
    train(model, train_loader, optimizer)
    # 훈련된 모델 저장
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved to disk.")

# 예측 및 결과 업데이트
predictions = predict(model, train_dataset, num_q)
for user_id, predict_accuracy in predictions.items():
    user_activities = [activity for activity in activities if activity['user_id'] == user_id]
    skill_ids = set(activity['skill_id'] for activity in user_activities)  # 사용자가 푼 문제의 skill_id 집합

    for skill_id in skill_ids:
        difficulties = set(activity['difficulty'] for activity in user_activities if activity['skill_id'] == skill_id)
        
        for difficulty in difficulties:
            current_accuracy = calculate_current_accuracy(user_id, skill_id, difficulty, activities)  # current_accuracy 계산
            skill_predict_accuracies = [predict_accuracy[i] for i, activity in enumerate(user_activities) if activity['skill_id'] == skill_id and activity['difficulty'] == difficulty]
            
            if skill_predict_accuracies:
                avg_predict_accuracy = np.mean(skill_predict_accuracies)
                question_ids = [activity['question_id'] for activity in user_activities if activity['skill_id'] == skill_id and activity['difficulty'] == difficulty]

                for question_id in question_ids:
                    try:
                        print(f'Updating user_id: {user_id}, question_id: {question_id}, predict_accuracy: {avg_predict_accuracy}, current_accuracy: {current_accuracy}, difficulty: {difficulty}, skill_id: {skill_id}')
                        update_skill_level(user_id, skill_id, avg_predict_accuracy, current_accuracy, difficulty)
                    except Exception as e:
                        print(f"Error processing question_id {question_id} for user_id {user_id}: {e}")
