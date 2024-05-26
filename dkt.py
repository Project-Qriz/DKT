import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import torch.nn.functional as F
import pymysql
from db_update import update_predictions  # db_update 모듈에서 함수 임포트
from datetime import datetime

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
        print("Database connection failed due to {}".format(e))
        return None

# 데이터 로드 함수
def load_activities():
    conn = connect_db()
    if conn is not None:
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM user_activity"
                cursor.execute(sql)
                activities = cursor.fetchall()
                return activities
        except Exception as e:
            print("Failed to load data due to {}".format(e))
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
        self.lstm = LSTM(emb_size, hidden_size, batch_first=True)
        self.fc = Linear(hidden_size, num_q)
        self.dropout = Dropout(0.5)

    def forward(self, questions, responses):
        interactions = questions + self.num_q * responses
        x = self.interaction_emb(interactions)
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
        seq_len = len(q_ids)

        # Ensure indices are within the embedding range
        q_ids = [min(q_id, self.num_q - 1) for q_id in q_ids]

        # Padding 처리
        if seq_len < self.max_seq_len:
            q_ids += [self.num_q * 2] * (self.max_seq_len - seq_len)  # Use out-of-bound index for padding
            responses += [0] * (self.max_seq_len - seq_len)
        else:
            q_ids = q_ids[:self.max_seq_len]
            responses = responses[:self.max_seq_len]

        return torch.tensor(q_ids, dtype=torch.long), torch.tensor(responses, dtype=torch.long), user_id

# 모델 훈련 함수
def train(model, train_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for questions, responses, _ in train_loader:
            optimizer.zero_grad()
            predictions = model(questions, responses)
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
        for questions, responses, user_id in dataset:
            questions, responses = questions.unsqueeze(0), responses.unsqueeze(0)
            output = model(questions, responses)
            preds = output.squeeze(0).detach().numpy()
            predictions[user_id] = preds.mean(axis=0)
    return predictions

# difficulty 가져오는 함수
def get_question_difficulty(question_id):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT difficulty FROM question WHERE question_id = %s"
            cursor.execute(sql, (question_id,))
            result = cursor.fetchone()
            if result:
                return result['difficulty']
            else:
                return None
    except Exception as e:
        print(f"Failed to get question difficulty due to {e}")
        return None
    finally:
        conn.close()

# SkillLevel 업데이트 함수
def update_skill_level(user_id, skill_id, predict_accuracy, current_accuracy, difficulty):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            # 먼저 해당 user_id와 skill_id에 대한 행이 존재하는지 확인합니다.
            cursor.execute("SELECT 1 FROM skill_level WHERE user_id = %s AND skill_id = %s", (user_id, skill_id))
            exists = cursor.fetchone()
            
            if exists:
                sql = """
                UPDATE skill_level
                SET predict_accuracy = %s, current_accuracy = %s, difficulty = %s, last_updated = %s
                WHERE user_id = %s AND skill_id = %s
                """
                cursor.execute(sql, (predict_accuracy, current_accuracy, difficulty, datetime.now(), user_id, skill_id))
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
def calculate_current_accuracy(user_id, activities, num_recent=10):
    user_activities = [activity for activity in activities if activity['user_id'] == user_id]
    if len(user_activities) == 0:
        return 0.0
    # 최근 num_recent 개의 활동만 사용
    recent_activities = user_activities[-num_recent:]
    correct_answers = sum(1 if activity['correction'] else 0 for activity in recent_activities)  # 'correction' 값을 직접 int로 변환
    return correct_answers / len(recent_activities)

# 데이터 로드 및 모델 설정
activities = load_activities()
num_q = 100
max_seq_len = 100
train_dataset = UserActivityDataset(activities, max_seq_len, num_q)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = DKTModel(num_q=num_q, emb_size=128, hidden_size=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 훈련 시작
train(model, train_loader, optimizer)

# 예측 및 결과 업데이트
predictions = predict(model, train_dataset, num_q)
for user_id, predict_accuracy in predictions.items():
    current_accuracy = calculate_current_accuracy(user_id, activities)  # current_accuracy 계산
    for skill_id, accuracy in enumerate(predict_accuracy):
        # question_id를 기반으로 difficulty 가져오기
        difficulty = get_question_difficulty(skill_id + 1)  # skill_id는 0부터 시작하므로 1을 더합니다.
        print(f'Updating user_id: {user_id}, skill_id: {skill_id}, predict_accuracy: {accuracy}, current_accuracy: {current_accuracy}, difficulty: {difficulty}')
        update_skill_level(user_id, skill_id, accuracy, current_accuracy, difficulty)
