import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import torch.nn.functional as F
import pymysql
from db_update import update_predictions  # db_update 모듈에서 함수 임포트

# 데이터베이스 연결 함수
def connect_db():
    try:
        conn = pymysql.connect(
            host='mydb',
            user='mydb',
            password='testmydb1',
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
                sql = "SELECT * FROM UserActivity"
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
        self.interaction_emb = Embedding(num_q * 2, emb_size)
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
    def __init__(self, activities, max_seq_len):
        self.activities = activities
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.activities)

    def __getitem__(self, idx):
        user_data = self.activities[idx]
        q_ids = [activity['question_id'] for activity in user_data]
        responses = [activity['correction'] for activity in user_data]
        seq_len = len(q_ids)

        # 패딩 처리
        if seq_len < self.max_seq_len:
            q_ids += [0] * (self.max_seq_len - seq_len)
            responses += [0] * (self.max_seq_len - seq_len)

        return torch.tensor(q_ids, dtype=torch.long), torch.tensor(responses, dtype=torch.long)

# 모델 훈련 함수
def train(model, train_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for questions, responses in train_loader:
            optimizer.zero_grad()
            predictions = model(questions, responses[:, :-1])
            loss = F.binary_cross_entropy(predictions, responses[:, 1:].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')

# 데이터 로드 및 모델 설정
activities = load_activities()
train_dataset = UserActivityDataset(activities, max_seq_len=100)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = DKTModel(num_q=100, emb_size=128, hidden_size=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 훈련 시작
train(model, train_loader, optimizer)

# 예측 후 결과 업데이트
update_predictions(user_id, skill_id, predict_accuracy)
