#!/usr/bin/env python3
"""
DKT 모델 학습 스크립트
- bastion 호스트에서 실행
- RDS에서 데이터를 가져와 모델 학습
- 학습된 모델을 S3에 저장
"""

import os
import sys
import torch
import pymysql
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
import logging
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  handlers=[
    logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    logging.StreamHandler(sys.stdout)
  ]
)
logger = logging.getLogger(__name__)

# 스크립트의 기본 디렉터리를 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# S3 클라이언트 초기화
s3_client = boto3.client('s3')
MODEL_BUCKET = "qriz-model-data"
DATA_BUCKET = "qriz-data-storage"
LOGS_BUCKET = "qriz-training-logs"

# 데이터베이스 연결 정보
DB_CONFIG = {
  'host': 'dev-mysql.c36q2sk4sbrv.ap-northeast-2.rds.amazonaws.com',  # RDS 엔드포인트
  'user': 'qriz',
  'password': 'qriz1234!!',
  'db': 'qriz',
  'charset': 'utf8mb4',
}


# 모델 정의 (from app/models/dkt.py)
class DKTModel(torch.nn.Module):
  def __init__(self, num_q, emb_size, hidden_size):
    super(DKTModel, self).__init__()
    self.num_q = num_q
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.interaction_emb = torch.nn.Embedding(num_q * 2 + 1, emb_size)  # +1 for padding
    self.time_emb = torch.nn.Linear(1, emb_size)
    self.lstm = torch.nn.LSTM(emb_size, hidden_size, batch_first=True)
    self.fc = torch.nn.Linear(hidden_size, num_q)
    self.dropout = torch.nn.Dropout(0.5)

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


# 데이터셋 클래스
class UserActivityDataset(torch.utils.data.Dataset):
  def __init__(self, activities, max_seq_len, num_q):
    self.activities = activities
    self.max_seq_len = max_seq_len
    self.num_q = num_q

    # 사용자별로 활동 그룹화
    self.user_activities = {}
    for activity in activities:
      user_id = activity['user_id']
      if user_id not in self.user_activities:
        self.user_activities[user_id] = []
      self.user_activities[user_id].append(activity)

    # 사용자 ID 목록
    self.user_ids = list(self.user_activities.keys())

  def __len__(self):
    return len(self.user_ids)

  def __getitem__(self, idx):
    user_id = self.user_ids[idx]
    user_activities = self.user_activities[user_id]

    # 각 사용자의 활동을 시간순으로 정렬
    user_activities.sort(key=lambda x: x['created_at'])

    q_ids = [activity['skill_id'] for activity in user_activities]
    responses = [1 if activity['correct'] == 1 else 0 for activity in user_activities]
    times_spent = [activity['time_spent'] for activity in user_activities]

    seq_len = len(q_ids)

    # 인덱스가 범위 내에 있는지 확인
    q_ids = [min(q_id, self.num_q - 1) for q_id in q_ids]

    # 패딩
    if seq_len < self.max_seq_len:
      q_ids += [self.num_q * 2] * (self.max_seq_len - seq_len)
      responses += [0] * (self.max_seq_len - seq_len)
      times_spent += [0] * (self.max_seq_len - seq_len)
    else:
      q_ids = q_ids[:self.max_seq_len]
      responses = responses[:self.max_seq_len]
      times_spent = times_spent[:self.max_seq_len]

    return (
      torch.tensor(q_ids, dtype=torch.long),
      torch.tensor(responses, dtype=torch.long),
      torch.tensor(times_spent, dtype=torch.float),
      user_id
    )


# 데이터베이스에서 활동 로드
def load_activities():
  try:
    logger.info("Connecting to database and loading activities...")
    conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)

    with conn.cursor() as cursor:
      sql = """
            SELECT ua.*, ua.user_id, ua.question_id, ua.checked as correct, ua.time_spent, 
                   q.skill_id, q.difficulty, q.answer, ua.created_at
            FROM user_activity ua
            JOIN question q ON ua.question_id = q.question_id
            WHERE q.category = 2
            """
      cursor.execute(sql)
      activities = cursor.fetchall()
      logger.info(f"Loaded {len(activities)} activities from database")

    conn.close()
    return activities
  except Exception as e:
    logger.error(f"Error loading activities: {str(e)}")
    return []


# 모델 학습 함수
def train_model(train_loader, num_q, num_epochs=10, learning_rate=0.001):
  logger.info(f"Training model with num_q={num_q}, epochs={num_epochs}, lr={learning_rate}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logger.info(f"Using device: {device}")

  model = DKTModel(num_q=num_q, emb_size=128, hidden_size=256).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = torch.nn.BCELoss()

  train_losses = []

  for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
      for questions, responses, times_spent, _ in pbar:
        questions = questions.to(device)
        responses = responses.to(device)
        times_spent = times_spent.to(device)

        optimizer.zero_grad()

        # 모델 예측
        output = model(questions, responses, times_spent)

        # 다음 문제에 대한 예측만 사용
        target_responses = responses.clone()
        target_responses[:, :-1] = target_responses[:, 1:].clone()
        target_responses[:, -1] = 0  # 마지막 응답은 패딩

        # 손실 계산
        loss = criterion(output.flatten(), target_responses.float().flatten())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # 진행 상황 업데이트
        pbar.set_postfix(loss=total_loss / batch_count)

    avg_loss = total_loss / batch_count
    train_losses.append(avg_loss)
    logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

  return model, train_losses


# 모델 검증
def validate_model(model, val_loader, device):
  logger.info("Validating model...")
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for questions, responses, times_spent, _ in val_loader:
      questions = questions.to(device)
      responses = responses.to(device)
      times_spent = times_spent.to(device)

      output = model(questions, responses, times_spent)

      # 다음 문제에 대한 예측 평가
      target_responses = responses.clone()
      target_responses[:, :-1] = target_responses[:, 1:].clone()

      # 예측값 이진화 (0.5 임계값)
      predictions = (output.flatten() > 0.5).float()

      # 정확도 계산
      correct += (predictions == target_responses.float().flatten()).sum().item()
      total += target_responses.numel()

  accuracy = correct / total if total > 0 else 0
  logger.info(f"Validation Accuracy: {accuracy:.4f}")

  return accuracy


# 모델 저장
def save_model(model, num_q, train_losses, accuracy):
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  local_path = os.path.join(MODEL_DIR, f"dkt_model_{timestamp}.pth")

  # 모델 저장
  torch.save({
    'model_state_dict': model.state_dict(),
    'num_q': num_q,
    'train_losses': train_losses,
    'accuracy': accuracy,
    'timestamp': timestamp
  }, local_path)

  logger.info(f"Model saved locally to {local_path}")

  # S3에 업로드
  try:
    # 모델 파일 업로드
    s3_model_key = f"models/{timestamp}/dkt_model.pth"
    s3_client.upload_file(local_path, MODEL_BUCKET, s3_model_key)
    logger.info(f"Model uploaded to s3://{MODEL_BUCKET}/{s3_model_key}")

    # latest 폴더에도 복사
    s3_client.copy_object(
      Bucket=MODEL_BUCKET,
      CopySource=f"{MODEL_BUCKET}/{s3_model_key}",
      Key="models/latest/dkt_model.pth"
    )
    logger.info(f"Model copied to s3://{MODEL_BUCKET}/models/latest/dkt_model.pth")

    # 학습 메타데이터 저장
    metadata = {
      'timestamp': timestamp,
      'num_q': num_q,
      'train_losses': train_losses,
      'accuracy': accuracy,
      'model_s3_path': f"s3://{MODEL_BUCKET}/{s3_model_key}"
    }

    metadata_json = json.dumps(metadata)
    s3_client.put_object(
      Bucket=LOGS_BUCKET,
      Key=f"training_metadata/{timestamp}.json",
      Body=metadata_json
    )
    logger.info(f"Training metadata saved to s3://{LOGS_BUCKET}/training_metadata/{timestamp}.json")

    # 학습 로그 업로드
    log_file = f"training_{timestamp}.log"
    if os.path.exists(log_file):
      s3_client.upload_file(log_file, LOGS_BUCKET, f"logs/{log_file}")
      logger.info(f"Training log uploaded to s3://{LOGS_BUCKET}/logs/{log_file}")

    return True
  except Exception as e:
    logger.error(f"Error uploading model to S3: {str(e)}")
    return False


# 메인 함수
def main():
  logger.info("Starting model training process...")

  try:
    # 활동 데이터 로드
    activities = load_activities()

    if not activities:
      logger.error("No activities found, aborting training")
      return

    # 스킬 ID 맵핑
    all_skill_ids = set(activity['skill_id'] for activity in activities)
    skill_id_map = {skill_id: i for i, skill_id in enumerate(sorted(all_skill_ids))}

    # 스킬 ID 변환
    for activity in activities:
      activity['skill_id'] = skill_id_map[activity['skill_id']]

    num_q = len(skill_id_map)
    logger.info(f"Found {num_q} unique skill IDs")

    # 데이터 분할 (학습:검증 = 8:2)
    np.random.seed(42)
    user_ids = list(set(activity['user_id'] for activity in activities))
    np.random.shuffle(user_ids)

    split_idx = int(len(user_ids) * 0.8)
    train_user_ids = set(user_ids[:split_idx])
    val_user_ids = set(user_ids[split_idx:])

    train_activities = [a for a in activities if a['user_id'] in train_user_ids]
    val_activities = [a for a in activities if a['user_id'] in val_user_ids]

    logger.info(f"Split data: {len(train_activities)} training samples, {len(val_activities)} validation samples")

    # 데이터셋 및 데이터 로더 생성
    max_seq_len = 100
    batch_size = 32

    train_dataset = UserActivityDataset(train_activities, max_seq_len, num_q)
    val_dataset = UserActivityDataset(val_activities, max_seq_len, num_q)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 모델 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_losses = train_model(train_loader, num_q, num_epochs=10)

    # 모델 검증
    accuracy = validate_model(model, val_loader, device)

    # 모델 저장
    save_success = save_model(model, num_q, train_losses, accuracy)

    if save_success:
      logger.info("Model training and saving completed successfully")
    else:
      logger.warning("Model training completed but there was an issue saving the model")

  except Exception as e:
    logger.error(f"Error in training process: {str(e)}", exc_info=True)


if __name__ == "__main__":
  main()