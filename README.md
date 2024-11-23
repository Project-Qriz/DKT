<div align="right">
  <a href="#korean">한글</a> | <a href="#english">English</a>
</div>

<h1 id="korean">Deep Knowledge Tracing (DKT) 시스템</h1>

학습자의 문제 풀이 기록을 기반으로 맞춤형 문제를 추천해주는 Deep Knowledge Tracing 시스템 구현체입니다. LSTM 기반의 딥러닝을 활용하여 학습자의 성취도를 예측하고 적절한 문제를 추천합니다.

## 🌟 주요 기능

- 학습자의 지식 상태 실시간 예측
- 개인화된 문제 난이도 추천
- MySQL 데이터베이스 연동으로 데이터 영속성 보장
- 타 시스템과의 쉬운 연동을 위한 Flask API 제공
- 다중 스킬 및 난이도 레벨 지원
- 문제 풀이 소요 시간을 활용한 정확한 예측

## 🛠 시스템 구조

시스템은 세 가지 주요 컴포넌트로 구성됩니다:

1. **DKT 모델** (`dkt.py`)
   - 지식 추적을 위한 LSTM 기반 신경망
   - 문제 및 응답 표현을 위한 임베딩 레이어
   - 풀이 시간 특성 통합
   - 모델 학습 및 예측 기능

2. **Flask API** (`app.py`)
   - 모델 상호작용을 위한 RESTful API
   - 실시간 예측 엔드포인트
   - 모델 로딩 및 초기화
   - 오류 처리 및 로깅

3. **데이터베이스 연동** (`db_update.py`)
   - MySQL 데이터베이스 연결 관리
   - 스킬 레벨 업데이트
   - 예측 결과 저장 및 조회

## 💻 설치 요구사항

```bash
- Python 3.7+
- PyTorch
- Flask
- PyMySQL
- NumPy
```

## ⚙️ 설치 방법

1. 저장소 클론:
```bash
git clone [repository-url]
cd dkt-system
```

2. 필요 패키지 설치:
```bash
pip install -r requirements.txt
```

3. `db_update.py`에서 데이터베이스 연결 설정:
```python
host='localhost'
user='root'
password='your-password'
db='qriz'
```

## 🚀 사용 방법

### 서버 실행

1. Flask 애플리케이션 실행:
```bash
python app.py
```

### 예측 요청하기

`/predict` 엔드포인트로 다음과 같은 JSON 구조의 POST 요청을 보냅니다:

```json
{
    "user_id": "user_123",
    "activities": [
        {
            "question_id": 1,
            "correct": 1,
            "time_spent": 120
        }
    ]
}
```

### 응답 형식

```json
{
    "user_id": "user_123",
    "predictions": [0.85, 0.76, 0.92, ...]
}
```

## 🔍 모델 상세

### DKT 모델 구조

```python
DKTModel(
    num_q=num_questions,
    emb_size=128,
    hidden_size=256
)
```

주요 구성요소:
- 문제/응답 임베딩 레이어
- 풀이 시간 특성 통합
- LSTM 레이어
- 드롭아웃 레이어 (0.5)
- 시그모이드 활성화가 있는 출력 레이어

### 학습 프로세스

모델은 다음과 같은 설정으로 학습됩니다:
- 이진 교차 엔트로피 손실 함수
- Adam 옵티마이저
- 학습률: 0.001
- 배치 크기: 32
- 시퀀스 길이: 100

## 📊 데이터베이스 스키마

### 사용자 활동 테이블
```sql
CREATE TABLE user_activity (
    activity_id INT PRIMARY KEY,
    user_id INT,
    question_id INT,
    correct BOOLEAN,
    time_spent INT,
    created_at TIMESTAMP
);
```

### 스킬 레벨 테이블
```sql
CREATE TABLE skill_level (
    user_id INT,
    skill_id INT,
    predict_accuracy FLOAT,
    current_accuracy FLOAT,
    difficulty INT,
    last_updated TIMESTAMP,
    PRIMARY KEY (user_id, skill_id, difficulty)
);
```

---

<h1 id="english">Deep Knowledge Tracing (DKT) System</h1>

[Previous English content remains the same...]
