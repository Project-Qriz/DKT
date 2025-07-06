from app.models.dkt import load_activities, UserActivityDataset, predict, load_model_v2

class ModelService:
    def __init__(self):
        # (1) load_model_v2 로 모델과 num_q 함께 불러오기
        self.model, self.num_q = load_model_v2('dkt_model.pth')

        # (2) DB 전체 활동에서 question_id → skill_id_idx 맵 생성
        acts = load_activities()
        self.qid2idx = {a['question_id']: a['skill_id_idx'] for a in acts}

    def get_prediction(self, data):
        user_id = data['user_id']
        activities = data['activities']

        # (3) 유효성 검사 및 skill_id_idx 주입
        for i, act in enumerate(activities):
            if not isinstance(act, dict) \
               or 'question_id' not in act \
               or ('correction' not in act and 'correct' not in act) \
               or 'time_spent' not in act:
                raise ValueError(f"Invalid activity at index {i}: {act!r}")

            qid = act['question_id']
            if qid not in self.qid2idx:
                raise ValueError(f"No mapping for question_id={qid}")
            act['skill_id_idx'] = self.qid2idx[qid]

        # (4) Dataset 생성 → predict 호출
        ds = UserActivityDataset(activities, user_id, max_seq_len=100, num_q=self.num_q)
        preds = predict(self.model, ds, self.num_q, user_id)

        # (5) JSON 직렬화 후 반환
        return {
            'user_id': user_id,
            'predictions': {str(uid): v.tolist() for uid, v in preds.items()}
        }