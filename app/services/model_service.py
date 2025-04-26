from app.models.dkt import load_model, load_model_v2, UserActivityDataset, predict


class ModelService:
    def __init__(self):
        self.model, self.num_q = self._initialize_model()

    def _initialize_model(self):
        # return load_model('dkt_model.pth')
        return load_model_v2('dkt_model.pth')

    def get_prediction(self, data):
        user_id = data['user_id']
        activities = data['activities']
        dataset = UserActivityDataset(activities, user_id, max_seq_len=100, num_q=self.num_q)
        predictions = predict(self.model, dataset, self.num_q, user_id)
        return {'user_id': user_id, 'predictions': predictions}