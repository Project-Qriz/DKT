import logging
import traceback
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.dkt import UserActivityDataset, load_activities, DKTModel, train, save_model
from torch.utils.data import DataLoader
import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train_new_model():
  try:
    activities = load_activities()
    num_q = len(set(activity['skill_id'] for activity in activities))
    logger.info(f"Number of questions: {num_q}")

    train_dataset = UserActivityDataset(activities, max_seq_len=100, num_q=num_q)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = DKTModel(num_q=num_q, emb_size=128, hidden_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger.info("Starting model training...")
    train(model, train_loader, optimizer)

    model_path = '../dkt_model.pth'
    save_model(model, num_q, model_path)
    logger.info("Model training completed and saved.")

    return model, num_q

  except Exception as e:
    logger.error(f"Failed to train model: {e}")
    logger.error(traceback.format_exc())
    raise


if __name__ == '__main__':
  train_new_model()