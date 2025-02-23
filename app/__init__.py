from flask import Flask
from flask_cors import CORS
from config import config
from app.api import init_api
import os


def create_app(config_name=None):
  if config_name is None:
    config_name = os.getenv('FLASK_ENV', 'default')

  app = Flask(__name__)
  app.config.from_object(config[config_name])
  CORS(app)

  # 설정 초기화
  config[config_name].init_app(app)

  # Blueprint 등록
  init_api(app)

  return app