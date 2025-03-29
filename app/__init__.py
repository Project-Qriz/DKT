from flask import Flask
from flask_cors import CORS
from config import config
from app.api import init_api
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app(config_name=None):
  if config_name is None:
    config_name = os.getenv('FLASK_ENV', 'local')

  app = Flask(__name__)

  # 설정 초기화
  config[config_name].init_app(app)
  config_instance = config[config_name]()
  config_instance.init_app(app)

  logger.info(f"Config: {config_name}")
  logger.info(f"DB_HOST: {app.config['DB_HOST']}")
  logger.info(f"PORT: {app.config['PORT']}")
  logger.info(f"HOST: {app.config['HOST']}")
  logger.info(f"DATABASE: {app.config['DB_DATABASE']}")

  CORS(app)

  # Blueprint 등록
  init_api(app)

  return app