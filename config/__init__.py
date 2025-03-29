import os
import logging
import pathlib

from dotenv import load_dotenv
from .default import Config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리 경로 구하기
ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()


class DevelopmentConfig(Config):
    DEBUG = True

    def __init__(self):
        env_path = os.path.join(ROOT_DIR, 'env', '.dev.env')
        logger.debug(f'Loading env from: {env_path}')
        load_dotenv(env_path)

        logger.debug('DevelopmentConfig initialized')
        self.DB_HOST = os.getenv('DB_HOST')
        self.DB_DATABASE = os.getenv('DB_DATABASE')
        self.DB_USER = os.getenv('DB_USER')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.HOST = os.getenv('HOST', 'localhost')
        self.PORT = os.getenv('PORT')


class LocalConfig(Config):
    DEBUG = True

    def __init__(self):
        env_path = os.path.join(ROOT_DIR, 'env', '.env')
        logger.debug(f'Loading env from: {env_path}')
        load_dotenv(env_path)

        logger.debug('LocalConfig initialized')
        self.DB_HOST = os.getenv('DB_HOST')
        self.DB_DATABASE = os.getenv('DB_DATABASE')
        self.DB_USER = os.getenv('DB_USER')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.HOST = os.getenv('HOST', 'localhost')
        self.PORT = os.getenv('PORT')

config = {
    'dev': DevelopmentConfig(),
    'local': LocalConfig(),
    'default': LocalConfig()
}