import os
from dotenv import load_dotenv
from .default import Config

class DevelopmentConfig(Config):
    DEBUG = True

    def __init__(self):
        load_dotenv('./env/.dev.env')
        self.DB_HOST = os.getenv('DB_HOST')
        self.DB_DATABASE = os.getenv('DB_DATABASE')
        self.DB_USER = os.getenv('DB_USER')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.HOST = os.getenv('HOST', 'localhost')
        self.PORT = os.getenv('PORT')

class LocalConfig(Config):
    DEBUG = True

    def __init__(self):
        load_dotenv('./env/.env')
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