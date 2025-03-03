class Config:
  # 공통 설정
  MODEL_PATH = 'dkt_model.pth'
  MAX_SEQ_LEN = 100
  DEBUG = False

  # 환경변수에서 가져올 설정
  DB_HOST = None
  DB_DATABASE = None
  DB_USER = None
  DB_PASSWORD = None
  HOST = None
  PORT = None

  @staticmethod
  def init_app(app):
    pass