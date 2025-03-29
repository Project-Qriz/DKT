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

  def init_app(cls, app):
    # 이 메서드에서 환경 변수를 로드하고 설정을 적용
    app.config['DB_HOST'] = cls.DB_HOST
    app.config['DB_DATABASE'] = cls.DB_DATABASE
    app.config['DB_USER'] = cls.DB_USER
    app.config['DB_PASSWORD'] = cls.DB_PASSWORD
    app.config['HOST'] = cls.HOST or 'localhost'
    app.config['PORT'] = cls.PORT
    app.config['DEBUG'] = cls.DEBUG
    app.config['MODEL_PATH'] = cls.MODEL_PATH
    app.config['MAX_SEQ_LEN'] = cls.MAX_SEQ_LEN
