from .health import health_bp
from .predict import predict_bp

def init_api(app):
    app.register_blueprint(health_bp)
    app.register_blueprint(predict_bp)