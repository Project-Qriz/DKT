import pymysql
from flask import current_app
import logging

logger = logging.getLogger(__name__)

def connect_db():
    try:
        return pymysql.connect(
            host=current_app.config['DB_HOST'],
            user=current_app.config['DB_USER'],
            password=current_app.config['DB_PASSWORD'],
            db=current_app.config['DB_DATABASE'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def update_predictions(user_id, skill_id, predict_accuracy):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            sql = """
            UPDATE skill_level
            SET predict_accuracy = %s, last_updated = CURRENT_TIMESTAMP
            WHERE user_id = %s AND skill_id = %s
            """
            cursor.execute(sql, (predict_accuracy, user_id, skill_id))
            conn.commit()
            logger.debug(f"Updated prediction for user {user_id}, skill {skill_id}")
    except Exception as e:
        logger.error(f"Failed to update prediction: {e}")
        raise
    finally:
        conn.close()