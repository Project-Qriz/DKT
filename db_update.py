import pymysql

def connect_db():
    return pymysql.connect(
            host='localhost',
            user='root',
            password='1234',
            db='qriz',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

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
    except Exception as e:
        print(f"Failed to update prediction due to {e}")
    finally:
        conn.close()
