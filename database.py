import sqlite3
import json

def init_db():
    conn = sqlite3.connect('sleep_app.db')
    c = conn.cursor()
    # 用户表 (新增 avatar 头像字段模拟)
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, 
                  role TEXT, is_banned INTEGER, intro TEXT, avatar TEXT)''')
    # 申请记录表 (新增 ai_json 用于存储预测出的阶段序列)
    c.execute('''CREATE TABLE IF NOT EXISTS applications 
                 (id INTEGER PRIMARY KEY, patient_id INTEGER, doctor_id INTEGER, 
                  hea_name TEXT, dat_content BLOB, status TEXT, 
                  ai_json TEXT, doctor_feedback TEXT, apply_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    # 举报表
    c.execute('''CREATE TABLE IF NOT EXISTS reports 
                 (id INTEGER PRIMARY KEY, reporter_id INTEGER, reported_id INTEGER, reason TEXT, status TEXT)''')
    # 收件箱
    c.execute('''CREATE TABLE IF NOT EXISTS messages 
                 (id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT, is_read INTEGER DEFAULT 0)''')

    # 初始化账号 (增加头像 Emoji 模拟)
    users = [
        (1, 'admin', 'admin123', 'admin', 0, '系统管理员', '🛠️'),
        (2, '王医生', '123', 'doctor', 0, '精通脑电波分析，10年睡眠障碍诊疗经验。', '👨‍⚕️'),
        (3, '李医生', '123', 'doctor', 0, '擅长REM期失眠诊断，副主任医师。', '👩‍⚕️'),
        (4, '张医生', '123', 'doctor', 0, '专注于青少年发育期睡眠研究。', '👨‍⚕️'),
        (5, 'patient1', '123', 'patient', 0, '普通受试者', '👤')
    ]
    c.executemany("INSERT OR IGNORE INTO users VALUES (?,?,?,?,?,?,?)", users)
    conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect('sleep_app.db')