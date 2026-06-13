import sqlite3
import os

os.makedirs("database", exist_ok=True)

def create_table():
    conn = sqlite3.connect("database/questions.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        question TEXT,
        context TEXT,
        status TEXT DEFAULT 'unanswered',
        type TEXT,
        topic TEXT           
    )
    """)
    conn.commit()
    conn.close()

create_table()  # always runs on import

def add_question(conversation_id, question, context, topic, type):
    conn = sqlite3.connect("database/questions.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO questions (conversation_id, question, context, type, topic) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, question, context, type, topic)
    )
    conn.commit()
    conn.close()

def get_questions(status="unanswered"):
    conn = sqlite3.connect("database/questions.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT conversation_id, question, context, topic, type FROM questions WHERE status = ?",
        (status,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"conversation_id": r[0], "question": r[1], "context": r[2], "topic": r[3], "type": r[4]} for r in rows]

def get_conversation_context(conversation_id, question):
    conn = sqlite3.connect("database/questions.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT context FROM questions WHERE conversation_id = ? AND question = ?",
        (conversation_id, question)
    )
    context = cursor.fetchone()
    conn.close()
    return context[0] if context else None

def mark_question_answered(question):
    conn = sqlite3.connect("database/questions.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE questions SET status = 'answered' WHERE question = ?",
        (question,)
    )
    conn.commit()
    conn.close()