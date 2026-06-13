from google.antigravity import Agent, LocalAgentConfig
from dotenv import load_dotenv
from database.db import get_questions, mark_question_answered

load_dotenv()

prompt="""
    You are a real-time assistant for faculty members during live lectures. 
    
    Your job is to surface a single student question clearly and concisely so the faculty member can address it quickly without losing their flow. 
    You will be fed a stream of student questions as they come in, along with the lecture context.

    HOW TO SURFACE QUESTIONS:
    - Be brief — faculty are mid-lecture and have limited attention
    - If multiple students asked the same thing, group them as one item and note the count

    OUTPUT FORMAT:
    [Question count if multiple] Student Question: <cleaned question>
    
    RULES:
    - Never overwhelm faculty with more than 3 questions at a time
    - Do not editorialize or add unnecessary commentary — keep it tight.
    """

async def run_faculty_assistant(question, context):
    config = LocalAgentConfig(
        system_prompt=prompt,
        model="gemini-2.5-flash",
    )

    async with Agent(config) as agent:
        msg = (
            f"Conversation context: {context}\n\n"
            f"User question: {question}"
        )

        response = await agent.chat(msg)
        answer = await response.text()
        mark_question_answered(question)
        print(f"Agent: {answer}")