import os
import uuid
from dotenv import load_dotenv
from google.antigravity import Agent, LocalAgentConfig
from database.db import get_questions, mark_question_answered
import asyncio
from intake_agent import run_managed_memory_session
from rag_agent import rag_agent_main
from question_classifier_agent import handle_student_question


load_dotenv()

async def start_conversation():
    reply, question, context = await run_managed_memory_session()

    final_response = await rag_agent_main(question, context)


async def main_conversation_loop(conversation_id):
    reply, question, _ = await run_managed_memory_session()

    final_response = await handle_student_question(conversation_id=conversation_id, question=question)

async def main():
    conversation_id = uuid.uuid4()

    await start_conversation()

    while True:
        message = input("User: ").strip()

        if message.lower() == "quit":
            print("Session ended.")
            break

        await main_conversation_loop(conversation_id=conversation_id)


if __name__ == "__main__":
    asyncio.run(main())





