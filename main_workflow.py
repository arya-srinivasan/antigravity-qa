import os
import uuid
from database.db import get_questions
import asyncio
from intake_agent import run_managed_memory_session
from rag_agent import run_rag_agent
from question_classifier_agent import handle_follow_up_question
import time

zoom_meeting_active = False


async def main():
    global zoom_meeting_active
    print("Zoom not started yet. Running intake agent to collect questions until Zoom meeting is active...")
    while not zoom_meeting_active:
        result = await run_managed_memory_session()
        if result is None:
            print("Exiting session.")
            break

    print("Zoom meeting is active. Starting main workflow to process questions in real-time...")
    pre_meeting_questions = get_questions(status="pending")
    pre_meeting_tasks = [run_rag_agent(q["question"], q["context"]) for q in pre_meeting_questions]
    await asyncio.gather(*pre_meeting_tasks)

    while True:
        await handle_follow_up_question()


async def wait_for_zoom_and_run():
    global zoom_meeting_active

    main_task = asyncio.create_task(main())
    await asyncio.get_event_loop().run_in_executor(None, input, "\nPress Enter once the Zoom meeting has started... ")
    zoom_meeting_active = True

    await main_task



if __name__ == "__main__":
    asyncio.run(wait_for_zoom_and_run())






