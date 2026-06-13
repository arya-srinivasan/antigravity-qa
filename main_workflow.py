import os
import uuid
from database.db import get_questions
import asyncio
from intake_agent import run_managed_memory_session
from rag_agent import run_rag_agent, handle_transcript_chunk, end_meeting
from question_classifier_agent import handle_follow_up_question

zoom_meeting_active = False

async def main():
    global zoom_meeting_active
    print("Zoom not started yet. Running intake agent to collect questions until Zoom meeting is active...")

    while not zoom_meeting_active:
        result = await run_managed_memory_session()
        if result == "start zoom":
            zoom_meeting_active = True
            print("\nZoom meeting started. Switching to real-time processing...")
            break
        if result is None:
            continue

    print("Zoom meeting is active. Starting main workflow to process questions in real-time...")
    pre_meeting_questions = get_questions(status="unanswered")
    await asyncio.gather(*[run_rag_agent(q["question"], q["context"]) for q in pre_meeting_questions])

    while True:
        await handle_follow_up_question()


async def wait_for_zoom_and_run():
    print("\n=== Loading static transcript into Pinecone ===")
    transcript = [
        ("Professor Smith", "Today we are going to talk about neural networks and how they learn."),
        ("Professor Smith", "The key concept here is backpropagation which adjusts weights during training."),
        ("Professor Smith", "Gradient descent is the optimization algorithm we use to minimize the loss function."),
        ("Professor Smith", "Let me show you how the chain rule applies to multi-layer networks."),
    ]
    for speaker, text in transcript:
        await handle_transcript_chunk(speaker, text)
    await end_meeting()
    await asyncio.sleep(2)
    print("Transcript loaded.\n")

    await main()


if __name__ == "__main__":
    asyncio.run(wait_for_zoom_and_run())




