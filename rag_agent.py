import os
import sys
import uuid
import time
import json
import asyncio
from typing import Dict, Any, Optional, List
import websockets
from pinecone import Pinecone
from google.antigravity import Agent, LocalAgentConfig
from dotenv import load_dotenv
from database.db import mark_question_answered

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("PINECONE_API_KEY variable missing")
    PINECONE_API_KEY = "dummy-key"

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "zoom-transcripts"
index = pc.Index(INDEX_NAME)

chunk_buffer: list[tuple[str, str, float]] = []
TARGET_WORD_COUNT = 100
OVERLAP_WORDS = 20

async def get_embedding_passage(text: str) -> list[float]:
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[text],
            parameters={
                "input_type": "passage",
                "dimension": 2048
            }
        )
    )
    return response.data[0].values

async def get_embedding_query(text: str) -> list[float]:
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[text],
            parameters={
                "input_type": "query",
                "dimension": 2048
            }
        )
    )
    return response.data[0].values

def add_words_to_buffer(speaker: str, text: str) -> Optional[dict]:
    global chunk_buffer
    now = time.time()
    for word in text.split():
        chunk_buffer.append((speaker, word, now))
        if len(chunk_buffer) >= TARGET_WORD_COUNT:
            return flush_and_slide()
    return None

def flush_and_slide() -> Optional[dict]:
    global chunk_buffer
    if not chunk_buffer:
        return None

    full_text = " ".join(item[1] for item in chunk_buffer)
    speakers = list(set(item[0] for item in chunk_buffer))
    start_time = chunk_buffer[0][2]
    end_time = chunk_buffer[-1][2]

    chunk = {
        "text": full_text,
        "metadata": {
            "speakers": speakers,
            "start_time": start_time,
            "end_time": end_time,
            "word_count": len(chunk_buffer)
        }
    }

    chunk_buffer = chunk_buffer[-OVERLAP_WORDS:]
    return chunk


async def write_chunk_to_pinecone(chunk: dict):
    try: 
        print("Starting chunking pipeline")
        vector = await get_embedding_passage(chunk["text"])

        record_id = f"zoom_chunk{str(uuid.uuid4())}"
        metadata = {
            "text": chunk["text"],
            "speakers": ", ".join(chunk["metadata"]["speakers"]),
            "start_time": chunk["metadata"]["start_time"],
            "end_time": chunk["metadata"]["end_time"]
        }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: index.upsert(
            vectors=[{"id": record_id, "values": vector, "metadata": metadata}]
        ))
        print(f"Document chunk {record_id} successfully synced to Pinecone")
    except Exception as e:
        print(f"Failed to upsert transcript chunk to Pinecone: {e}")


async def handle_transcript_chunk(speaker: str, text: str):
    chunk = add_words_to_buffer(speaker, text)
    if chunk:
        await write_chunk_to_pinecone(chunk)

async def end_meeting():
    chunk = flush_and_slide()
    if chunk:
        await write_chunk_to_pinecone(chunk)

async def query_pinecone(query: str) -> str:
    query_vector = await get_embedding_query(query)
    try:
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

        matched_docs = []
        for match in results.get("matches", []):
            if "text" in match.get("metadata", {}):
                matched_docs.append(match["metadata"]["text"])

        if matched_docs:
            return "\n---\n".join(matched_docs)
        return "No matching docs in pinecone"
    except Exception as e:
        return f"Error executing index query: {str(e)}"

async def run_rag_agent(question, context):
    
    config = LocalAgentConfig(
        system_instructions= (
            "Your job is to answer questions using the live meeting transcript chunks." 
            "Synthesize specific, accurate answers utilizing the context and provided Pinecone tool."
            "Mention certain timestamps whenever possible."
        ),
        model="gemini-2.5-flash",
        tools=[query_pinecone]
    )

    async with Agent(config) as agent:
        composed_msg = (
            f"Conversation context: {context}\n\n"
            f"User's question: {question}"
        )
        response = await agent.chat(composed_msg)
        answer = await response.text()
        mark_question_answered(question)
        print(f"Agent: {answer}")

if __name__ == "__main__":
    question = "What is gradient descent"
    context = "Student is confused about concepts from today's lecture"
    asyncio.run(run_rag_agent(question, context))
