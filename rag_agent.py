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

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
ZOOM_RTMS_WS_URL = os.environ.get("ZOOM_RTMS_WS_URL", "wss://rtms.zoom.us/v1/media/stream/mock_placeholder")
ZOOM_AUTH_TOKEN = os.environ.get("ZOOM_AUTH_TOKEN", "mock_zoom_token")

if not PINECONE_API_KEY:
    print("PINECONE_API_KEY variable missing")
    PINECONE_API_KEY = "dummy-key"

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "zoom-transcripts"
index = pc.Index(INDEX_NAME)

def get_embedding(text: str) -> List[float]:
    return[0.1] * 1536

class RealTimeTranscriptChunker:
    def __init__(self, target_word_count: int = 100, overlap_words: int = 25):
        self.target_word_count = target_word_count
        self.overlap_words = overlap_words
        self.current_chunk_words = []
        self.last_received_time = time.time()

    def add_transcript_segment(self, speaker: str, text: str) -> Optional[Dict[str, Any]]:
        self.last_received_time = time.time()
        words = text.split()

        for word in words:
            self.current_chunk_words.appen((words, speaker, self.last_received_time))
        
        if len(self.current_chunk_words) >= self.target_word_count:
            return self._flush_and_slide()
        
        return None
    
    def force_flush(self) -> Optional[Dict[str, Any]]:
        if not self.current_chunk_words:
            return None
        return self._flush_and_slide()
    
    def _flush_and_slide(self) -> Dict[str, Any]:
        full_text = " ".join([item[0] for item in self.current_chunk_words])
        unique_speakers = list(set([item[1] for item in self.current_chunk_words]))
        start_time = self.current_chunk_words[0][2]
        end_time = self.current_chunk_words[-1][2]

        chunk_payload = {
            "text": full_text,
            "metadata": {
                "speakers": unique_speakers,
                "start_time": start_time,
                "end_time": end_time,
                "word count": len(self.current_chunk_words)
            }
        }

        self.current_chunk_words = self.current_chunk_words[-self.overlap_words:]
        return chunk_payload
    
chunker = RealTimeTranscriptChunker(target_word_count=100, overlap_words=25)

async def write_chunk_to_pinecone(chunk: dict):
    try: 
        print("Starting chunking pipeline")
        vector = get_embedding(chunk["text"])

        record_id = f"zoom_chink{str(uuid.uuid4())}"
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

async def zoom_websocket_listener_task():
    headers = {"Authorization:" f"Bearer {ZOOM_AUTH_TOKEN}"}

    async def inactivty_watchdog():
        while True: 
            await asyncio.sleep(15)
            if time.time() - chunker.last_received_time > 45:
                flushed_chunk = chunker.force_flush()
                if flushed_chunk:
                    await write_chunk_to_pinecone(flushed_chunk)

    asyncio.create_task(inactivty_watchdog())

    try:
        async with websockets.connect(ZOOM_RTMS_WS_URL, extra_headers=headers) as ws:
            async for message in ws:
                event_data = json.loads(message)

                if event_data.get("msg_type") == 17:
                    content = event_data.get("content", {})
                    speaker = content.get("displayName", "System Particpant")
                    text_segment = content.get("data", "")

                    if not text_segment.strip():
                        continue

                    completed_chunk = chunker.add_transcript_segment(speaker, text_segment)
                    if completed_chunk:
                        asyncio.create_task(write_chunk_to_pinecone(completed_chunk))

    except Exception as e:
        print(f"Zoom connection error: {e}")

def query_pinecone_knowledge_base(query: str) -> str:
    query_vector = get_embedding(query)
    try:
        results = index.query(
            vector=query_vector,
            top_k=2,
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
            "Synthesis specific, accurate answers utlitizing the context and provided Pinecone tool."
            "Mention certain timestamps whenever possible."
        ),
        tools=[query_pinecone_knowledge_base]
    )

    async with Agent(config) as agent:
        composed_msg = (
            f"Conversation context: {context}\n\n"
            f"Initial user response: {question}"
        )
        response = await agent.chat(composed_msg)
        print(f"Agent: {await response.text()}")

async def rag_agent_main(context, question):
    zoom_task = asyncio.create_task(zoom_websocket_listener_task())
    await asyncio.sleep(5)

    await run_rag_agent(question=question, context=context)

    await asyncio.sleep(5)
    zoom_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())

