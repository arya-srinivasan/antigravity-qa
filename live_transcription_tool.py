import os
import asyncio 
import re
import time
import uuid
from typing import Any, Callable, Awaitable, Dict, List, Optional
from playwright.async_api import async_playwright, Page
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("PINECONE_API_KEY variable missing")
    PINECONE_API_KEY = "dummy-key"

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "zoom-transcripts"
index = pc.Index(INDEX_NAME)


def get_embedding(text: str) -> List[float]:
    # Placeholder for embedding generation logic
    return [0.1] * 512  # Example fixed-size embedding

async def write_chunk_to_pinecone(chunk: str, timestamp: float):
    try: 
        print("Start chunking")
        vector = get_embedding(chunk["text"])

        record_id = f"zoom_chunk_{str(uuid.uuid4())}"
        metadata = {
            "text": chunk["text"],
            "speakers": ",".join(chunk["metadata"]["speakers"]),
            "start_time": chunk["metadata"]["start_time"],
            "end_time": chunk["metadata"]["end_time"]
        }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: index.upsert(
            vectors=[{
                "id": record_id,
                "values": vector,
                "metadata": metadata
            }]
        ))
        print(f"Document chunk {record_id} successfully written to Pinecone")
    except Exception as e:
        print(f"Error writing chunk to Pinecone: {e}")

class RealTimeTranscriptChunker:
    def __init__(self, target_word_count: int = 100, overlap_word_count: int = 20):
        self.target_word_count = target_word_count
        self.overlap_word_count = overlap_word_count
        self.current_chunk_words = []
        self.last_received_timestamp = time.time()

    def add_transcript_chunk(self, speaker: str, text: str) -> Optional[Dict[str, Any]]:
        self.last_received_timestamp = time.time()
        words = text.split()
        for word in words: 
            self.current_chunk_words.append(word)
        if len(self.current_chunk_words) >= self.target_word_count:
            return self.flush_and_slide()
        return None

    def force_flush(self) -> Optional[Dict[str, Any]]:
        if self.current_chunk_words:
            return self.flush_and_slide()
        return None

    def flush_and_slide(self) -> Dict[str, Any]:
        full_text = " ".join([item[0] for item in self.current_chunk_words])
        unique_speakers = list(set([item[1] for item in self.current_chunk_words]))
        start_time = self.current_chunk_words[0][2] if self.current_chunk_words else 0
        end_time = self.current_chunk_words[-1][2] if self.current_chunk_words else 0

        chunk_payload = {
            "text": full_text,
            "metadata": {
                "speakers": unique_speakers,
                "start_time": start_time,
                "end_time": end_time,
                "word_count": len(self.current_chunk_words),
            }
        }

        self.current_chunk_words = self.current_chunk_words[-self.overlap_word_count:]
        return chunk_payload

chunker = RealTimeTranscriptChunker(target_word_count=100, overlap_word_count=20)

SELECTORS = {
    "name_input": 'input[placeholder*-"name" i], input[id*=inputname"i]',
    "join_button": 'button[class*="preview-join-button"], button[class*="joinBtn"]',
    "cc_btn":          'button[aria-label*="caption" i], button[aria-label*="CC" i]',
    "caption_item":    '[class*="caption-line"], [class*="transcript-item"]',
    "caption_speaker": '[class*="caption-speaker"], [class*="speaker-name"]',
    "caption_text":    '[class*="caption-text"], [class*="caption-content"]',
    "ended_overlay":   '[class*="meeting-ended"], [class*="leave-meeting"]',
}

async def run_zoom_bot(
    meeting_url: str,
    on_caption: Callable[[str, str], Awaitable[None]] | None = None,
    display_name: str = "Transcription Bot",
    poll_interval_ms: int = 1000,
):
    if on_caption is None:
        on_caption = _default_on_caption

    meeting_id = _extract_meeting_id(meeting_url)
    seen: dict[str, bool] = {}
    interval_s = poll_interval_ms / 1000

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--diable-dev-shm-usage",
            ],                                               
        )
        context = await browser.new_context(
            permission=["microphone", "camera"],
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try: 
            await __join(page, meeting_url, display_name)
            await _enable_captions(page)
            
            while True:
                if await page.query_selector(SELECTORS["ended_overlay"]):
                    print("Meeting has ended. Exiting.")
                    break
                
                now = time.time()
                caption_items = await page.query_selector_all(SELECTORS["caption_item"])
                for item in caption_items:
                    caption_speaker = await item.query_selector(SELECTORS["caption_speaker"])
                    caption_text = await item.query_selector(SELECTORS["caption_text"])
                    if not caption_text:
                        continue
                        
                    speaker = (await caption_speaker.inner_text()) if caption_speaker else "Unknown"

                    text = (await caption_text.inner_text()).strip()
                    if not text:
                        continue
                        
                    key = f"{speaker}:{text}"
                    if now - seen.get(key, 0) > 5.0:  
                        seen[key] = now
                        try: 
                            await on_caption(speaker, text)
                        except Exception as e:
                            print(f"Error in on_caption callback: {e}")

                    seen = {k: v for k, v in seen.items() if now - v < 30.0}
                    await asyncio.sleep(interval_s)
        except Exception as e:
            print(f"Error in run_zoom_bot: {e}")
            raise
        finally:
            chunk = chunker.force_flush()
            if chunk:
                await write_chunk_to_pinecone(chunk, time.time())
            await browser.close()
            print("Browser closed, bot exiting.")

async def _default_on_caption(speaker: str, text: str):
    chunk = chunker.add_transcript_chunk(speaker, text)
    if chunk:
        await write_chunk_to_pinecone(chunk, time.time())

async def __join(page: Page, meeting_url: str, display_name: str):
    await page.goto(meeting_url, wait_until="domcontentloaded", timeout=30_000)
    try:
        await page.wait_for_selector(SELECTORS["name_input"], timeout=15_000)
        await page.fill(SELECTORS["name_input"], display_name)
    except Exception:
        print("Failed to find name input field.")
        raise
    try:
        btn = await page.wait_for_selector(SELECTORS["join_button"])
        await btn.click()
    except Exception:
        print("Failed to find or click join button.")
        raise
        
    await page.wait_for_selector(SELECTORS["cc_btn"], timeout=60_000)
    print("Joined meeting successfully.")

async def _enable_captions(page: Page):
    try:
        btn = await page.wait_for_selector(SELECTORS["cc_btn"], timeout=15_000)
        if btn and await btn.get_attribute("aria-pressed") != "true":  
            await btn.click()
            await asyncio.sleep(2)
            print("Closed captions enabled.")
    except Exception:
        print("Failed to find or click closed captions button.")
        raise
        
def _extract_meeting_id(meeting_url: str) -> Optional[str]:
    match = re.search(r"zoom\.us/(?:j|my)/(\d+)", meeting_url)
    return match.group(1) if match else None

