import asyncio

from google.antigravity import Agent, LocalAgentConfig
from database.db import add_question 
import uuid
from dotenv import load_dotenv

load_dotenv()

async def classify_question(question: str, summary: str) -> str | None:
    categories = ["Neural networks", "Backpropagation", "Gradient descent", "Chain rule"]
    
    config = LocalAgentConfig(model="gemini-2.5-flash")
    
    async with Agent(config) as classifier:
        response = await classifier.chat(
            f"You are a classifier. Output only the category name, nothing else.\n"
            f"Categories: Neural networks, Backpropagation, Gradient descent, Chain rule, Other\n"
            f"Question: {question}\n"
            f"Context: {summary}\n"
            f"Output only the category name."
        )
        text = await response.text()
        print(f"DEBUG classifier output: {text}")
        
        text_lower = text.lower()
        for category in categories:
            if category.lower() in text_lower:
                return category
        
    return None
    
async def run_managed_memory_session():
    conversation_id = str(uuid.uuid4())
    conversation_history = []
    conversation_summary = "No conversation yet."
    config = LocalAgentConfig()

    print("\nUser: ", end="", flush=True)
    first_user_query = await asyncio.get_event_loop().run_in_executor(None, input, "")

    if first_user_query.lower() == "exit":
        return
    
    if first_user_query.lower() == "start zoom":
        return "start zoom"

    conversation_history.append({"role": "user", "content": first_user_query})
    conversation_summary = f"Student asked: {first_user_query}"

    max_turn = 0

    async with Agent(config) as agent:

        while True:
            # Try to classify with current context
            topic = await classify_question(first_user_query, conversation_summary)
            max_turn += 1
            if topic or max_turn == 5:
                add_question(conversation_id=conversation_id, question=first_user_query, context=conversation_summary, topic=topic, type="basic")
                print(f"\nAgent: Got it! I've routed your question about {topic}.")
                return None, first_user_query, conversation_summary
            
            response = await agent.chat(
                f"Conversation so far:\n{conversation_summary}\n\n"
                f"Write only a single question mark ending sentence that asks for clarification. "
                f"Do not write anything else. Do not explain. Do not teach. "
                f"Your entire response must be one sentence ending with a question mark."
            )
            clarifying_question = await response.text()
            print(f"\nAgent: {clarifying_question}\n")

            conversation_history.append({"role": "assistant", "content": clarifying_question})

            print("\nUser: ", end="", flush=True)
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "")

            if user_input.lower() == "exit":
                return
            
            if user_input.lower() == "start zoom":
                return "start zoom"

            conversation_history.append({"role": "user", "content": user_input})
            conversation_summary += f"\nAgent asked: {clarifying_question}\nStudent replied: {user_input}"

if __name__ == "__main__":
    asyncio.run(run_managed_memory_session())