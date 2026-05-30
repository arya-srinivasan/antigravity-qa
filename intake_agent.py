import asyncio

from google.antigravity import Agent, LocalAgentConfig
from database.db import add_question 
import uuid
from dotenv import load_dotenv

load_dotenv()

instructions="""
You are a routing agent.
Your solve objective is to help the user classify their question into the correct category.

The available catergories are:
1. Career exploration
2. Homework help
3. Other

Rules:
- Ask target clarifying questions.
- Once confident, output: "CLASSIFICATION: [Catergory Name]"
"""


async def update_summary(current_summary: str, user_msg: str, agent_msg: str) -> str:
    config = LocalAgentConfig()

    summarizer_prompt = f"""
    Your job is to update a short, running summary of the conversation based on the new information provided to you.

    Current Summary: "{current_summary}"
    New Exchange:
    User: {user_msg}
    Agent: {agent_msg}

    Output a new, consolidated summary combining the past context with the new details. Keep it under 2 sentences.
    """

    async with Agent(config) as summarizer:
        response = await summarizer.chat(summarizer_prompt)
        return await response.text()
    
async def run_managed_memory_session():
    conversation_id = str(uuid.uuid4())
    config = LocalAgentConfig()

    conversation_summary = "No conversation yet."

    print("Router Agent intialized with managed memory. How can we help you?")

    async with Agent(config) as intake_agent:

        first_time = True

        while True:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                break
            if first_time:
                first_user_query = user_input
                first_time = False

            context_prompt = f"""
            CURRENT CONVERSATION MEMORY SUMMARY:
            {conversation_summary}

            System Instructions: {instructions}

            User's new message: {user_input}
            """

            response = await intake_agent.chat(context_prompt)
            agent_response = await response.text()

            print(f"\nAgent: {agent_response}")

            conversation_summary - await update_summary(conversation_summary, user_input, agent_response)

            if "CLASSIFICATION:" in agent_response:
                print(f"\n[Final Memory State]: {conversation_summary}")
                add_question(conversation_id=conversation_id, question=first_user_query, context=conversation_summary, topic=agent_response.strip("CLASSIFICATION:"), type="basic")
                break
            

if __name__ == "__main__":
    asyncio.run(run_managed_memory_session())