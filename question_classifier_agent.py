from google.antigravity import Agent, LocalAgentConfig
from dotenv import load_dotenv
import json
from database.db import get_conversation_context
from faculty_assistant import run_faculty_assistant
from rag_agent import run_rag_agent
from intake_agent import run_managed_memory_session

load_dotenv()

prompt="""
    You are a classifier for a student Q&A system used during live lectures.

    Your job is to decide whether a student's question can be answered by the AI agent, 
    or whether it requires the faculty member's attention.

    RULES - The agent CAN answer questions that are:
    - Factual and grounded in general course knowledge or provided materials
    - Conceptual clarifications that don't depend on what was just said in the lecture
    - Common questions with clear, well-established answers (definitions, formulas, processes)

    RULES - The agent CANNOT answer questions that:
    - Reference something specific the faculty just said or showed (e.g. "what did you mean by that?")
    - Require the faculty's personal opinion, judgment, or teaching intent
    - Are ambiguous and could be misinterpreted without more context from the live session
    - Are about logistics only the faculty would know (deadlines, grading, expectations)

    OUTPUT FORMAT:
    Respond only with a JSON object. No explanation, no extra text.

    {
    "decision": "agent" | "faculty",
    "confidence": 0.0 - 1.0,
    "reason": "one sentence explaining the decision"
    }
    """

async def handle_follow_up_question():
    result = await run_managed_memory_session()
    if result is None:
        print("Exiting session.")
        return
    
    _, question, context = result
    config = LocalAgentConfig(
        system_prompt=prompt,
        model="gemini-2.5-flash",
    )

    async with Agent(config) as agent:
        response = await agent.chat(question)
        try:
            decision = json.loads(await response.text())
        except json.JSONDecodeError:
                return "Classifier returned an unexpected response."
        
    if decision.get("decision") == "faculty" or decision.get("confidence", 0) < 0.8:
        await run_faculty_assistant(question, context)
        print("Your question was sent to the faculty")
    else:
        await run_rag_agent(question, context)