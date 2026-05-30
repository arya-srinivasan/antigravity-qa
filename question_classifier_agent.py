from google.antigravity import Agent, LocalAgentConfig
from dotenv import load_dotenv
import json
from database.db import add_question, get_conversation_context
from faculty_assistant import run_faculty_assistant
from rag_agent import run_rag_agent

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

async def handle_student_question(conversation_id, question, context):
    config = LocalAgentConfig()

    async with Agent(config) as agent:
        response = await agent.chat(question)
        try:
            decision = json.load(await response.text())

            add_question(conversation_id, question)

            if decision["decision"] == "faculty":
                faculty_response = await run_faculty_assistant(question, get_conversation_context(conversation_id, question))
                return faculty_response
            else:
                agent_response = await run_rag_agent(question, get_conversation_context(conversation_id, question))
                return agent_response
        except json.JSONDecodeError:
                return "Classifier returned an unexpected response."
        


