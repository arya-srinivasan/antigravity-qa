import os
import asyncio
from pinecone import Pinecone
from google.antigravity import Agent, LocalAgentConfig

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "pinecone-index"
index = pc.Index(index_name)

def query_pinecone_knowledge_base(query: str) -> str:
    dummy_vector = [0.1] * 1536

    results = index.query(
        vector=dummy_vector,
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

async def main():
    config = LocalAgentConfig(
        tools=[query_pinecone_knowledge_base]
    )

    async with Agent(config) as agent:
        prompt = ""
        print(f"User: {prompt}\n")

        response = await agent.chat(prompt)
        print(f"Agent: {await response.text()}")

if __name__ == "__main__":
    asyncio.run(main())