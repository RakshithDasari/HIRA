import os
import re
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

qwen = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

CONTROLLER_PROMPT = """\
You are an intelligent research agent.

User question: {query}

Retrieved context:
Entities: {entities}
Facts: {facts}

Based on this context, decide:
1. If you have ENOUGH information to answer the question accurately — answer it.
2. If you need MORE information — provide a better search query.

Return ONLY valid JSON. No explanation. No markdown.

If you can answer:
{{"done": true, "answer": "your complete answer here", "next_query": null}}

If you need more information:
{{"done": false, "answer": null, "next_query": "your improved search query here"}}
"""

class Controller:
    """
    Agentic loop — decides whether retrieved context is sufficient
    to answer the query or whether another retrieval pass is needed.
    Replaces Graph-R1's RL-trained exit policy with LLM reasoning.
    This makes the system domain-agnostic — no retraining needed.
    """

    def __init__(self, max_turns: int = 3):
        # max_turns prevents infinite loop if LLM keeps saying "need more info"
        self.max_turns = max_turns

    def decide(self, query: str, context: dict) -> dict:
        # format context for prompt
        entity_names = [e["name"] for e in context.get("entities", [])]
        facts = context.get("facts", [])

        prompt = CONTROLLER_PROMPT.format(
            query=query,
            entities=", ".join(entity_names) if entity_names else "none",
            facts="\n".join(f"- {f}" for f in facts) if facts else "none"
        )

        response = qwen.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise research agent. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        raw = response.choices[0].message.content

        # defensive parse
        raw = re.sub(r'```json|```', '', raw).strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            # if parse fails — force retry with original query
            log.warning("Controller parse failed — forcing retry")
            return {"done": False, "answer": None, "next_query": query}

        result = json.loads(match.group())
        return {
            "done": bool(result.get("done", False)),
            "answer": result.get("answer"),
            "next_query": result.get("next_query")
        }

    def run(self, query: str, retriever) -> str:
        """
        Full agentic loop.
        Calls retriever → controller → retriever → controller
        until done=True or max_turns reached.
        """
        current_query = query
        context = {}

        for turn in range(self.max_turns):
            log.info(f"Turn {turn+1}/{self.max_turns} — searching: '{current_query}'")

            # retrieve context
            context = retriever.search(current_query)

            # decide
            decision = self.decide(query, context)

            if decision["done"]:
                log.info(f"Answer found on turn {turn+1}")
                return decision["answer"]

            # not done — update query for next turn
            current_query = decision["next_query"]
            log.info(f"Retrying with: '{current_query}'")

        # max turns reached — answer with whatever we have
        log.warning("Max turns reached — answering with available context")
        final = self.decide(query, context)
        return final["answer"] or "I could not find a sufficient answer."