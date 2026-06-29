"""Query router using Groq Llama 3.3 70B for fast classification + query rewrite."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import groq_chat
from .prompts import ROUTER_FEWSHOT, ROUTER_SYSTEM

_ALLOWED_INTENTS = {
    "statute_lookup",
    "case_research",
    "legal_concept",
    "procedure",
    "general_legal_qa",
    "out_of_scope",
}
_ALLOWED_TYPES = {"statute", "case", "rule", "constitution"}


@dataclass
class RouteDecision:
    intent: str
    doc_type_filter: list[str]
    rewritten_query: str
    keywords: list[str]

    @property
    def is_legal(self) -> bool:
        return self.intent != "out_of_scope"


def route(query: str) -> RouteDecision:
    """Classify the user's legal query and rewrite it for retrieval."""
    if not query.strip():
        return RouteDecision("out_of_scope", [], "", [])

    raw = groq_chat(
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM + "\n\n" + ROUTER_FEWSHOT},
            {"role": "user", "content": f"Q: {query.strip()}\nA:"},
        ],
        temperature=0.0,
        max_tokens=400,
        response_format_json=True,
    )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return RouteDecision("general_legal_qa", [], query.strip(), [])

    intent = data.get("intent", "general_legal_qa")
    if intent not in _ALLOWED_INTENTS:
        intent = "general_legal_qa"

    doc_types = [t for t in data.get("doc_type_filter") or [] if t in _ALLOWED_TYPES]
    rewritten = (data.get("rewritten_query") or query).strip()
    keywords = [k for k in data.get("keywords") or [] if isinstance(k, str)]
    return RouteDecision(intent, doc_types, rewritten, keywords)
