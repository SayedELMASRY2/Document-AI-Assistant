import re
from typing import List, Optional

def _tokens(text: str) -> set:
    return set(re.findall(r"[\w\u0600-\u06FF]{3,}", text.lower()))

def calculate_keyword_coverage(answer: str, expected_keywords: List[str]) -> tuple:
    found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
    coverage = round(len(found) / len(expected_keywords) * 100, 1) if expected_keywords else 100.0
    return found, coverage

def calculate_faithfulness(answer: str, context_docs: Optional[list]) -> float:
    if not context_docs:
        return 0.0
    ctx_text   = " ".join(d.page_content for d in context_docs)
    ans_tokens = _tokens(answer)
    ctx_tokens = _tokens(ctx_text)
    return round(len(ans_tokens & ctx_tokens) / max(len(ans_tokens), 1) * 100, 1)

def calculate_context_precision(expected_keywords: List[str], context_docs: Optional[list]) -> float:
    if not context_docs or not expected_keywords:
        return 0.0
    useful = sum(
        1 for d in context_docs
        if any(kw.lower() in d.page_content.lower() for kw in expected_keywords)
    )
    return round(useful / len(context_docs) * 100, 1)

def calculate_context_recall(expected_keywords: List[str], context_docs: Optional[list]) -> float:
    if not context_docs or not expected_keywords:
        return 0.0
    ctx_text = " ".join(d.page_content for d in context_docs)
    found_in_ctx = [kw for kw in expected_keywords if kw.lower() in ctx_text.lower()]
    return round(len(found_in_ctx) / len(expected_keywords) * 100, 1)

def calculate_answer_relevance(question: str, answer: str) -> float:
    q_tokens  = _tokens(question)
    a_tokens  = _tokens(answer)
    return round(len(q_tokens & a_tokens) / max(len(q_tokens), 1) * 100, 1)
