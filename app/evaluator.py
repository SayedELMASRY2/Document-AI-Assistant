"""
Evaluation Layer — §4.10 Expanded Metrics
Measures: faithfulness, answer relevance, context precision, context recall,
          groundedness, latency, answer length, keyword coverage.
"""

import time
import json
import re
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class SystemEvaluator:
    LATENCY_THRESHOLD = 20.0   # seconds — suitable for API-based LLMs
    MIN_ANSWER_LENGTH = 50

    def __init__(self):
        self.results: List[dict] = []

    # ── Tokenizer (shared, Arabic + Latin) ────────────────────────────
    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[\w\u0600-\u06FF]{3,}", text.lower()))

    # ── Core evaluation ───────────────────────────────────────────────
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        expected_keywords: List[str],
        latency: float,
        context_docs: Optional[list] = None,     # List[Document]
    ) -> dict:
        """Evaluate one Q-A pair. context_docs optional for richer metrics."""

        # ── 1. Keyword coverage (legacy) ──────────────────────────────
        found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        coverage = round(len(found) / len(expected_keywords) * 100, 1) if expected_keywords else 100.0

        # ── 2. Faithfulness — answer tokens in context ─────────────────
        faithfulness = 0.0
        if context_docs:
            ctx_text   = " ".join(d.page_content for d in context_docs)
            ans_tokens = self._tokens(answer)
            ctx_tokens = self._tokens(ctx_text)
            faithfulness = round(
                len(ans_tokens & ctx_tokens) / max(len(ans_tokens), 1) * 100, 1
            )

        # ── 3. Context Precision — retrieved chunks containing keywords ─
        context_precision = 0.0
        if context_docs and expected_keywords:
            useful = sum(
                1 for d in context_docs
                if any(kw.lower() in d.page_content.lower() for kw in expected_keywords)
            )
            context_precision = round(useful / len(context_docs) * 100, 1)

        # ── 4. Context Recall — keywords found in context ──────────────
        context_recall = 0.0
        if context_docs and expected_keywords:
            ctx_text = " ".join(d.page_content for d in context_docs)
            found_in_ctx = [kw for kw in expected_keywords if kw.lower() in ctx_text.lower()]
            context_recall = round(len(found_in_ctx) / len(expected_keywords) * 100, 1)

        # ── 5. Groundedness (same as faithfulness for simplicity) ─────
        groundedness = faithfulness

        # ── 6. Answer Relevance — keyword overlap Q ↔ A ───────────────
        q_tokens  = self._tokens(question)
        a_tokens  = self._tokens(answer)
        answer_relevance = round(
            len(q_tokens & a_tokens) / max(len(q_tokens), 1) * 100, 1
        )

        result = {
            "question":          question,
            "answer_length":     len(answer),
            "answer_length_ok":  len(answer) >= self.MIN_ANSWER_LENGTH,
            # Legacy
            "keyword_coverage":    coverage,
            "found_keywords":      found,
            "missing_keywords":    [kw for kw in expected_keywords if kw not in found],
            # New metrics
            "faithfulness":        faithfulness,
            "answer_relevance":    answer_relevance,
            "context_precision":   context_precision,
            "context_recall":      context_recall,
            "groundedness":        groundedness,
            # Latency
            "latency_seconds":   latency,
            "latency_ok":        latency < self.LATENCY_THRESHOLD,
        }
        self.results.append(result)
        return result

    # ── Latency stress test ───────────────────────────────────────────
    def latency_test(self, ask_fn, question: str, runs: int = 3) -> dict:
        latencies = []
        for _ in range(runs):
            start = time.time()
            ask_fn(question, "eval_session")
            latencies.append(time.time() - start)
        return {
            "avg_latency": round(sum(latencies) / len(latencies), 2),
            "min_latency": round(min(latencies), 2),
            "max_latency": round(max(latencies), 2),
            "within_5s":   all(l < 5 for l in latencies),
        }

    # ── Persistence ───────────────────────────────────────────────────
    def save_report(self, path: str = "eval_report.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Eval report saved → {path}")

    # ── Summary ───────────────────────────────────────────────────────
    def print_summary(self):
        if not self.results:
            print("لا توجد نتائج")
            return
        n = len(self.results)
        def avg(key): return sum(r.get(key, 0) for r in self.results) / n

        print("\n" + "="*55)
        print("📊 Evaluation Report")
        print("="*55)
        print(f"Total questions      : {n}")
        print(f"Keyword coverage     : {avg('keyword_coverage'):.1f}%")
        print(f"Faithfulness         : {avg('faithfulness'):.1f}%")
        print(f"Answer relevance     : {avg('answer_relevance'):.1f}%")
        print(f"Context precision    : {avg('context_precision'):.1f}%")
        print(f"Context recall       : {avg('context_recall'):.1f}%")
        print(f"Groundedness         : {avg('groundedness'):.1f}%")
        print(f"Avg latency          : {avg('latency_seconds'):.2f}s  (limit: {self.LATENCY_THRESHOLD}s)")
        lat_ok = sum(1 for r in self.results if r["latency_ok"])
        print(f"Within time limit    : {lat_ok}/{n} {'✅' if lat_ok == n else '⚠️'}")
        print("="*55)
