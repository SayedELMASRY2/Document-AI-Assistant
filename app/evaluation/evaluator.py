import time
import json
import logging
from typing import List, Optional

from app.evaluation.metrics import (
    calculate_keyword_coverage,
    calculate_faithfulness,
    calculate_context_precision,
    calculate_context_recall,
    calculate_answer_relevance
)

logger = logging.getLogger(__name__)

class SystemEvaluator:
    LATENCY_THRESHOLD = 20.0
    MIN_ANSWER_LENGTH = 50

    def __init__(self):
        self.results: List[dict] = []

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        expected_keywords: List[str],
        latency: float,
        context_docs: Optional[list] = None,
    ) -> dict:
        found, coverage = calculate_keyword_coverage(answer, expected_keywords)
        faithfulness = calculate_faithfulness(answer, context_docs)
        context_precision = calculate_context_precision(expected_keywords, context_docs)
        context_recall = calculate_context_recall(expected_keywords, context_docs)
        answer_relevance = calculate_answer_relevance(question, answer)
        groundedness = faithfulness

        result = {
            "question":          question,
            "answer_length":     len(answer),
            "answer_length_ok":  len(answer) >= self.MIN_ANSWER_LENGTH,
            "keyword_coverage":    coverage,
            "found_keywords":      found,
            "missing_keywords":    [kw for kw in expected_keywords if kw not in found],
            "faithfulness":        faithfulness,
            "answer_relevance":    answer_relevance,
            "context_precision":   context_precision,
            "context_recall":      context_recall,
            "groundedness":        groundedness,
            "latency_seconds":   latency,
            "latency_ok":        latency < self.LATENCY_THRESHOLD,
        }
        self.results.append(result)
        return result

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

    def save_report(self, path: str = "eval_report.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Eval report saved → {path}")

    def print_summary(self):
        if not self.results:
            print("No results found.")
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
