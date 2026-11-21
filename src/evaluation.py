"""Human-reviewed evaluation utilities for SatQuery AI."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CRITERIA = ("feature_correctness", "spatial_accuracy", "groundedness",
            "completeness", "uncertainty_handling")


def load_dataset(path="data/evaluation_dataset.json"):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError("Dataset must contain an 'images' list.")
    seen_images = set()
    for image in data["images"]:
        image_id = image.get("image_id")
        if not image_id or image_id in seen_images:
            raise ValueError(f"Missing or duplicate image_id: {image_id!r}")
        seen_images.add(image_id)
        seen_questions = set()
        for question in image.get("questions", []):
            qid = question.get("question_id")
            if not qid or qid in seen_questions or not question.get("question"):
                raise ValueError(f"Invalid question in image {image_id}: {qid!r}")
            if "reference_answer" not in question:
                raise ValueError(f"Missing reference_answer for {image_id}/{qid}")
            seen_questions.add(qid)
    return data


def list_cases(dataset):
    return [
        {"image_id": image["image_id"],
         "category": image.get("category", "uncategorized"),
         "image_path": image.get("image_path", ""),
         "question_id": q["question_id"],
         "question_type": q.get("question_type", "general"),
         "question": q["question"],
         "reference_answer": q.get("reference_answer", "")}
        for image in dataset["images"] for q in image.get("questions", [])
    ]


def validate_scores(scores):
    missing = [c for c in CRITERIA if c not in scores]
    if missing:
        raise ValueError("Missing scores: " + ", ".join(missing))
    for criterion in CRITERIA:
        value = scores[criterion]
        if isinstance(value, bool) or not isinstance(value, int) or value not in range(1, 6):
            raise ValueError(f"{criterion} must be an integer from 1 to 5.")


def create_evaluation_record(case, model_answer, scores, reviewer_notes="", reviewer=""):
    validate_scores(scores)
    if not model_answer or not model_answer.strip():
        raise ValueError("model_answer cannot be empty.")
    return {
        **{k: case[k] for k in ("image_id", "category", "question_id",
                                  "question_type", "question", "reference_answer")},
        "model_answer": model_answer.strip(),
        "scores": {c: scores[c] for c in CRITERIA},
        "average_score": round(sum(scores[c] for c in CRITERIA) / len(CRITERIA), 2),
        "reviewer_notes": reviewer_notes.strip(),
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat()
    }


def append_result(record, path="results/evaluation_results.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            results = json.load(f)
        if not isinstance(results, list):
            raise ValueError("Results file must contain a JSON list.")
    results.append(record)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(path="results/evaluation_results.json"):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise ValueError("Results file must contain a JSON list.")
    return results


def summarize_results(results):
    if not results:
        return {"response_count": 0, "overall_average": None,
                "criterion_averages": {}, "category_averages": {}}
    criterion_values = defaultdict(list)
    category_values = defaultdict(list)
    averages = []
    for row in results:
        validate_scores(row["scores"])
        avg = row.get("average_score", sum(row["scores"].values()) / len(CRITERIA))
        averages.append(avg)
        category_values[row.get("category", "uncategorized")].append(avg)
        for c in CRITERIA:
            criterion_values[c].append(row["scores"][c])
    return {
        "response_count": len(results),
        "overall_average": round(sum(averages) / len(averages), 2),
        "criterion_averages": {k: round(sum(v) / len(v), 2)
                               for k, v in criterion_values.items()},
        "category_averages": {k: round(sum(v) / len(v), 2)
                              for k, v in category_values.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI evaluation helper")
    parser.add_argument("command", choices=("list", "summary"))
    parser.add_argument("--dataset", default="data/evaluation_dataset.json")
    parser.add_argument("--results", default="results/evaluation_results.json")
    args = parser.parse_args()
    if args.command == "list":
        cases = list_cases(load_dataset(args.dataset))
        for case in cases:
            print(f'{case["image_id"]} | {case["question_id"]} | '
                  f'{case["category"]} | {case["question"]}')
        print(f"\nTotal questions: {len(cases)}")
    else:
        print(json.dumps(summarize_results(load_results(args.results)), indent=2))


if __name__ == "__main__":
    main()
