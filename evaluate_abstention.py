
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "hard_evaluation_v1.json"
OUTPUT_PATH = ROOT / "reports" / "abstention_analysis_v1.json"


def load_cases():
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Report not found: {REPORT_PATH}"
        )

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    cases = []

    for record in report["results"]:
        results = record["results"]

        if not results:
            raise ValueError(
                "A query has no retrieval results."
            )

        relevant = set(record["relevant_documents"])

        # A positive prediction requires a document
        # that actually supports the question.
        answerable = bool(relevant)

        cases.append({
            "question": record["question"],
            "answerable": answerable,
            "top1_document": results[0]["filename"],
            "distance": float(results[0]["distance"]),
        })

    return cases


def evaluate_threshold(cases, threshold):
    tp = tn = fp = fn = 0
    details = []

    for case in cases:
        predicted_answerable = (
            case["distance"] <= threshold
        )

        actual = case["answerable"]

        if predicted_answerable and actual:
            tp += 1
        elif not predicted_answerable and not actual:
            tn += 1
        elif predicted_answerable and not actual:
            fp += 1
        else:
            fn += 1

        details.append({
            **case,
            "predicted_answerable": predicted_answerable,
            "correct": predicted_answerable == actual,
        })

    positive_count = tp + fn
    negative_count = tn + fp

    tpr = tp / positive_count if positive_count else 0.0
    tnr = tn / negative_count if negative_count else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_recall": tpr,
        "negative_rejection_rate": tnr,
        "balanced_accuracy": (tpr + tnr) / 2,
        "details": details,
    }


def main():
    cases = load_cases()

    positives = [
        case["distance"]
        for case in cases
        if case["answerable"]
    ]

    negatives = [
        case["distance"]
        for case in cases
        if not case["answerable"]
    ]

    if not positives or not negatives:
        raise ValueError(
            "Both positive and negative cases are required."
        )

    values = sorted(set(
        case["distance"] for case in cases
    ))

    # Evaluate boundaries and all meaningful
    # thresholds between observed distances.
    thresholds = [values[0] - 0.001]

    thresholds.extend(
        (left + right) / 2
        for left, right in zip(values, values[1:])
    )

    thresholds.append(values[-1] + 0.001)

    evaluations = [
        evaluate_threshold(cases, threshold)
        for threshold in thresholds
    ]

    # Prefer balanced performance, then fewer
    # false acceptances, then fewer false rejections.
    best = max(
        evaluations,
        key=lambda r: (
            r["balanced_accuracy"],
            -r["fp"],
            -r["fn"],
        ),
    )

    print("=" * 60)
    print("BGE-M3 ABSTENTION THRESHOLD ANALYSIS")
    print("=" * 60)

    print("Positive questions:", len(positives))
    print("Negative questions:", len(negatives))

    print("\nPositive distance range:")
    print(f"  {min(positives):.4f} - {max(positives):.4f}")

    print("Negative distance range:")
    print(f"  {min(negatives):.4f} - {max(negatives):.4f}")

    print("\nBest development threshold:")
    print(f"  {best['threshold']:.4f}")

    print("\nConfusion Matrix:")
    print("TP:", best["tp"])
    print("TN:", best["tn"])
    print("FP:", best["fp"])
    print("FN:", best["fn"])

    print("\nBalanced Accuracy:")
    print(f"  {best['balanced_accuracy']:.2%}")

    print("\nFalse Acceptances:")
    for case in best["details"]:
        if case["predicted_answerable"] and not case["answerable"]:
            print(
                f"  {case['question']}"
                f" | distance={case['distance']:.4f}"
            )

    print("\nFalse Rejections:")
    for case in best["details"]:
        if not case["predicted_answerable"] and case["answerable"]:
            print(
                f"  {case['question']}"
                f" | distance={case['distance']:.4f}"
            )

    output = {
        "dataset": "hard_development_v1",
        "method": "top1_cosine_distance_threshold",
        "development_only": True,
        "not_holdout_validated": True,
        "best_result": best,
        "all_thresholds": [
            {
                k: v
                for k, v in result.items()
                if k != "details"
            }
            for result in evaluations
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nReport saved:", OUTPUT_PATH)
    print(
        "\nWARNING: This threshold was fitted "
        "and evaluated on the same development data."
    )
    print(
        "It must NOT be reported as independent "
        "abstention performance."
    )


if __name__ == "__main__":
    main()
