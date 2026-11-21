
from src.evaluation import (
    load_dataset,
    list_cases,
    create_evaluation_record,
    append_result
)

dataset = load_dataset("data/evaluation_dataset.json")
cases = list_cases(dataset)

case = next(
    c for c in cases
    if c["question_id"] == "barren_01_q3"
)

model_answer = """PASTE THE COMPLETE BARREN LAND Q3 MODEL RESPONSE HERE"""

scores = {
    "feature_correctness": 4,
    "spatial_accuracy": 4,
    "groundedness": 3,
    "completeness": 4,
    "uncertainty_handling": 4,
}

record = create_evaluation_record(
    case=case,
    model_answer=model_answer,
    scores=scores,
    reviewer_notes=(
        "Good identification of terrain features and explicit "
        "acknowledgement of uncertainty. Interpretations involving "
        "playas, alluvial fans, desert classification, erosion, "
        "and mineralogy require further evidence."
    ),
    reviewer="human_reviewer"
)

append_result(record, "results/evaluation_results.json")

print("Evaluation recorded successfully.")
print("Average score:", record["average_score"])
