import os
import argparse
import sys
from narrative_classification import NarrativeSimilarityClassifier
from scoring import NarrativeAggregator

GEMMA_PATH = "google/gemma-3-12b-it"
HF_TOKEN = 'dummy' 

def step_9_narrative_classification(file_list):
    """
    Step 9: Use Gemma 3 to compare event sequences and perform final classification.
    """
    print("\n[Step 9] Starting Final Narrative Classification...")

    classifier = NarrativeSimilarityClassifier(
        model_id=GEMMA_PATH, 
        access_token=HF_TOKEN
    )
    
    classifier.process_files(file_list)
    print("[Step 9] Pipeline Complete. Final results saved to _output.jsonl files.")

def step_10_scoring(file_list):
    """
    Step 10: Aggregate results from all experts and produce final decision.
    """
    print("\n[Step 10] Starting Final Metric Fusion (MoE)...")
    
    aggregator = NarrativeAggregator()
    aggregator.process_files(file_list)
    
    print("[Step 10] Aggregation Complete. Final classification is done.\n")

def main():
    parser = argparse.ArgumentParser(description="Narrative Similarity Multi-Step Pipeline")
    
    parser.add_argument("--test", type=str, choices=['yes', 'no'], default='yes', 
                        help="Use test data (amrtestdata) or production data (amrdata). Default: yes")

    parser.add_argument("--step9", action="store_true", help="Run Step 9: Narrative Classigication")
    parser.add_argument("--step10", action="store_true", help="Run Step 10: Final Scoring")
    parser.add_argument("--all", action="store_true", help="Run all steps sequentially")

    args = parser.parse_args()

    if not (args.step9 or args.step10 or args.all):
        parser.print_help()
        sys.exit(0)

    base_dir = "./amrdatafinal4" if args.test == "no" else "./amrtestdata"

    FILES_9 = [f"{base_dir}/synthetic_data_for_classification_moe_abstract_outcome_events_coreferenced_sequence.jsonl", f"{base_dir}/dev_track_a_moe_abstract_outcome_events_coreferenced_sequence.jsonl"]
    FILES_10 = [f"{base_dir}/synthetic_data_for_classification_moe_abstract_outcome_events_output.jsonl", f"{base_dir}/dev_track_a_moe_abstract_outcome_events_output.jsonl"]
    
    if args.step9 or args.all: 
        step_9_narrative_classification(FILES_9)

    if args.step10 or args.all: 
        step_10_scoring(FILES_10)

if __name__ == "__main__":
    main()