import os
import argparse
import sys
from narrative_events import NarrativeEventExtractor

GEMMA_PATH = "google/gemma-3-12b-it"
HF_TOKEN = 'dummy' 
AMR_MODEL_DIR = "./amrlib_models/model_parse_xfm_bart_large-v0_1_0"

def step_8_event_extraction(file_list):
    """
    Step 8: Parse text into AMR graphs and extract action predicates.
    """
    print("\n[Step 8] Starting Event Extraction (AMR)...")
    
    extractor = NarrativeEventExtractor(amr_model_dir=AMR_MODEL_DIR, semlink_path="./code/pb-vn2.json")
    extractor.process_files(file_list)
    
    print("[Step 8] Event Extraction complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Narrative Similarity Pipeline - Phase 2")
    
    parser.add_argument("--test", type=str, choices=['yes', 'no'], default='yes', 
                        help="Use test data (amrtestdata) or production data (amrdata). Default: yes")

    parser.add_argument("--step8", action="store_true", help="Run Step 8: Event Extraction")
    parser.add_argument("--all", action="store_true", help="Run all Phase 2 steps sequentially")

    args = parser.parse_args()

    if not (args.step8 or args.all):
        parser.print_help()
        sys.exit(0)

    base_dir = "./amrdatafinal4" if args.test == "no" else "./amrtestdata"

    FILES_8 = [
        f"{base_dir}/synthetic_data_for_classification_moe_abstract_outcome_events_coreferenced.jsonl",
        f"{base_dir}/dev_track_a_moe_abstract_outcome_events_coreferenced.jsonl"
    ]

    if args.step8 or args.all:
        step_8_event_extraction(FILES_8)

if __name__ == "__main__":
    main()