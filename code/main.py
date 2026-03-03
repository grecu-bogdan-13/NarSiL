import os
import argparse
import sys
from moe import NarrativeMoEPipeline
from consensus import ConsensusAnalyzer
from abstract_theme import NarrativeAbstractor
from abstract_comparison import AbstractComparator
from narrative_outcome import NarrativeOutcomeAnalyzer
from narrative_outcome_scorer import NarrativeOutcomeScorer
from narrative_coref import NarrativeCoreferenceSolver

GEMMA_PATH = "google/gemma-3-12b-it"
HF_TOKEN = 'dummy' 
OPENAI_KEY = "dummy"
GOOGLE_KEY = "dummy"


def step_1_moe(file_list):
    """
    Step 1: Load models and run the Mixture of Experts (MoE) inference.
    """
    print("\n[Step 1] Starting MoE Inference...")
    
    if "your_" in OPENAI_KEY or "your_" in GOOGLE_KEY:
        print("Error: Please set your real API keys in the configuration section.")
        sys.exit(1)

    pipeline = NarrativeMoEPipeline(
        gemma_path=GEMMA_PATH,
        hf_token=HF_TOKEN,
        openai_key=OPENAI_KEY,
        google_key=GOOGLE_KEY
    )
    
    pipeline.process_files(file_list)
    print("[Step 1] Inference complete.\n")

def step_2_consensus(file_list):
    """
    Step 2: Check consensus and generate final output files.
    """
    print("\n[Step 2] Starting Analysis...")
    
    analyzer = ConsensusAnalyzer()
    analyzer.process_files(file_list)
    
    print("[Step 2] Analysis complete.\n")

def step_3_abstraction(file_list):
    """
    Step 3: Generate abstracts for entries where consensus failed (no final_answer).
    """
    print("\n[Step 3] Starting Theme Abstraction...")
    
    abstractor = NarrativeAbstractor(
        model_path=GEMMA_PATH,
        hf_token=HF_TOKEN
    )
    abstractor.process_files(file_list)
    
    print("[Step 3] Abstraction complete.\n")

def step_4_abstract_comparison(file_list):
    """
    Step 4: Compare abstracts for entries that lacked a final_answer.
    """
    print("\n[Step 4] Starting Abstract Comparison...")
    comparator = AbstractComparator(GEMMA_PATH, HF_TOKEN)
    comparator.process_files(file_list)
    print("[Step 4] Comparison complete.\n")

def step_5_outcome(file_list):
    """
    Step 5: Generate rich outcome metrics (Taxonomy, Sentiment, Embedding).
    """
    print("\n[Step 5] Starting Outcome Evaluation...")
    outcome_analyzer = NarrativeOutcomeAnalyzer(GEMMA_PATH, HF_TOKEN)
    outcome_analyzer.process_files(file_list)
    print("[Step 5] Outcome Evaluation complete.\n")

def step_6_scoring(file_list):
    """
    Step 6: Calculate Weighted Distance Scores and decide the winner.
    """
    print("\n[Step 6] Starting Outcome Scoring...")
    scorer = NarrativeOutcomeScorer()
    scorer.process_files(file_list)
    print("[Step 6] Scoring complete.\n")

def step_7_coref(file_list):
    """
    Step 7: Resolve Coreferences (He -> John) for better Event Extraction.
    """
    print("\n[Step 7] Coreference Resolution...")
    solver = NarrativeCoreferenceSolver(GEMMA_PATH, HF_TOKEN)
    solver.process_files(file_list)
    print("[Step 7] Coreference Resolution complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Narrative Similarity Multi-Step Pipeline")
    
    parser.add_argument("--test", type=str, choices=['yes', 'no'], default='yes', 
                        help="Use test data (amrtestdata) or production data (amrdata). Default: yes")

    parser.add_argument("--step1", action="store_true", help="Run Step 1: MoE Inference")
    parser.add_argument("--step2", action="store_true", help="Run Step 2: Consensus evaluation")
    parser.add_argument("--step3", action="store_true", help="Run Step 3: abstract theme generation")
    parser.add_argument("--step4", action="store_true", help="Run Step 4: Abstract Comparison")
    parser.add_argument("--step5", action="store_true", help="Run Step 5: Outcome Evaluation")
    parser.add_argument("--step6", action="store_true", help="Run Step 6: Outcome Scoring")
    parser.add_argument("--step7", action="store_true", help="Run Step 7. Coreference Resolution")
    parser.add_argument("--all", action="store_true", help="Run all steps sequentially")

    args = parser.parse_args()

    if not (args.step1 or args.step2 or args.step3 or args.step4 or args.step5 or args.step6 or args.step7 or args.all):
        parser.print_help()
        sys.exit(0)

    base_dir = "./amrdatafinal4" if args.test == "no" else "./amrtestdata"

    FILES_1 = [f"{base_dir}/synthetic_data_for_classification.jsonl", f"{base_dir}/dev_track_a.jsonl"]
    FILES_2 = [f"{base_dir}/synthetic_data_for_classification_moe.jsonl", f"{base_dir}/dev_track_a_moe.jsonl"]
    FILES_3 = [f"{base_dir}/synthetic_data_for_classification_moe_output.jsonl", f"{base_dir}/dev_track_a_moe_output.jsonl"]
    FILES_4 = [f"{base_dir}/synthetic_data_for_classification_moe_abstract.jsonl", f"{base_dir}/dev_track_a_moe_abstract.jsonl"]
    FILES_5 = [f"{base_dir}/synthetic_data_for_classification_moe_abstract_output.jsonl", f"{base_dir}/dev_track_a_moe_abstract_output.jsonl"]
    FILES_6 = [f"{base_dir}/synthetic_data_for_classification_moe_abstract_outcome.jsonl", f"{base_dir}/dev_track_a_moe_abstract_outcome.jsonl"]
    FILES_7 = [f"{base_dir}/synthetic_data_for_classification_moe_abstract_outcome_output.jsonl", f"{base_dir}/dev_track_a_moe_abstract_outcome_output.jsonl"]

    if args.step1 or args.all:
        step_1_moe(FILES_1)

    if args.step2 or args.all:
        step_2_consensus(FILES_2)

    if args.step3 or args.all:
        step_3_abstraction(FILES_3)

    if args.step4 or args.all:
        step_4_abstract_comparison(FILES_4)

    if args.step5 or args.all:
        step_5_outcome(FILES_5)
    
    if args.step6 or args.all: 
        step_6_scoring(FILES_6)

    if args.step7 or args.all: 
        step_7_coref(FILES_7)

if __name__ == "__main__":
    main()