"""Main entry point for the LLM Fine-tuning Pipeline.

End-to-end LoRA/QLoRA fine-tuning pipeline that handles:
1. Configuration loading
2. Dataset preparation
3. Model setup with quantization and LoRA
4. Training with SFTTrainer
5. Evaluation and benchmarking

Usage:
    python main.py --dry-run
    python main.py
"""
import argparse
import sys
from pathlib import Path

from src.config_loader import load_config
from src.data_preparation import create_sample_dataset, load_and_prepare_dataset
from src.model_setup import setup_model_and_tokenizer
from src.trainer import train
from src.evaluator import evaluate_model, save_eval_results

def main():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Pipeline")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--create-dataset", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  LLM Fine-tuning Pipeline (LoRA/QLoRA)")
    print("=" * 60)

    config = load_config(args.config)
    print(f"Model: {config.model.name}, Method: {'QLoRA' if config.quantization.enabled else 'LoRA'}")

    if args.create_dataset:
        create_sample_dataset(f"data/{config.dataset.name}")
        return

    dataset = load_and_prepare_dataset(config.dataset)

    if args.dry_run:
        setup_model_and_tokenizer(config, dry_run=True)
        train(config, dry_run=True)
        eval_results = evaluate_model(config, dry_run=True)
        save_eval_results(eval_results)
        print("Dry run complete!")
        return

    model, tokenizer, peft_config = setup_model_and_tokenizer(config)
    dataset = load_and_prepare_dataset(config.dataset, tokenizer)
    train_metrics = train(config, model, tokenizer, dataset)
    eval_results = evaluate_model(config, model, tokenizer, dataset["eval"])
    save_eval_results(eval_results)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
