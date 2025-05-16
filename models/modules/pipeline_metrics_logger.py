#!/usr/bin/env python
# pipeline_metrics_logger.py

import os
import pandas as pd
import time
import json
import argparse
from datetime import timedelta

class PipelineMetricsLogger:
    """
    Logger class for tracking MOF pipeline metrics like timing and epochs
    for different stages of the pipeline.
    """
    
    def __init__(self, output_dir, dataset_name):
        """
        Initialize the logger with output directory and dataset name
        
        Args:
            output_dir: Directory to save the metrics log
            dataset_name: Name of the dataset being processed
        """
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics dictionary
        self.metrics = {
            "dataset": dataset_name,
            "finetune": {
                "time_seconds": 0,
                "epochs": 0,
                "start_time": None
            },
            "finetune_inference": {
                "time_seconds": 0,
                "generations": 0,
                "start_time": None
            },
            "rl_training": {},  # Will be populated with target-specific data
            "rl_inference": {}  # Will be populated with target-specific data
        }
        
        # Path for the metrics file
        self.metrics_file = os.path.join(output_dir, f"{dataset_name}_pipeline_metrics.json")
        self.summary_file = os.path.join(output_dir, f"{dataset_name}_pipeline_summary.csv")
        
        # Load existing metrics if the file exists
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
            print(f"Loaded existing metrics from {self.metrics_file}")
    
    def start_finetune(self, epochs):
        """Start tracking fine-tuning process"""
        self.metrics["finetune"]["start_time"] = time.time()
        self.metrics["finetune"]["epochs"] = epochs
        self._save_metrics()
        print(f"Started tracking fine-tuning with {epochs} epochs")
    
    def end_finetune(self):
        """End tracking fine-tuning process"""
        if self.metrics["finetune"]["start_time"] is not None:
            elapsed = time.time() - self.metrics["finetune"]["start_time"]
            self.metrics["finetune"]["time_seconds"] = elapsed
            self.metrics["finetune"]["start_time"] = None
            self._save_metrics()
            print(f"Fine-tuning completed in {timedelta(seconds=elapsed)}")
            return elapsed
        return 0
    
    def start_finetune_inference(self, num_generations):
        """Start tracking fine-tuning inference process"""
        self.metrics["finetune_inference"]["start_time"] = time.time()
        self.metrics["finetune_inference"]["generations"] = num_generations
        self._save_metrics()
        print(f"Started tracking fine-tune inference with {num_generations} generations")
    
    def end_finetune_inference(self):
        """End tracking fine-tuning inference process"""
        if self.metrics["finetune_inference"]["start_time"] is not None:
            elapsed = time.time() - self.metrics["finetune_inference"]["start_time"]
            self.metrics["finetune_inference"]["time_seconds"] = elapsed
            self.metrics["finetune_inference"]["start_time"] = None
            self._save_metrics()
            print(f"Fine-tune inference completed in {timedelta(seconds=elapsed)}")
            return elapsed
        return 0
    
    def start_rl_training(self, target_name, epochs):
        """
        Start tracking RL training for a specific target
        
        Args:
            target_name: Name of the target (e.g., 'mean', 'mean_plus_1std')
            epochs: Number of epochs planned for RL training
        """
        if target_name not in self.metrics["rl_training"]:
            self.metrics["rl_training"][target_name] = {
                "time_seconds": 0,
                "epochs": epochs,
                "start_time": None
            }
        
        self.metrics["rl_training"][target_name]["start_time"] = time.time()
        self.metrics["rl_training"][target_name]["epochs"] = epochs
        self._save_metrics()
        print(f"Started tracking RL training for target '{target_name}' with {epochs} epochs")
    
    def end_rl_training(self, target_name):
        """End tracking RL training for a specific target"""
        if target_name in self.metrics["rl_training"] and \
           self.metrics["rl_training"][target_name]["start_time"] is not None:
            elapsed = time.time() - self.metrics["rl_training"][target_name]["start_time"]
            self.metrics["rl_training"][target_name]["time_seconds"] = elapsed
            self.metrics["rl_training"][target_name]["start_time"] = None
            self._save_metrics()
            print(f"RL training for target '{target_name}' completed in {timedelta(seconds=elapsed)}")
            return elapsed
        return 0
    
    def start_rl_inference(self, target_name, num_generations):
        """
        Start tracking RL inference for a specific target
        
        Args:
            target_name: Name of the target (e.g., 'mean', 'mean_plus_1std')
            num_generations: Number of MOF generations planned
        """
        if target_name not in self.metrics["rl_inference"]:
            self.metrics["rl_inference"][target_name] = {
                "time_seconds": 0,
                "generations": num_generations,
                "total_attempts": 0,
                "start_time": None
            }
        
        self.metrics["rl_inference"][target_name]["start_time"] = time.time()
        self.metrics["rl_inference"][target_name]["generations"] = num_generations
        self._save_metrics()
        print(f"Started tracking RL inference for target '{target_name}' with {num_generations} generations")
    
    def end_rl_inference(self, target_name, total_attempts=None):
        """
        End tracking RL inference for a specific target
        
        Args:
            target_name: Name of the target
            total_attempts: Total number of MOF generation attempts (if available)
        """
        if target_name in self.metrics["rl_inference"] and \
           self.metrics["rl_inference"][target_name]["start_time"] is not None:
            elapsed = time.time() - self.metrics["rl_inference"][target_name]["start_time"]
            self.metrics["rl_inference"][target_name]["time_seconds"] = elapsed
            
            if total_attempts is not None:
                self.metrics["rl_inference"][target_name]["total_attempts"] = total_attempts
                
            self.metrics["rl_inference"][target_name]["start_time"] = None
            self._save_metrics()
            print(f"RL inference for target '{target_name}' completed in {timedelta(seconds=elapsed)}")
            
            if total_attempts is not None:
                efficiency = self.metrics["rl_inference"][target_name]["generations"] / total_attempts * 100
                print(f"Efficiency: {efficiency:.1f}% ({self.metrics['rl_inference'][target_name]['generations']} / {total_attempts})")
                
            return elapsed
        return 0
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            # Create a clean copy without any possible None values that aren't JSON serializable
            clean_metrics = self._clean_metrics(self.metrics)
            json.dump(clean_metrics, f, indent=2)
    
    def _clean_metrics(self, metrics_dict):
        """Create a clean copy of metrics without None values for JSON serialization"""
        clean = {}
        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                clean[k] = self._clean_metrics(v)
            elif v is not None:
                clean[k] = v
            else:
                clean[k] = "null"  # Convert None to string "null"
        return clean
    
    def generate_summary_table(self):
        """
        Generate a summary table in CSV format showing all metrics
        in a flattened, easy-to-read format
        """
        summary_data = []
        
        # Add finetune row
        finetune_time_secs = self.metrics["finetune"]["time_seconds"]
        summary_data.append({
            "Process": "Fine-tuning",
            "Target": "N/A",
            "Time (seconds)": round(finetune_time_secs, 2),
            "Epochs": self.metrics["finetune"]["epochs"],
            "Generations": "N/A",
            "Total Attempts": "N/A",
            "Efficiency (%)": "N/A"
        })
        
        # Add finetune inference row
        finetune_inf_time_secs = self.metrics["finetune_inference"]["time_seconds"]
        summary_data.append({
            "Process": "Fine-tune Inference",
            "Target": "N/A",
            "Time (seconds)": round(finetune_inf_time_secs, 2),
            "Epochs": "N/A",
            "Generations": self.metrics["finetune_inference"]["generations"],
            "Total Attempts": "N/A",
            "Efficiency (%)": "N/A"
        })
        
        # Add RL training rows for each target
        for target, target_data in self.metrics["rl_training"].items():
            rl_train_time_secs = target_data["time_seconds"]
            summary_data.append({
                "Process": "RL Training",
                "Target": target,
                "Time (seconds)": round(rl_train_time_secs, 2),
                "Epochs": target_data["epochs"],
                "Generations": "N/A",
                "Total Attempts": "N/A",
                "Efficiency (%)": "N/A"
            })
        
        # Add RL inference rows for each target
        for target, target_data in self.metrics["rl_inference"].items():
            rl_inf_time_secs = target_data["time_seconds"] 
            generations = target_data["generations"]
            total_attempts = target_data.get("total_attempts", "N/A")
            
            # Calculate efficiency if possible
            if isinstance(total_attempts, (int, float)) and total_attempts > 0:
                efficiency = round(generations / total_attempts * 100, 1)
            else:
                efficiency = "N/A"
                
            summary_data.append({
                "Process": "RL Inference",
                "Target": target,
                "Time (seconds)": round(rl_inf_time_secs, 2),
                "Epochs": "N/A",
                "Generations": generations,
                "Total Attempts": total_attempts,
                "Efficiency (%)": efficiency
            })
        
        # Create dataframe and save
        df = pd.DataFrame(summary_data)
        df.to_csv(self.summary_file, index=False)
        print(f"Summary table saved to {self.summary_file}")
        
        # Also print the table to console
        print("\n" + "="*100)
        print(f"PIPELINE METRICS SUMMARY FOR {self.dataset_name}")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
        
        return df

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MOF Pipeline Metrics Logger")
    parser.add_argument("--output-dir", type=str, default="./metrics",
                       help="Directory to save metrics")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("--cmd", choices=["start_finetune", "end_finetune", 
                                         "start_finetune_inference", "end_finetune_inference",
                                         "start_rl_training", "end_rl_training",
                                         "start_rl_inference", "end_rl_inference",
                                         "generate_summary"],
                       required=True, help="Command to execute")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (for start_finetune or start_rl_training)")
    parser.add_argument("--generations", type=int, default=None,
                       help="Number of generations (for start_finetune_inference or start_rl_inference)")
    parser.add_argument("--target", type=str, default=None,
                       help="Target name (for RL commands)")
    parser.add_argument("--total-attempts", type=int, default=None,
                       help="Total attempts (for end_rl_inference)")
    
    return parser.parse_args()

def main():
    """Main function to run the logger from command line"""
    args = parse_args()
    
    # Initialize logger
    logger = PipelineMetricsLogger(args.output_dir, args.dataset)
    
    # Execute the requested command
    if args.cmd == "start_finetune":
        if args.epochs is None:
            print("Error: --epochs is required for start_finetune")
            return 1
        logger.start_finetune(args.epochs)
        
    elif args.cmd == "end_finetune":
        logger.end_finetune()
        
    elif args.cmd == "start_finetune_inference":
        if args.generations is None:
            print("Error: --generations is required for start_finetune_inference")
            return 1
        logger.start_finetune_inference(args.generations)
        
    elif args.cmd == "end_finetune_inference":
        logger.end_finetune_inference()
        
    elif args.cmd == "start_rl_training":
        if args.target is None:
            print("Error: --target is required for start_rl_training")
            return 1
        if args.epochs is None:
            print("Error: --epochs is required for start_rl_training")
            return 1
        logger.start_rl_training(args.target, args.epochs)
        
    elif args.cmd == "end_rl_training":
        if args.target is None:
            print("Error: --target is required for end_rl_training")
            return 1
        logger.end_rl_training(args.target)
        
    elif args.cmd == "start_rl_inference":
        if args.target is None:
            print("Error: --target is required for start_rl_inference")
            return 1
        if args.generations is None:
            print("Error: --generations is required for start_rl_inference")
            return 1
        logger.start_rl_inference(args.target, args.generations)
        
    elif args.cmd == "end_rl_inference":
        if args.target is None:
            print("Error: --target is required for end_rl_inference")
            return 1
        logger.end_rl_inference(args.target, args.total_attempts)
        
    elif args.cmd == "generate_summary":
        logger.generate_summary_table()
    
    return 0

if __name__ == "__main__":
    main()