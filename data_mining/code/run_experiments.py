"""
Automated Experiment Runner for Multi-Agent Entropy Data Mining Analysis.

This script runs multiple experiments with different parameter combinations
and collects the results for analysis.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures
from threading import Lock


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread lock for safe console output
print_lock = Lock()


def run_single_experiment(config: Dict, experiment_id: int, total_experiments: int) -> Dict:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Dictionary containing experiment configuration
        experiment_id: Current experiment ID for progress tracking
        total_experiments: Total number of experiments to run
    
    Returns:
        Dictionary containing experiment results
    """
    start_time = time.time()
    
    # Safely print progress with thread lock
    with print_lock:
        print(f"[{experiment_id}/{total_experiments}] Running experiment: {config.get('name', 'unnamed')}")
        print(f"  Parameters: {config['params']}")
    
    try:
        # Build command
        cmd = [sys.executable, "main.py"]
        
        # Add parameters from config
        for param, value in config['params'].items():
            if isinstance(value, list):
                # Handle list arguments like --model-name, --dataset, --architecture
                cmd.extend([f"--{param.replace('_', '-')}"])
                cmd.extend([str(v) for v in value])
            elif isinstance(value, bool):
                # Handle boolean flags
                if value:
                    cmd.append(f"--{param.replace('_', '-')}")
            else:
                # Handle single-value arguments
                cmd.extend([f"--{param.replace('_', '-')}", str(value)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.get('timeout', 3600)  # Default 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Collect results
        experiment_result = {
            'experiment_id': experiment_id,
            'name': config.get('name', f'experiment_{experiment_id}'),
            'params': config['params'],
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'status': 'SUCCESS' if result.returncode == 0 else 'FAILED'
        }
        
        # Safely print completion status
        with print_lock:
            status_msg = "✓ SUCCESS" if result.returncode == 0 else "✗ FAILED"
            print(f"[{experiment_id}/{total_experiments}] Completed: {config.get('name', 'unnamed')} - {status_msg} ({duration:.2f}s)")
        
        return experiment_result
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        
        with print_lock:
            print(f"[{experiment_id}/{total_experiments}] TIMEOUT: {config.get('name', 'unnamed')} ({duration:.2f}s)")
        
        return {
            'experiment_id': experiment_id,
            'name': config.get('name', f'experiment_{experiment_id}'),
            'params': config['params'],
            'return_code': -1,
            'stdout': '',
            'stderr': 'Timeout expired',
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'status': 'TIMEOUT'
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        with print_lock:
            print(f"[{experiment_id}/{total_experiments}] ERROR: {config.get('name', 'unnamed')} - {str(e)} ({duration:.2f}s)")
        
        return {
            'experiment_id': experiment_id,
            'name': config.get('name', f'experiment_{experiment_id}'),
            'params': config['params'],
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'status': 'ERROR'
        }


def run_experiments_serial(configurations: List[Dict]) -> List[Dict]:
    """
    Run experiments serially (one after another).
    
    Args:
        configurations: List of experiment configurations
        
    Returns:
        List of experiment results
    """
    results = []
    total_experiments = len(configurations)
    
    print(f"Starting {total_experiments} experiments in SERIAL mode...")
    print("="*80)
    
    for i, config in enumerate(configurations, 1):
        result = run_single_experiment(config, i, total_experiments)
        results.append(result)
    
    return results


def run_experiments_parallel(configurations: List[Dict], max_workers: Optional[int] = None) -> List[Dict]:
    """
    Run experiments in parallel using ThreadPoolExecutor.
    
    Args:
        configurations: List of experiment configurations
        max_workers: Maximum number of parallel workers (default: CPU count)
        
    Returns:
        List of experiment results
    """
    results = []
    total_experiments = len(configurations)
    
    print(f"Starting {total_experiments} experiments in PARALLEL mode (max {max_workers or 'CPU count'} workers)...")
    print("="*80)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_config = {
            executor.submit(run_single_experiment, config, i, total_experiments): (i, config)
            for i, config in enumerate(configurations, 1)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_config):
            result = future.result()
            results.append(result)
            
            # Sort results by experiment_id for consistent ordering
            results.sort(key=lambda x: x['experiment_id'])
    
    return results


def generate_report(results: List[Dict], output_dir: str = "experiment_reports"):
    """
    Generate a comprehensive report of all experiments.
    
    Args:
        results: List of experiment results
        output_dir: Directory to save the report files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON results
    json_filename = output_path / f"experiment_results_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate summary statistics
    total_experiments = len(results)
    successful_experiments = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_experiments = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])
    success_rate = (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0
    
    # Calculate average duration
    successful_durations = [r['duration'] for r in results if r['status'] == 'SUCCESS']
    avg_duration = sum(successful_durations) / len(successful_durations) if successful_durations else 0
    
    # Generate summary report
    summary_lines = [
        "EXPERIMENT SUMMARY REPORT",
        "="*50,
        f"Total Experiments: {total_experiments}",
        f"Successful: {successful_experiments}",
        f"Failed/Errored: {failed_experiments}",
        f"Success Rate: {success_rate:.2f}%",
        f"Average Duration (successful): {avg_duration:.2f}s",
        "",
        "DETAILED RESULTS:",
        "-"*50
    ]
    
    for result in results:
        status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
        summary_lines.append(f"{status_icon} [{result['experiment_id']}] {result['name']} - {result['status']} ({result['duration']:.2f}s)")
        
        if result['status'] != 'SUCCESS':
            summary_lines.append(f"    Error: {result['stderr'][:200]}...")  # Truncate long errors
    
    summary_filename = output_path / f"experiment_summary_{timestamp}.txt"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    # Generate CSV report for easier analysis
    csv_lines = ["experiment_id,name,status,duration,return_code,params,error"]
    for result in results:
        params_str = json.dumps(result['params']).replace('"', '""')  # Escape quotes for CSV
        error_str = result['stderr'].replace('\n', ' ').replace('"', '""')[:100]  # Truncate and escape
        csv_line = f"{result['experiment_id']},{result['name']},{result['status']},{result['duration']:.2f},{result['return_code']},\"{params_str}\",\"{error_str}\""
        csv_lines.append(csv_line)
    
    csv_filename = output_path / f"experiment_results_{timestamp}.csv"
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_lines))
    
    logger.info(f"Reports saved to {output_path}/")
    logger.info(f"- Detailed JSON: {json_filename.name}")
    logger.info(f"- Summary TXT: {summary_filename.name}")
    logger.info(f"- Analysis CSV: {csv_filename.name}")

def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(description="Automated Experiment Runner for Data Mining Analysis")
    parser.add_argument(
        '--config-file',
        type=str,
        default="multiagent-entropy/data_mining/configs/sample_experiment_config.json",
        help="Path to JSON file containing experiment configurations"
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help="Run experiments in parallel mode"
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiment_reports',
        help="Directory to save experiment reports"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what experiments would be run without executing them"
    )
    
    args = parser.parse_args()
    
    # Get experiment configurations
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            logger.error(f"Configuration file {args.config_file} not found")
            sys.exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        # Handle both direct array and object with 'experiment_configs' key
        if isinstance(config_data, list):
            configurations = config_data
        elif isinstance(config_data, dict) and 'experiment_configs' in config_data:
            configurations = config_data['experiment_configs']
        else:
            logger.error(f"Configuration file must contain either an array of configs or an object with 'experiment_configs' key")
            sys.exit(1)
        
        logger.info(f"Loaded {len(configurations)} experiment configurations from {args.config_file}")
    else:
        raise ValueError("No configuration file provided")
    
    if args.dry_run:
        print("DRY RUN MODE - Would run the following experiments:")
        print("="*80)
        for i, config in enumerate(configurations, 1):
            print(f"[{i}] {config['name']}")
            print(f"    Parameters: {config['params']}")
            print(f"    Timeout: {config.get('timeout', 3600)}s")
            print()
        print(f"Total: {len(configurations)} experiments")
        return
    
    logger.info(f"Starting {len(configurations)} experiments")
    
    # Run experiments
    start_time = time.time()
    
    if args.parallel:
        results = run_experiments_parallel(configurations, args.max_workers)
    else:
        results = run_experiments_serial(configurations)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Generate reports
    generate_report(results, args.output_dir)
    
    # Print final summary
    successful_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    total_count = len(results)
    success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
    
    print("")
    print("="*80)
    print("EXPERIMENT BATCH COMPLETED")
    print("="*80)
    print(f"Total Experiments: {total_count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {total_count - successful_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()