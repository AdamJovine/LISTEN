"""
Script to run AutoPDL prompt optimization experiment.

Usage:
    python run_autopdl_experiment.py --data input/data.csv --output autopdl_results/
"""

import argparse
from autopdl_optimizer import PromptOptimizer


def main():
    parser = argparse.ArgumentParser(description="Run AutoPDL prompt optimization")
    parser.add_argument("--data", required=True, help="Path to schedule CSV")
    parser.add_argument("--output", default="autopdl_results", help="Output directory")
    parser.add_argument(
        "--preferences",
        nargs="+",
        default=[
            "I prefer schedules with fewer conflicts",
            "Minimize evening/morning back-to-back sessions",
        ],
        help="List of preference statements",
    )

    args = parser.parse_args()

    # Setup and run optimization
    optimizer = PromptOptimizer(autopdl_dir=args.output)
    optimizer.setup_autopdl_experiment(args.data, args.preferences)

    print("Starting AutoPDL optimization...")
    optimized_prompt = optimizer.run_optimization()

    if optimized_prompt:
        print(f"Success! Optimized prompt saved in {args.output}")
        print("You can now use this in your dueling bandit experiments.")
    else:
        print("Optimization failed. Check logs for details.")


if __name__ == "__main__":
    main()
