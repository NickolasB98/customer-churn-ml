"""
Model Promotion Script - Automatically promote the best model to production.

This script:
1. Reads all MLflow runs from the experiment
2. Finds the best run by recall (or other metric)
3. Copies it to artifacts/production_model/
4. Creates a PRODUCTION_MODEL_INFO.json with metadata
5. Optionally commits to git
"""

import os
import json
import shutil
import argparse
import glob
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
EXPERIMENT_NAME = "Telco Churn"
PRODUCTION_DIR = Path("artifacts/production_model")
METRIC_TO_OPTIMIZE = "recall"  # Can be: recall, f1, roc_auc, accuracy


def find_best_run(experiment_name: str, metric: str = "recall") -> dict:
    """
    Find the best run from an MLflow experiment.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize (higher is better)

    Returns:
        Dictionary with run_id, metrics, and params
    """
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    print(f"🔍 Searching experiment: {experiment_name}")

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=100
    )

    if not runs:
        raise ValueError("No runs found in experiment")

    # Get best run
    best_run = runs[0]

    print(f"\n✅ Best Run Found!")
    print(f"   Run ID: {best_run.info.run_id}")
    print(f"   {metric.capitalize()}: {best_run.data.metrics.get(metric, 'N/A'):.4f}")
    print(f"   Timestamp: {best_run.info.start_time}")

    return {
        "run_id": best_run.info.run_id,
        "metrics": dict(best_run.data.metrics),
        "params": dict(best_run.data.params),
    }


def promote_model(run_id: str, destination: Path) -> None:
    """
    Copy best model from MLflow to production directory.

    Args:
        run_id: MLflow run ID
        destination: Where to copy the model
    """
    # Find model artifacts in MLflow Model Registry
    model_artifacts = glob.glob(f"mlruns/*/models/*/artifacts/")

    if not model_artifacts:
        raise ValueError("No model artifacts found in MLflow registry")

    # Use the latest model (usually only one exists for simple projects)
    source = model_artifacts[0]
    print(f"📍 Found model at: {source}")

    # Also collect run artifacts (feature_columns.txt, preprocessing.pkl)
    run_artifacts = glob.glob(f"mlruns/*/*/{run_id}/artifacts/")
    if not run_artifacts:
        run_artifacts = glob.glob(f"mlruns/*/*/*/{run_id}/artifacts/")

    # Remove old production model if exists
    if destination.exists():
        print(f"🔄 Removing old production model...")
        shutil.rmtree(destination)

    # Copy model artifacts
    print(f"📦 Copying model to {destination}...")
    shutil.copytree(source, destination)

    # Copy run-specific artifacts if they exist
    if run_artifacts:
        run_artifact_dir = run_artifacts[0]
        for file in os.listdir(run_artifact_dir):
            src_file = os.path.join(run_artifact_dir, file)
            dst_file = destination / file
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"   ✅ Copied {file}")

    print(f"✅ Model promoted to: {destination}")


def create_metadata(run_id: str, best_run: dict, destination: Path) -> None:
    """
    Create metadata file tracking which run is in production.

    Args:
        run_id: MLflow run ID
        best_run: Dictionary with metrics and params
        destination: Where metadata is stored
    """
    metadata = {
        "run_id": run_id,
        "experiment": EXPERIMENT_NAME,
        "metric_optimized": METRIC_TO_OPTIMIZE,
        "metrics": best_run["metrics"],
        "params": best_run["params"],
    }

    metadata_file = destination.parent / "PRODUCTION_MODEL_INFO.json"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📝 Metadata saved to: {metadata_file}")
    print(json.dumps(metadata, indent=2))


def main(args):
    """Main promotion workflow."""

    print("=" * 70)
    print("🚀 MODEL PROMOTION PIPELINE")
    print("=" * 70)

    # Step 1: Find best run
    print("\nStep 1: Finding best run...")
    best_run = find_best_run(EXPERIMENT_NAME, args.metric)

    # Step 2: Promote model
    print("\nStep 2: Promoting model to production...")
    PRODUCTION_DIR.parent.mkdir(parents=True, exist_ok=True)
    promote_model(best_run["run_id"], PRODUCTION_DIR)

    # Step 3: Create metadata
    print("\nStep 3: Creating metadata...")
    create_metadata(best_run["run_id"], best_run, PRODUCTION_DIR)

    # Step 4: Optional git commit
    if args.commit:
        print("\nStep 4: Committing to git...")
        os.system("git add artifacts/production_model/ artifacts/PRODUCTION_MODEL_INFO.json")
        commit_msg = f"Promote model: {best_run['run_id'][:8]} (Recall: {best_run['metrics'].get('recall', 0):.4f})"
        os.system(f'git commit -m "{commit_msg}"')
        print(f"✅ Committed: {commit_msg}")

    print("\n" + "=" * 70)
    print("✅ MODEL PROMOTION COMPLETE!")
    print("=" * 70)
    print(f"\n📍 Production model location: {PRODUCTION_DIR}")
    print(f"📊 Metrics: {best_run['metrics']}")
    print(f"\n💡 Update inference.py to use: {PRODUCTION_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote best MLflow model to production")
    parser.add_argument(
        "--metric",
        type=str,
        default="recall",
        choices=["recall", "f1", "roc_auc", "accuracy"],
        help="Metric to optimize (find best run by this metric)"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit promoted model to git"
    )

    args = parser.parse_args()
    main(args)
