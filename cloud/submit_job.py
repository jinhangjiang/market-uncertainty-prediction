"""
submit_job.py
Submit the EUI prediction pipeline as an AzureML command job.

Usage:
    python cloud/submit_job.py
    python cloud/submit_job.py --dataset reddit
    python cloud/submit_job.py --workspace-config cloud/azure_config.json

Prerequisites:
    pip install azure-ai-ml
    az login  (or set AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID env vars)
"""
import argparse
import os
import sys


def submit(dataset: str = "all", workspace_config: str = None, job_yml: str = "cloud/azureml_job.yml"):
    try:
        from azure.ai.ml import MLClient, load_job
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("ERROR: azure-ai-ml not installed. Run: pip install azure-ai-ml azure-identity")
        sys.exit(1)

    credential = DefaultAzureCredential()

    if workspace_config and os.path.exists(workspace_config):
        import json
        with open(workspace_config) as f:
            ws_cfg = json.load(f)
        ml_client = MLClient(
            credential=credential,
            subscription_id=ws_cfg["subscription_id"],
            resource_group_name=ws_cfg["resource_group"],
            workspace_name=ws_cfg["workspace_name"],
        )
    else:
        try:
            ml_client = MLClient.from_config(credential=credential)
        except Exception:
            print("ERROR: Cannot find AzureML workspace config.")
            print("Create cloud/azure_config.json with subscription_id, resource_group, workspace_name")
            sys.exit(1)

    job = load_job(job_yml)
    job.inputs.dataset = dataset

    print(f"Submitting job: {job.display_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Compute: {job.compute}")
    print(f"  Environment: {job.environment}")

    submitted = ml_client.jobs.create_or_update(job)

    print(f"\nJob submitted!")
    print(f"  Name: {submitted.name}")
    print(f"  Status: {submitted.status}")
    print(f"  Studio URL: {submitted.studio_url}")

    return submitted


def main():
    parser = argparse.ArgumentParser(description="Submit EUI pipeline to AzureML")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "reddit", "stackexchange", "bogleheads"])
    parser.add_argument("--workspace-config", default="cloud/azure_config.json",
                        help="Path to AzureML workspace config JSON")
    parser.add_argument("--job-yml", default="cloud/azureml_job.yml")
    args = parser.parse_args()

    submit(
        dataset=args.dataset,
        workspace_config=args.workspace_config,
        job_yml=args.job_yml,
    )


if __name__ == "__main__":
    main()
