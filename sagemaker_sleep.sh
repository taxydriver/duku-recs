#!/usr/bin/env bash
set -euo pipefail
REGION="${AWS_REGION:-ap-southeast-2}"
echo "Region: $REGION"

# Endpoints
for ep in $(aws sagemaker list-endpoints --region "$REGION" --status InService | jq -r '.Endpoints[].EndpointName'); do
  echo "Deleting endpoint: $ep"
  aws sagemaker delete-endpoint --region "$REGION" --endpoint-name "$ep"
done

# Notebooks (legacy)
for nb in $(aws sagemaker list-notebook-instances --region "$REGION" --status InService | jq -r '.NotebookInstances[].NotebookInstanceName'); do
  echo "Stopping notebook: $nb"
  aws sagemaker stop-notebook-instance --region "$REGION" --notebook-instance-name "$nb"
done

# Training/Processing/Transform
for j in $(aws sagemaker list-training-jobs --region "$REGION" --status-equals InProgress | jq -r '.TrainingJobSummaries[].TrainingJobName'); do
  echo "Stopping training: $j"; aws sagemaker stop-training-job --region "$REGION" --training-job-name "$j"; done
for j in $(aws sagemaker list-processing-jobs --region "$REGION" --status-equals InProgress | jq -r '.ProcessingJobSummaries[].ProcessingJobName'); do
  echo "Stopping processing: $j"; aws sagemaker stop-processing-job --region "$REGION" --processing-job-name "$j"; done
for j in $(aws sagemaker list-transform-jobs --region "$REGION" --status-equals InProgress | jq -r '.TransformJobSummaries[].TransformJobName'); do
  echo "Stopping transform: $j"; aws sagemaker stop-transform-job --region "$REGION" --transform-job-name "$j"; done
echo "✅ Safe to sleep."