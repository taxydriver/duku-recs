#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# REQUIRED ENV VARS (set these before running)
#   AWS_REGION              e.g. ap-southeast-2
#   ENDPOINT_NAME           e.g. duku-lightfm-ep
#
# CHOOSE ONE MODE:
#   Mode A: Recreate from an existing EndpointConfig
#     ENDPOINT_CONFIG_NAME  e.g. duku-lightfm-ep-config
#
#   Mode B: Create a new EndpointConfig from a Model (uncomment one block: real-time OR serverless)
#     MODEL_NAME            e.g. duku-lightfm-model
#     # For real-time:
#     # INSTANCE_TYPE       e.g. ml.m5.large
#     # INITIAL_INSTANCE_COUNT e.g. 1
#
#     # For serverless:
#     # SERVERLESS_MAX_CONCURRENCY e.g. 5
#     # SERVERLESS_MEMORY_SIZE     e.g. 4096
# ---------------------------

REGION="${AWS_REGION:-ap-southeast-2}"
EP_NAME="${ENDPOINT_NAME:?ENDPOINT_NAME not set}"

echo "🔧 Region: $REGION"
echo "🔧 Endpoint name: $EP_NAME"

# Helpers
describe_ep() {
  aws sagemaker describe-endpoint --region "$REGION" --endpoint-name "$EP_NAME" 2>/dev/null || true
}

wait_inservice() {
  echo "⏳ Waiting for endpoint to be InService..."
  aws sagemaker wait endpoint-in-service --region "$REGION" --endpoint-name "$EP_NAME"
  aws sagemaker describe-endpoint --region "$REGION" --endpoint-name "$EP_NAME" \
    | jq '{EndpointName,EndpointStatus,CreationTime,LastModifiedTime,ProductionVariants}'
}

# ---------------------------
# If endpoint already exists, just show status and exit
# ---------------------------
EXISTING="$(describe_ep)"
if [[ -n "$EXISTING" && "$(echo "$EXISTING" | jq -r .EndpointStatus)" != "Deleted" ]]; then
  STATUS="$(echo "$EXISTING" | jq -r .EndpointStatus)"
  echo "✅ Endpoint already exists: status = $STATUS"
  if [[ "$STATUS" != "InService" ]]; then
    wait_inservice
  else
    echo "✨ Ready."
  fi
  exit 0
fi

# ---------------------------
# Create endpoint (choose Mode A or Mode B)
# ---------------------------

if [[ -n "${ENDPOINT_CONFIG_NAME:-}" ]]; then
  # -------- Mode A: use existing endpoint config ----------
  EP_CONFIG="$ENDPOINT_CONFIG_NAME"
  echo "🔁 Creating endpoint from existing EndpointConfig: $EP_CONFIG"
  aws sagemaker create-endpoint \
    --region "$REGION" \
    --endpoint-name "$EP_NAME" \
    --endpoint-config-name "$EP_CONFIG" >/dev/null

  wait_inservice
  exit 0
fi

# -------- Mode B: create a new endpoint config from a model ----------
MODEL="${MODEL_NAME:?MODEL_NAME not set}"

# Pick ONE of the following blocks:

# --- Real-time inference (uncomment to use) ---
# INSTANCE="${INSTANCE_TYPE:?INSTANCE_TYPE not set}"
# COUNT="${INITIAL_INSTANCE_COUNT:-1}"
# EP_CONFIG="${EP_NAME}-$(date +%Y%m%d%H%M%S)"
# echo "🧱 Creating EndpointConfig (real-time): $EP_CONFIG"
# aws sagemaker create-endpoint-config \
#   --region "$REGION" \
#   --endpoint-config-name "$EP_CONFIG" \
#   --production-variants "VariantName=AllTraffic,ModelName=$MODEL,InitialInstanceCount=$COUNT,InstanceType=$INSTANCE,InitialVariantWeight=1.0" >/dev/null

# --- Serverless inference (uncomment to use) ---
# MAX_CONC="${SERVERLESS_MAX_CONCURRENCY:?SERVERLESS_MAX_CONCURRENCY not set}"
# MEM_MB="${SERVERLESS_MEMORY_SIZE:?SERVERLESS_MEMORY_SIZE not set}"
# EP_CONFIG="${EP_NAME}-$(date +%Y%m%d%H%M%S)-svl"
# echo "🧱 Creating EndpointConfig (serverless): $EP_CONFIG"
# aws sagemaker create-endpoint-config \
#   --region "$REGION" \
#   --endpoint-config-name "$EP_CONFIG" \
#   --serverless-config "MaxConcurrency=$MAX_CONC,MemorySizeInMB=$MEM_MB" \
#   --production-variants "VariantName=AllTraffic,ModelName=$MODEL,InitialVariantWeight=1.0,ServerlessConfig={MaxConcurrency=$MAX_CONC,MemorySizeInMB=$MEM_MB}" >/dev/null

# ---- Create endpoint from the new config ----
echo "🚀 Creating endpoint: $EP_NAME from $EP_CONFIG"
aws sagemaker create-endpoint \
  --region "$REGION" \
  --endpoint-name "$EP_NAME" \
  --endpoint-config-name "$EP_CONFIG" >/dev/null

wait_inservice