#!/usr/bin/env bash
set -euo pipefail

# === Config you might tweak once ===
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ROLE_ARN="${ROLE_ARN:-}" # e.g. arn:aws:iam::<acct>:role/service-role/AmazonSageMaker-ExecutionRole-...
MEM_MB="${MEM_MB:-2048}" # serverless memory
CONCURRENCY="${CONCURRENCY:-1}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") build-push <dockerfile> <ecr-repo> <tag> [context=.]
  $(basename "$0") fix-manifest <ecr-repo> <tag>
  $(basename "$0") verify-manifest <ecr-repo> <tag>
  $(basename "$0") create-model <model-name> <ecr-repo> <tag> <s3-model-tar-url>
  $(basename "$0") create-endpoint <model-name> <endpoint-name> [serverless|m5]
  $(basename "$0") deploy <model-name> <endpoint-name> <ecr-repo> <tag> <s3-model-tar-url> [serverless|m5]
  $(basename "$0") invoke <endpoint-name> '<json-body>'
  $(basename "$0") nuke <endpoint-name> <endpoint-config-name> <model-name>

Env:
  AWS_REGION (default: ap-southeast-2)
  ACCOUNT_ID (autodetected)
  ROLE_ARN   (required for create-model / create-endpoint)
  MEM_MB (serverless, default: ${MEM_MB})  CONCURRENCY (default: ${CONCURRENCY})

Examples:
  # Build/push train image
  $0 build-push docker/sagemaker.Dockerfile duku-lightfm-train v1 .

  # Build/push serve image
  $0 build-push docker/sagemaker_serve.Dockerfile duku-lightfm-infer v1 .

  # Force Docker v2 manifest on an already-pushed tag
  $0 fix-manifest duku-lightfm-infer v1

  # Create model from serve image + model.tar.gz
  export ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-YYYYMMDD
  $0 create-model duku-lightfm duku-lightfm-infer v1 s3://<bucket>/training-output/model.tar.gz

  # Serverless endpoint
  $0 create-endpoint duku-lightfm duku-lightfm-ep serverless

  # Invoke
  $0 invoke duku-lightfm-ep '{"op":"recommend_new_user","liked_ids":[1,2571,1196],"topk":5}'
EOF
}

login_ecr() {
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" >/dev/null
}

ensure_repo() {
  local repo="$1"
  aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$repo" --region "$AWS_REGION" >/dev/null
}

image_uri() {
  local repo="$1" tag="$2"
  echo "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${repo}:${tag}"
}

build_push() {
  local dockerfile="$1" repo="$2" tag="$3" ctx="${4:-.}"
  ensure_repo "$repo"
  login_ecr
  local uri; uri="$(image_uri "$repo" "$tag")"

  # Build amd64. We won't rely on buildx flags for manifest; we'll fix with skopeo.
  docker buildx create --use >/dev/null 2>&1 || true
  docker buildx build \
    --platform linux/amd64 \
    -f "$dockerfile" \
    -t "$uri" \
    "$ctx" \
    --load

  docker push "$uri"
  echo "Pushed: $uri"
  fix_manifest "$repo" "$tag"
  verify_manifest "$repo" "$tag"
}

fix_manifest() {
  local repo="$1" tag="$2"
  local uri; uri="$(image_uri "$repo" "$tag")"
  # skopeo login
  skopeo login \
    --username AWS \
    --password "$(aws ecr get-login-password --region "$AWS_REGION")" \
    "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" >/dev/null

  # Rewrite to Docker v2 schema (v2s2)
  skopeo copy --quiet --format v2s2 "docker://${uri}" "docker://${uri}"
  echo "Rewrote manifest to Docker v2: $uri"
}

verify_manifest() {
  local repo="$1" tag="$2"
  local mt
  mt="$(aws ecr batch-get-image \
        --repository-name "$repo" \
        --image-ids imageTag="$tag" \
        --query 'images[].imageManifestMediaType' \
        --region "$AWS_REGION" \
        --output text)"
  echo "Manifest: $mt"
  if [[ "$mt" != "application/vnd.docker.distribution.manifest.v2+json" ]]; then
    echo "ERROR: manifest is not Docker v2; run fix-manifest." >&2
    exit 2
  fi
}

create_model() {
  local model="$1" repo="$2" tag="$3" s3url="$4"
  [[ -z "${ROLE_ARN}" ]] && { echo "ROLE_ARN is required" >&2; exit 1; }
  local uri; uri="$(image_uri "$repo" "$tag")"
  verify_manifest "$repo" "$tag"

  aws sagemaker delete-model --model-name "$model" >/dev/null 2>&1 || true
  aws sagemaker create-model \
    --model-name "$model" \
    --primary-container "Image=${uri},ModelDataUrl=${s3url}" \
    --execution-role-arn "$ROLE_ARN" \
    --region "$AWS_REGION"
  echo "Created model: $model"
}

create_endpoint() {
  local model="$1" ep="$2" kind="${3:-serverless}"
  local cfg="${model}-cfg"

  aws sagemaker delete-endpoint --endpoint-name "$ep" >/dev/null 2>&1 || true
  aws sagemaker delete-endpoint-config --endpoint-config-name "$cfg" >/dev/null 2>&1 || true

  if [[ "$kind" == "serverless" ]]; then
    aws sagemaker create-endpoint-config \
      --endpoint-config-name "$cfg" \
      --production-variants "[{\"VariantName\":\"AllTraffic\",\"ModelName\":\"${model}\",\"ServerlessConfig\":{\"MemorySizeInMB\":${MEM_MB},\"MaxConcurrency\":${CONCURRENCY}}}]" \
      --region "$AWS_REGION"
  else
    # instance-backed example: ml.m5.large
    aws sagemaker create-endpoint-config \
      --endpoint-config-name "$cfg" \
      --production-variants "[{\"VariantName\":\"AllTraffic\",\"ModelName\":\"${model}\",\"InitialInstanceCount\":1,\"InstanceType\":\"ml.m5.large\"}]" \
      --region "$AWS_REGION"
  fi

  aws sagemaker create-endpoint \
    --endpoint-name "$ep" \
    --endpoint-config-name "$cfg" \
    --region "$AWS_REGION"

  echo "Creating endpoint $ep (waiting for InService)..."
  aws sagemaker wait endpoint-in-service --endpoint-name "$ep" --region "$AWS_REGION"
  echo "Endpoint InService: $ep"
}

invoke_ep() {
  local ep="$1" body="$2"
  aws sagemaker-runtime invoke-endpoint \
    --endpoint-name "$ep" \
    --content-type application/json \
    --body "$body" \
    --region "$AWS_REGION" \
    /dev/stdout | cat
  echo
}

nuke_all() {
  local ep="$1" cfg="$2" model="$3"
  aws sagemaker delete-endpoint --endpoint-name "$ep" --region "$AWS_REGION" >/dev/null 2>&1 || true
  aws sagemaker delete-endpoint-config --endpoint-config-name "$cfg" --region "$AWS_REGION" >/dev/null 2>&1 || true
  aws sagemaker delete-model --model-name "$model" --region "$AWS_REGION" >/dev/null 2>&1 || true
  echo "Deleted endpoint=$ep, config=$cfg, model=$model"
}

deploy_from_s3() {
  local model="$1" ep="$2" repo="$3" tag="$4" s3url="$5" kind="${6:-serverless}"

  create_model "$model" "$repo" "$tag" "$s3url"
  create_endpoint "$model" "$ep" "$kind"
}

cmd="${1:-}"
case "$cmd" in
  build-push)         shift; build_push "$@";;
  fix-manifest)       shift; fix_manifest "$@";;
  verify-manifest)    shift; verify_manifest "$@";;
  create-model)       shift; create_model "$@";;
  create-endpoint)    shift; create_endpoint "$@";;
  deploy)             shift; deploy_from_s3 "$@";;
  invoke)             shift; invoke_ep "$@";;
  nuke)               shift; nuke_all "$@";;
  ""|-h|--help|help)  usage;;
  *)                  echo "Unknown command: $cmd"; usage; exit 1;;
esac
