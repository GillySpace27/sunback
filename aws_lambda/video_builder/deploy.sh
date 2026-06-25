#!/usr/bin/env bash
#
# Deploy the Sun-Right-Now video-builder Lambda + ffmpeg layer + S3 trigger.
# Idempotent: safe to re-run (creates or updates).
#
# Prereqs: awscli v2 configured with credentials that can manage Lambda/IAM/S3,
#          plus curl + tar + zip. Run from anywhere; paths are resolved here.
#
# Usage:   ./deploy.sh                # full deploy
#          SKIP_LAYER=1 ./deploy.sh   # reuse existing ffmpeg layer, code only
#
set -euo pipefail

# ---- Config (override via env) ---------------------------------------------
REGION="${REGION:-us-east-2}"
BUCKET="${SUN_BUCKET:-the-sun-now}"
FUNCTION="${FUNCTION:-sun-video-builder}"
ROLE_NAME="${ROLE_NAME:-sun-video-builder-role}"
LAYER_NAME="${LAYER_NAME:-ffmpeg-static}"
RUNTIME="python3.12"
HANDLER="video_builder.handler.handler"
MEMORY=1024
TIMEOUT=120
EPHEMERAL=1024                       # /tmp size (MB) for frames + mp4
FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
# Lambda env (handler reads these; see handler.py)
ENV_VARS="VIDEO_FPS=18,FRAME_WINDOW=144,INTEGRATION_FRAMES=3,INTEGRATION_METHOD=median,SUN_BUCKET=${BUCKET}"
# ----------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../aws_lambda/video_builder
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
say(){ printf '\n=== %s ===\n' "$*"; }

# ---- 1. IAM role -----------------------------------------------------------
say "IAM role ${ROLE_NAME}"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  cat >"$WORK/trust.json" <<'JSON'
{ "Version": "2012-10-17",
  "Statement": [{ "Effect": "Allow",
    "Principal": { "Service": "lambda.amazonaws.com" },
    "Action": "sts:AssumeRole" }] }
JSON
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://$WORK/trust.json" >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "  created; waiting for propagation..."; sleep 12
else
  echo "  exists."
fi

# S3 access to the bucket (inline, idempotent)
cat >"$WORK/s3policy.json" <<JSON
{ "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:PutObjectAcl","s3:DeleteObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/*" },
    { "Effect": "Allow", "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::${BUCKET}" }
  ] }
JSON
aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name sun-bucket-access --policy-document "file://$WORK/s3policy.json"

# ---- 2. ffmpeg layer -------------------------------------------------------
LAYER_ARG=()
if [[ "${SKIP_LAYER:-0}" != "1" ]]; then
  say "ffmpeg layer ${LAYER_NAME}"
  curl -fsSL "$FFMPEG_URL" -o "$WORK/ffmpeg.tar.xz"
  mkdir -p "$WORK/layer/bin"
  tar -xJf "$WORK/ffmpeg.tar.xz" -C "$WORK" --wildcards '*/ffmpeg'
  cp "$(find "$WORK" -type f -name ffmpeg | head -1)" "$WORK/layer/bin/ffmpeg"
  chmod +x "$WORK/layer/bin/ffmpeg"
  ( cd "$WORK/layer" && zip -qr ../layer.zip bin )
  LAYER_ARN="$(aws lambda publish-layer-version --region "$REGION" \
      --layer-name "$LAYER_NAME" \
      --description "static ffmpeg at /opt/bin/ffmpeg" \
      --compatible-runtimes "$RUNTIME" \
      --zip-file "fileb://$WORK/layer.zip" \
      --query LayerVersionArn --output text)"
  echo "  published $LAYER_ARN"
  LAYER_ARG=(--layers "$LAYER_ARN")
else
  echo "SKIP_LAYER=1: reusing the layer already on the function."
fi

# ---- 3. Package the function code (preserve the package for relative imports)
say "Package code"
mkdir -p "$WORK/pkg/video_builder"
cp "$HERE"/{__init__.py,handler.py,frame_queue.py,manifest.py} "$WORK/pkg/video_builder/"
( cd "$WORK/pkg" && zip -qr ../code.zip video_builder )

# ---- 4. Create or update the function --------------------------------------
say "Lambda ${FUNCTION}"
if aws lambda get-function --region "$REGION" --function-name "$FUNCTION" >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name "$FUNCTION" \
    --zip-file "fileb://$WORK/code.zip" >/dev/null
  aws lambda wait function-updated --region "$REGION" --function-name "$FUNCTION"
  aws lambda update-function-configuration --region "$REGION" --function-name "$FUNCTION" \
    --handler "$HANDLER" --runtime "$RUNTIME" --role "$ROLE_ARN" \
    --memory-size "$MEMORY" --timeout "$TIMEOUT" \
    --ephemeral-storage "Size=$EPHEMERAL" \
    --environment "Variables={$ENV_VARS}" "${LAYER_ARG[@]}" >/dev/null
  echo "  updated."
else
  aws lambda create-function --region "$REGION" --function-name "$FUNCTION" \
    --runtime "$RUNTIME" --handler "$HANDLER" --role "$ROLE_ARN" \
    --memory-size "$MEMORY" --timeout "$TIMEOUT" \
    --ephemeral-storage "Size=$EPHEMERAL" \
    --environment "Variables={$ENV_VARS}" \
    --zip-file "fileb://$WORK/code.zip" "${LAYER_ARG[@]}" >/dev/null
  echo "  created."
fi
aws lambda wait function-updated --region "$REGION" --function-name "$FUNCTION"
FUNC_ARN="$(aws lambda get-function --region "$REGION" --function-name "$FUNCTION" \
            --query Configuration.FunctionArn --output text)"

# ---- 5. Allow S3 to invoke, then wire the bucket notification --------------
say "S3 -> Lambda trigger (prefix 1k/, suffix .png)"
aws lambda add-permission --region "$REGION" --function-name "$FUNCTION" \
  --statement-id s3invoke --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn "arn:aws:s3:::${BUCKET}" \
  --source-account "$ACCOUNT_ID" >/dev/null 2>&1 || echo "  invoke permission already present."

cat >"$WORK/notify.json" <<JSON
{ "LambdaFunctionConfigurations": [{
    "LambdaFunctionArn": "${FUNC_ARN}",
    "Events": ["s3:ObjectCreated:*"],
    "Filter": { "Key": { "FilterRules": [
      { "Name": "prefix", "Value": "1k/" },
      { "Name": "suffix", "Value": ".png" } ] } } }] }
JSON
# NOTE: this REPLACES the bucket's notification config. If the bucket has other
# notifications, merge them into notify.json instead of overwriting.
aws s3api put-bucket-notification-configuration --region "$REGION" \
  --bucket "$BUCKET" --notification-configuration "file://$WORK/notify.json"

say "Done"
echo "Function: $FUNC_ARN"
echo "Smoke test: upload a PNG to s3://${BUCKET}/1k/rhef_171_1k.png and check"
echo "  frames/171/, video/rhef_171_1k.mp4, manifest/171.json + CloudWatch logs."
