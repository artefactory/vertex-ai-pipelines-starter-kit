#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

set +u
if [[ -n "$GCP_PROJECT_ID" ]]; then
  echo "GCP_PROJECT_ID: ${GCP_PROJECT_ID}"
else
  echo "Env variable GCP_PROJECT_ID not defined. Exiting script.";
  exit 1
fi

if [[ -n "$IMAGE_NAME" ]]; then
  echo "Base Image name: ${IMAGE_NAME}"
  config_path=$(pwd)/components/base_images/${IMAGE_NAME}/cloudbuild.yaml
else
  echo "Env variable IMAGE_NAME not defined. Exiting script.";
  exit 1
fi

if [[ -z "$IMAGE_TAG" ]]; then
    echo "Warning: Env variable 'IMAGE_TAG' is not defined"
    echo "IMAGE_TAG will be set to default : latest"
    image_tag=latest
else
    echo "Image tag used: ${IMAGE_TAG}" 
    image_tag=${IMAGE_TAG}
fi
set -u 

gcloud builds submit --config=${config_path} \
  --substitutions=_IMAGE_NAME=${IMAGE_NAME},_IMAGE_TAG=${image_tag},_GCP_PROJECT_ID=${GCP_PROJECT_ID}
