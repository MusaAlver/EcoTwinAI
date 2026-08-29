#!/usr/bin/env bash
set -euo pipefail

# EcoTwin AI - Azure Container Apps deployment
# Run this script from the repository root after `az login`.

RESOURCE_GROUP="${RESOURCE_GROUP:-ecotwin-ai-rg}"
LOCATION="${LOCATION:-westeurope}"
ENVIRONMENT="${ENVIRONMENT:-ecotwin-ai-env}"
APP_NAME="${APP_NAME:-ecotwin-ai}"

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI (az) is not installed."
  echo "On macOS: brew update && brew install azure-cli"
  exit 1
fi

if ! az account show >/dev/null 2>&1; then
  echo "No active Azure session. Run: az login"
  exit 1
fi

echo "Updating Azure Container Apps CLI extension..."
az extension add --name containerapp --upgrade --yes >/dev/null

echo "Registering required Azure resource providers..."
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

echo "Deploying EcoTwin AI to Azure Container Apps..."
az containerapp up \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --environment "$ENVIRONMENT" \
  --source . \
  --ingress external \
  --target-port 8000

FQDN="$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"

echo
echo "EcoTwin AI deployment completed."
echo "API:    https://${FQDN}"
echo "Docs:   https://${FQDN}/docs"
echo "Health: https://${FQDN}/health"
echo

echo "Checking health endpoint..."
curl --fail --silent --show-error "https://${FQDN}/health" || {
  echo
  echo "Deployment exists, but the health check did not succeed yet."
  echo "Inspect logs with:"
  echo "az containerapp logs show --name $APP_NAME --resource-group $RESOURCE_GROUP --follow"
  exit 1
}

echo
