# EcoTwin AI — Microsoft Azure Deployment

EcoTwin AI can be deployed as a containerized FastAPI service on **Azure Container Apps**. This deployment keeps the existing ML and backend implementation unchanged: Azure acts as the cloud runtime for the existing Docker image.

## Architecture

```text
GitHub / local source
        ↓
Dockerfile
        ↓
Azure cloud build / container registry
        ↓
Azure Container Apps
        ↓
EcoTwin FastAPI + ML engine
        ↓
/health  /status  /forecast  /outcome  /incidents  /docs
```

Real IoT sensors and Azure IoT Hub are intentionally outside the scope of the current prototype and remain future work.

## Prerequisites

- An active Microsoft Azure subscription
- Azure CLI
- The repository cloned locally

On macOS:

```bash
brew update && brew install azure-cli
az login
```

## Deploy

Checkout the deployment branch and run the deployment helper from the repository root:

```bash
git checkout azure-deployment
chmod +x deploy/azure/deploy.sh
./deploy/azure/deploy.sh
```

Default resources:

```text
Resource group: ecotwin-ai-rg
Region:         westeurope
Environment:    ecotwin-ai-env
Container app:  ecotwin-ai
Target port:    8000
Ingress:        external HTTPS
```

The script uses `az containerapp up --source .`. Azure builds the container from the repository Dockerfile and deploys the resulting image to Azure Container Apps.

After deployment, the script prints URLs similar to:

```text
https://<generated-fqdn>/
https://<generated-fqdn>/docs
https://<generated-fqdn>/health
```

## Verify

The health endpoint should return an initialized EcoTwin engine:

```bash
curl https://<generated-fqdn>/health
```

The interactive FastAPI interface is available at:

```text
https://<generated-fqdn>/docs
```

## Logs

```bash
az containerapp logs show \
  --name ecotwin-ai \
  --resource-group ecotwin-ai-rg \
  --follow
```

## Remove Azure resources

Only run this when you intentionally want to delete the complete EcoTwin Azure deployment:

```bash
az group delete --name ecotwin-ai-rg
```

## Project scope statement

For the current prototype, Azure is used as the **cloud deployment layer** for the EcoTwin AI backend and ML runtime. Real sensor ingestion, IoT Hub integration, and a production end-user dashboard are future extensions rather than claimed current capabilities.
