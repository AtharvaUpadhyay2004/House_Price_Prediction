# Aim

Goal was to demonstrate practical MLOps best practice, include model eval,
promotion based on performance threshold and deployment readiness via api.

# how to run project

### Install dependencies
pip install -r requirements.txt
### Train model and apply deployment
python train_and_gate.py
### API inference
uvicorn inference.app:app --host 0.0.0.0 --port 8000

# Assumption
- Uses R^2 as the primary gating metric
- Feature inputs at inference must match the training feature schema

# Implementation
- Metric based deployment gate
- Deployment ready API

# Reflection
- Used the AI assistant to faster the development process
- Used to debug in real time situation
- Helps in understanding concepts with better approach