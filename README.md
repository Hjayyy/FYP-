# final_year_project

# A Real-Time Simulation-Based Decision-Support System for Organ Transport Monitoring, Risk Prediction and Routing Optimisation

## Project Overview
This project is a simulation-based, real-time organ transport monitoring and decision-support system. It combines transport simulation, temperature monitoring, rule-based risk evaluation, machine learning-based escalation prediction, dynamic rerouting, and a live dashboard interface.

## Technologies Used
- Python
- FastAPI
- WebSockets
- JavaScript
- Chart.js
- Leaflet
- scikit-learn
- NumPy

## Project Structure
- app.py - FastAPI server and WebSocket communication
- simulation_engine.py - main simulation controller
- transport_simulator.py - vehicle movement, delays, and traffic states
- temperature_simulator.py - organ temperature simulation
- risk_engine.py - rule-based risk evaluation
- anomaly_detector.py - EWMA anomaly detection
- routing_engine.py - dynamic hospital routing logic
- road_network.py - risk-aware A* pathfinding
- ml_predictor.py - runtime escalation probability prediction
- ml_features.py - feature generation for machine learning
- ml_model.py - Logistic Regression baseline model
- ml_train.py - machine learning model training
- ml_build_dataset.py - dataset generation from simulation logs
- generate_training_logs.py - simulation log generation for training
- simulation_logger.py - simulation data logging
- models.py - shared data structures
- organ_profile.py - organ constraints and parameters
- index.html - dashboard interface

## Required Files
Download all the code
The following trained models are required to run the dashboard:
- heart_gboost.pkl
- kidney_gboost.pkl
- lungs_gboost.pkl

## Installation
Install the required Python packages first:

```bash
pip install fastapi uvicorn joblib numpy scikit-learn

## Running Dashboard
uvicorn app:app --reload   or  python -m uvicorn app:app --reload

## Go to Browser using this link:
http://127.0.0.1:8000

Using the System
1. Start the server
2. Open the dashboard in a browser
3. Select an organ type, start hospital, and destination hospital
4. Start the simulation
5. Observe live temperature, risk level, escalation probability, traffic state, and rerouting behaviour

Notes
The system uses synthetic data only.
This is a proof-of-concept prototype and is not intended for live clinical deployment.
The dashboard requires the three *_gboost.pkl model files to be in the same folder as app.py.
