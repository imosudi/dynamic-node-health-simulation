# Dynamic Node Health Simulation

A Modular Python Simulation Framework for Dynamic Node Health Scoring and Probabilistic Fault Injection in Fog Computing Networks. 

## Overview
This project simulates node health degradation in a hierarchical fog computing network deployed in a smart agricultural farm. It models behaviour metrics like Packet Loss Rate (PLR), CPU Usage, and Response Time (RTT), introducing dynamic fault injection, adaptive health thresholding, simulation control, and live visualization. Exports logs in JSON/CSV for analysis using Exponential Moving Average (EMA). 

## Features
- Node-specific behaviour metric profiles (PLR, CPU, RTT)
- Time-varying probabilistic fault injection (stress scenarios)
- Adaptive thresholding with EMA
- Health status classification (Good, Fair, Poor, Faulty)
- CLI-based simulation execution
- Live visualisation of node health vs thresholds (per node subplots)
- Persistent log export (CSV & JSON)

## Project Structure
```
dynamic-node-health-simulation/
├── data                                 --> Auto-generated node_list.csv
│   ├── fault_templates.yaml
│   ├── fault_templates_zero.yaml
│   └── templates.yaml
├── logs
│   ├── csv_writer.py
│   └── __init__.py
├── modules                             --> Core simulation modules
│   ├── fault_injector.py
│   ├── health_classifier.py
│   ├── node__data_generator.py
│   ├── node_operations
│   │   ├── logic.py
│   │   ├── node_generators.py
│   │   └── node_id_extract.py
│   ├── node_profiler.py
│   └── simulation_controller.py
├── output
├── main.py                             --> CLI Entry Point
├── LICENSE
└── README.md


```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/dynamic-node-health-simulation.git
cd dynamic-node-health-simulation
```

### 2. Setup Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Simulation
```bash
python main.py
```

### 4. Output
- Logs: `logs/node_health_log.json` and `logs/node_health_log.csv`
- Visual: `logs/simulation_plot.png`

## Configuration Notes
- The simulation autogenerates sample node profiles if `data/node_list.csv` is missing.
- Behaviour metric profiles can be tailored in `sample_data_generator.py` or loaded from custom CSVs.
- Fault scenarios and probabilities will be configurable via YAML templates in future versions.

## Use Cases
- Fog/Edge network resilience testing
- CPS anomaly detection simulations
- Smart agriculture network performance modelling
- Academic research in fault-tolerant distributed systems

## Future Enhancements
- Scenario-based YAML configurations
- Layer-specific fault models (energy depletion, load stress)
- Integration with real-time sensor emulators
