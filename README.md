# Adaptive AI-Based Edge-Fog-Cloud IoT Framework

Dataset analysis and an exploratory Random Forest experiment for smart-city
indicators, followed by a proxy-based Edge/Fog/Cloud placement simulation.
The placement output is simulated from documented assumptions; it is not
measured infrastructure performance or a trained placement policy.

## Requirements

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Dataset

The dataset is not included. Provide a CSV containing numeric values for:

- `Smart_Mobility`
- `Smart_Environment`
- `Smart_Government`
- `Smart_Economy`
- `Smart_People`
- `Smart_Living`
- `SmartCity_Index`

`Timestamp` is optional. Rows without a numeric `SmartCity_Index` are omitted
from model analysis. The regression pipeline imputes missing indicators from
its training split; the separate placement simulation uses each indicator's
dataset median before normalizing its proxies.

## Run

```bash
python Adaptive_IoT_Dataset_Analysis.py --dataset path/to/dataset.csv --output-dir results --no-show
```

The dataset defaults to `dataset.csv` beside the script, and output defaults
to the current working directory. Use `--help` to see all options. The script
validates the required columns and minimum row count before analysis.

## Edge/Fog/Cloud placement simulation

The placement demonstration derives normalized urgency, compute, and privacy
proxies from the smart-city pillars, then scores each candidate with the
documented weights: latency (0.30), resource use (0.20), security (0.20),
privacy cost (0.15), and resilience (0.15). Latency, resource use, and privacy
are treated as costs; security and resilience are treated as benefits.

Candidate profiles encode illustrative assumptions: Edge favors low latency
and data locality, Fog is an intermediate option, and Cloud favors resource
capacity, security, and resilience but has higher latency and privacy cost.
The profiles and weights are heuristic placeholders, not learned parameters
or measurements. The dataset contains no infrastructure telemetry, workload
placement labels, or validated privacy/security measures. Consequently,
`Selected_Layer` is only a scenario simulation; it must not be presented as a
real deployment recommendation or performance result. Replace the proxies and
candidate profiles with measured, workload-specific data before operational
use.
