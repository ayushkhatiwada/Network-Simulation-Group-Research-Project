# Network Probing Tutorial

## Objective
- Probe different networks to estimate the **mean** and **standard deviation** of network delays.
- Aim to achieve a **KL divergence score ≤ 0.05**.
- Network only contains two nodes, node 1 and node 2, for now
- Choose between:
  - **One path ("1")**: Single edge between nodes 1 and 2.
  - **Two paths ("2")**: Two parallel edges between nodes 1 and 2.

## Step 1: Choose a Network
- Import and initialize a simulator version:
  ```python
  from active_simulator_v0 import ActiveSimulator_v0
  sim = ActiveSimulator_v0(paths="1")  # One path network
  ```
- Versions:
  - `ActiveSimulator_v0`: Basic network delay simulation.
  - `ActiveSimulator_v1`: Adds **random packet drops**.
  - `ActiveSimulator_v2`: Adds **network congestion**.

## Step 2: Probe the Network
- Send probes to measure delay:
  ```python
  delay = sim.send_probe_at(5.0)  # Probe at t=5.0 sec
  print(f"Measured delay: {delay} ms")
  ```
- Probes must be sent within **0-100 seconds**.
- Max **10 probes per second**.

- Send multiple probes:
  ```python
  delays = sim.send_multiple_probes([1.0, 2.5, 3.0, 10.7])
  print(delays)
  ```

## Step 3: Estimate Distribution Parameters
- Try to estimate the network delay’s **mean** and **std deviation**.
- Compare your guess using KL divergence:
  ```python
  kl_score = sim.compare_distribution_parameters(pred_mean=0.8, pred_std=0.15)
  print(f"KL divergence: {kl_score}")
  ```
- Your goal: **KL score ≤ 0.05** ✅

## Step 4: Experiment with More Complex Networks
- Try `ActiveSimulator_v1` and `ActiveSimulator_v2` to test robustness against **packet drops** and **congestion**.
  ```python
  from active_simulator_v2 import ActiveSimulator_v2
  sim = ActiveSimulator_v2(paths="2")
  ```

## Notes
- You **do not know** the congestion periods in `ActiveSimulator_v2`.
- Use **multiple probes** at different times to infer the delay behavior.
