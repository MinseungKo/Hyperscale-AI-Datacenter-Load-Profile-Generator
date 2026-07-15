# Datacenter Workload Model
Code for "Wide-Area Power System Oscillations from Large-Scale AI Workloads" in IEEE Transactions on Power Systems, 2026. (https://ieeexplore.ieee.org/abstract/document/11487948)
This code generates synthetic AI workload profiles and combines them with a steady-state load (non-server and non-AI loads) to construct a time-domain datacenter power profile. The model supports:

- Cyclic training workloads
- Cyclic fine-tuning workloads
- Multiple concurrently operating AI jobs
- Random job start times
- Construction of a total datacenter load profile

For the future hyperscale AI datacenter, we recommend a high workload ratio of a single, large training workload. The remaining can be composed of a small number of small training and fine-tuning workloads.

In the case of a smaller/current AI datacenter, AI workloads can be built only with small training and fine-tuning workloads.

## Parameter Reference

## Common simulation parameters

| Parameter | Meaning |
|---|---|
| `T` | Total simulation duration in seconds. |
| `Fs` | Sampling frequency in samples per second. |
| `seed` | Random seed controlling reproducibility. |

---

## `training_phase()` parameters

| Parameter | Meaning |
|---|---|
| `F` | Range from which the nominal training-cycle frequency \(f_0\) is sampled, in hertz. |
| `up_ratio` | Range of the fraction of each cycle assigned to the high-power state. |
| `p_hat_tr` | Nominal power of the training workload. |
| `sigma_xi` | Standard deviation of cycle-duration perturbation \(\xi_i\). |
| `sigma_delta` | Standard deviation of cycle-level power-offset variables. |
| `mu_delta` | Mean reduction applied during the down state. |
| `sigma_eta_up` | Standard deviation, or sampling range, of pointwise noise during the up state. |
| `sigma_eta_down` | Standard deviation, or sampling range, of pointwise noise during the down state. |

A larger `F` produces faster cyclic fluctuations. A larger `sigma_xi` produces greater cycle-to-cycle timing variability. A larger `mu_delta` increases the separation between the up and down power levels.

---

## `finetuning_phase()` parameters

| Parameter | Meaning |
|---|---|
| `F` | Range from which the nominal fine-tuning-cycle frequency \(f_1\) is sampled. |
| `tail_ratio` | Fraction of each cycle assigned to the high-power tail state. |
| `p_hat_ft` | Nominal power of one fine-tuning workload. |
| `sigma_zeta` | Standard deviation of fine-tuning cycle-duration perturbation. |
| `sigma_delta` | Standard deviation of cycle-level power offsets. |
| `mu_delta` | Mean reduction applied during the idle state. |
| `sigma_eta_tail` | Pointwise noise level during the tail state. |
| `sigma_eta_idle` | Pointwise noise level during the idle state. |

A larger `tail_ratio` causes the workload to spend more time near its high-power state. A larger `mu_delta` produces a lower idle-state power.

---

## `aggregated_ai_workload()` parameters

| Parameter | Meaning |
|---|---|
| `total_ai_nominal` | Total nominal power assigned to all modeled AI workloads. |
| `large_training_ratio` | Fraction of `total_ai_nominal` assigned to the single large training workload. |
| `small_training_total_ratio` | Total fraction assigned collectively to all small training workloads. |
| `finetuning_total_ratio` | Total fraction assigned collectively to all fine-tuning workloads. |
| `Ntr` | Number of small training workloads. |
| `Nft` | Number of fine-tuning workloads. |
| `max_start_time` | Latest possible random activation time of each workload. |
| `seed` | Seed controlling frequencies, durations, amplitudes, noise, and start times. |
| `return_components` | Returns the aggregated signal and its constituent signals when `True`. |

The three nominal workload ratios must satisfy $r_{\mathrm{large}}+r_{\mathrm{small}}+r_{\mathrm{ft}}=1.$

### Interpretation of `Ntr` and `Nft`

`Ntr` and `Nft` change the number of jobs, not the total nominal power allocated to their categories.

Increasing `Ntr` reduces the nominal power per small training job while preserving the total small-training allocation. More independently phased jobs can reduce aggregate variability through temporal diversity, although the exact result remains stochastic.


## `datacenter_load_profile()` parameters

| Parameter | Meaning |
|---|---|
| `datacenter_capacity` | Reference datacenter capacity used to calculate the steady component. |
| `steady_component_ratio` | Fraction of `datacenter_capacity` assigned to the constant background load. |
| `fluctuating_component_ratio` | Intended fraction assigned to the AI workload. In the current implementation, this parameter is used for validation and metadata but does not scale `P_AI`. |
| `P_AI` | AI workload array generated externally by `aggregated_ai_workload()`. |
| `components` | Metadata and component dictionary associated with the supplied `P_AI`. |
| `return_components` | Returns the total profile and updated component dictionary when `True`. |

The ratios must satisfy $r_{\mathrm{steady}}+r_{\mathrm{fluctuating}}=1.$
