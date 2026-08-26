# Datacenter Workload Model

Code for **"Wide-Area Power System Oscillations from Large-Scale AI Workloads"**, IEEE Transactions on Power Systems, 2026.  
https://ieeexplore.ieee.org/abstract/document/11487948

This code generates synthetic AI workload profiles and combines them with a steady-state load (non-server and non-AI loads) to construct a time-domain datacenter power profile. The model supports:

- Cyclic training workloads
- Cyclic fine-tuning workloads
- Multiple concurrently operating AI jobs
- Random or user-specified workload start times
- User-specified workload turn-off times
- Boundary ramp-rate limits for workload activation and deactivation
- Separate scheduling of large-training, small-training, and fine-tuning jobs
- Time-domain component visualization and FFT analysis
- Construction of a total datacenter load profile

For a future hyperscale AI datacenter, we recommend assigning a high workload ratio to a single large training workload. The remaining AI load can be composed of a small number of small training and fine-tuning workloads.

For a smaller/current AI datacenter, the AI workload can be constructed using only small training and fine-tuning workloads.

---

## Parameter Reference

### Common simulation parameters

| Parameter | Meaning |
|---|---|
| `T` | Total simulation duration in seconds. |
| `Fs` | Sampling frequency in samples per second. |
| `seed` | Random seed controlling reproducibility. |

---

## `training_phase()` parameters

| Parameter | Meaning |
|---|---|
| `T` | Duration of the generated training workload segment. |
| `Fs` | Sampling frequency in samples per second. |
| `F` | Range from which the nominal training-cycle frequency $f_0$ is sampled, in hertz. |
| `up_ratio` | Range of the fraction of each cycle assigned to the high-power state. |
| `p_hat_tr` | Nominal power of the training workload. |
| `sigma_xi` | Standard deviation of the cycle-duration perturbation $\xi_i$. |
| `sigma_delta` | Standard deviation of the cycle-level power-offset variables. |
| `mu_delta` | Mean reduction applied during the down state. |
| `sigma_eta_up` | Standard deviation, or sampling range, of pointwise noise during the up state. |
| `sigma_eta_down` | Standard deviation, or sampling range, of pointwise noise during the down state. |
| `seed` | Random seed for the workload realization. |
| `return_info` | Returns the generated signal together with cycle-level metadata when `True`. |

A larger `F` produces faster cyclic fluctuations. A larger `sigma_xi` produces greater cycle-to-cycle timing variability. A larger `mu_delta` increases the separation between the up and down power levels.

The native training waveform is generated as alternating high-power and down-state segments. Boundary ramp-rate limits, when requested, are applied later by the scheduling layer and do not modify the internal cyclic model after the start-up ramp has caught the native waveform.

---

## `finetuning_phase()` parameters

| Parameter | Meaning |
|---|---|
| `T` | Duration of the generated fine-tuning workload segment. |
| `Fs` | Sampling frequency in samples per second. |
| `F` | Range from which the nominal fine-tuning-cycle frequency $f_1$ is sampled, in hertz. |
| `tail_ratio` | Range of the fraction of each cycle assigned to the high-power tail state. |
| `p_hat_ft` | Nominal power of one fine-tuning workload. |
| `sigma_zeta` | Standard deviation of the fine-tuning cycle-duration perturbation. |
| `sigma_delta` | Standard deviation of the cycle-level power offsets. |
| `mu_delta` | Mean reduction applied during the idle state. |
| `sigma_eta_tail` | Standard deviation, or sampling range, of pointwise noise during the tail state. |
| `sigma_eta_idle` | Standard deviation, or sampling range, of pointwise noise during the idle state. |
| `seed` | Random seed for the workload realization. |
| `return_info` | Returns the generated signal together with cycle-level metadata when `True`. |

A larger `tail_ratio` causes the workload to spend more time near its high-power state. A larger `mu_delta` produces a lower idle-state power.

---

## Workload scheduling and boundary ramping

Each training or fine-tuning workload can be assigned an individual schedule. The schedule is specified by a dictionary with the following entries:

| Schedule key | Meaning |
|---|---|
| `turn_on_time` | Workload start time in seconds. This is the actual start of the native training/fine-tuning waveform. |
| `turn_off_time` | Time in seconds at which the workload becomes exactly zero. |
| `ramp_up_ratio` | Maximum start-up ramp rate normalized by the workload nominal power, in $\mathrm{s}^{-1}$. |
| `ramp_down_ratio` | Maximum shut-down ramp rate normalized by the workload nominal power, in $\mathrm{s}^{-1}$. |

For a workload with nominal power $\hat P$, the corresponding absolute ramp-rate limits are

$$
R_{\mathrm{up}} = r_{\mathrm{up}}\hat P,
\qquad
R_{\mathrm{down}} = r_{\mathrm{down}}\hat P.
$$

At start-up, the workload begins from zero at `turn_on_time` and is limited by

$$
P[k] \le P[k-1] + R_{\mathrm{up}}\Delta t,
\qquad \Delta t=\frac{1}{F_s}.
$$

The ramp constraint is applied only until the ramped signal reaches the native workload waveform. After that point, the original cyclic training/fine-tuning waveform is preserved.

At shut-down, `turn_off_time` is the exact zero-load boundary. The final portion of the workload is processed backward from

$$
P(t_{\mathrm{off}})=0
$$

using the limit

$$
P[k] \le P[k+1] + R_{\mathrm{down}}\Delta t.
$$

If `ramp_up_ratio=np.inf` or `ramp_down_ratio=np.inf`, the corresponding boundary ramp constraint is disabled.

### Schedule examples

```python
large_training_schedule = {
    "turn_on_time": 0.50,
    "turn_off_time": 9.00,
    "ramp_up_ratio": 2.0,
    "ramp_down_ratio": 1.0,
}

small_training_schedules = [
    {
        "turn_on_time": 0.80,
        "turn_off_time": 7.50,
        "ramp_up_ratio": 2.5,
        "ramp_down_ratio": 1.5,
    },
    {
        "turn_on_time": 1.50,
        "turn_off_time": 10.00,
        "ramp_up_ratio": 1.5,
        "ramp_down_ratio": 0.8,
    },
]
```

The number of dictionaries in `small_training_schedules` and `finetuning_schedules` must equal `Ntr` and `Nft`, respectively.

### Random start times

When a schedule is `None`, its start time is sampled uniformly as

$$
t_{\mathrm{on}} \sim \mathcal{U}(0,(\texttt{max start time})),
$$

while `turn_off_time` defaults to `T` and both boundary ramp-rate limits default to `np.inf`.

If a schedule dictionary is provided but omits `turn_on_time`, the same random-start rule is used. If it omits `turn_off_time`, the turn-off time defaults to `T`.

Requested times are mapped to the nearest sample index according to `Fs`; the sample-aligned values are stored in the returned workload metadata.

---

## `aggregated_ai_workload()` parameters

| Parameter | Meaning |
|---|---|
| `T` | Total simulation duration in seconds. |
| `Fs` | Sampling frequency in samples per second. |
| `total_ai_nominal` | Total nominal power assigned to all modeled AI workloads. |
| `large_training_ratio` | Fraction of `total_ai_nominal` assigned to the single large training workload. |
| `small_training_total_ratio` | Total fraction assigned collectively to all small training workloads. |
| `finetuning_total_ratio` | Total fraction assigned collectively to all fine-tuning workloads. |
| `Ntr` | Number of small training workloads. |
| `Nft` | Number of fine-tuning workloads. |
| `max_start_time` | Upper bound of the default random start time interval when a workload schedule does not explicitly specify `turn_on_time`. |
| `large_training_schedule` | Schedule dictionary for the large training workload, or `None` for the default random-start behavior. |
| `small_training_schedules` | List of schedule dictionaries for the `Ntr` small training workloads. |
| `finetuning_schedules` | List of schedule dictionaries for the `Nft` fine-tuning workloads. |
| `seed` | Seed controlling frequencies, durations, amplitudes, noise, and random start times. |
| `return_components` | Returns the aggregated signal and its constituent signals/metadata when `True`. |

The three nominal workload ratios must satisfy

$$
r_{\mathrm{large}}+r_{\mathrm{small}}+r_{\mathrm{ft}}=1.
$$

### Interpretation of `Ntr` and `Nft`

`Ntr` and `Nft` change the number of jobs, not the total nominal power allocated to their categories.

Increasing `Ntr` reduces the nominal power per small training job while preserving the total small-training allocation. Similarly, increasing `Nft` divides the total fine-tuning allocation among more independent jobs. More independently phased jobs can reduce aggregate variability through temporal diversity, although the exact result remains stochastic.

### Returned component information

When `return_components=True`, the returned `components` dictionary includes:

- `large_training`: large-training time series
- `small_training`: list of small-training time series
- `fine_tuning`: list of fine-tuning time series
- `start_times`: sample-aligned turn-on times
- `turn_off_times`: sample-aligned zero-load times
- `metadata`: global workload-allocation information
- `workload_info`: detailed schedule, native waveform, ramp, and cycle information for each workload

For each workload, `workload_info` includes the requested and sample-aligned turn-on/turn-off times, ramp ratios, the native active waveform, and information about where the start-up and shut-down ramp limits are active.

---

## `datacenter_load_profile()` parameters

| Parameter | Meaning |
|---|---|
| `T` | Total simulation duration in seconds. |
| `Fs` | Sampling frequency in samples per second. |
| `datacenter_capacity` | Reference datacenter capacity used to calculate the steady component. |
| `steady_component_ratio` | Fraction of `datacenter_capacity` assigned to the constant background load. |
| `fluctuating_component_ratio` | Intended fraction assigned to the AI workload. In the current implementation, this parameter is used for consistency checking and metadata but does not scale `P_AI`. |
| `max_start_time` | Retained in the current function interface but not used to generate or modify `P_AI`, which must already be supplied. |
| `P_AI` | AI workload array generated externally by `aggregated_ai_workload()`. |
| `components` | Metadata and component dictionary associated with the supplied `P_AI`. |
| `seed` | Retained in the current function interface; the supplied `P_AI` is not regenerated inside this function. |
| `return_components` | Returns the total profile and updated component dictionary when `True`. |

The ratios must satisfy

$$
r_{\mathrm{steady}}+r_{\mathrm{fluctuating}}=1.
$$

The total datacenter load is constructed as

$$
P_{\mathrm{DC}}(t)=P_{\mathrm{steady}}+P_{\mathrm{AI}}(t),
$$

with

$$
P_{\mathrm{steady}}
=(\texttt{datacenter capacity}) \times (\texttt{steady component ratio}).
$$

---

## Plotting utilities

The notebook also provides helper functions for inspecting the generated workload:

| Function | Purpose |
|---|---|
| `plot_ai_workload_components()` | Plots the large-training, aggregated small-training, aggregated fine-tuning, and total AI workload signals. Optional vertical lines indicate turn-on and turn-off times. |
| `plot_ai_workload_fft()` | Computes and plots the single-sided FFT of individual workload categories and the aggregated AI workload. |
| `plot_datacenter_load()` | Plots the steady load, AI workload, and total datacenter load. |
| `plot_datacenter_fft()` | Compares the FFT of the AI workload and total datacenter load. |

---

## Example: hyperscale AI datacenter

```python
T = 12.0
Fs = 100
seed = 1

large_training_schedule = {
    "turn_on_time": 0.50,
    "turn_off_time": 9.00,
    "ramp_up_ratio": 2.0,
    "ramp_down_ratio": 1.0,
}

P_AI, components = aggregated_ai_workload(
    T=T,
    Fs=Fs,
    total_ai_nominal=1.0,
    large_training_ratio=0.90,
    small_training_total_ratio=0.05,
    finetuning_total_ratio=0.05,
    Ntr=2,
    Nft=2,
    max_start_time=2.0,
    large_training_schedule=large_training_schedule,
    small_training_schedules=small_training_schedules,
    finetuning_schedules=finetuning_schedules,
    seed=seed,
    return_components=True,
)
```

The notebook contains a second example with no large training workload and many small training/fine-tuning workloads to emulate a smaller or more distributed AI datacenter workload composition.
