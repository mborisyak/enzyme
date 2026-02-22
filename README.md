# enzyme

Ground-truth model for enzyme kinetics experiments, designed for use in AI scientist workflows.

The package models a two-substrate enzymatic reaction with product inhibition and temperature-dependent activity:

```
A + B  →  C + D   (catalyzed by E)
```

Substrate **A** is consumed, **B** is consumed stoichiometrically with A, and products **C** and **D** are formed in equal amounts.
Both products can act as competitive inhibitors of their respective substrates.
The enzyme's catalytic rate and thermal stability are temperature-dependent.

## Usage

`scripts/experiment.py` simulates an experiment:
- solutions with `A`, `B` and `E` are added to the flask;
- the enzyme starts converting the substrate `A` and co-substrate `B` into products;
- noisy measurements are taken throughout the experiment.

```bash
python scripts/experiment.py \
    --parameters <path.yaml>   \  # kinetic parameters file (required)
    --conditions <path.json>   \  # initial conditions file (required)
    --output <path.json>       \  # output measurements file (required)
    [--seed <int>]             \  # RNG seed for noise (default: None)
    [--device <str>]           \  # JAX device, e.g. cpu, gpu:0 (default: cpu)
    [--config <path.yaml>]     \  # config file (default: config/config.yaml)
    [--plot <path.png>]           # save a debug plot (optional)
```

The experimental conditions are specified by the *condition file* --- a JSON dictionary with experiment labels and experimental conditions.
Each condition must be a dictionary with the following fields:
- `A`, `B`, `E`: **volumes** of the corresponding solutions (units don't really matter here, but let's say mL);
- `temperature`, in Celsius.

Concentrations of the solutions can be found in the config file:

```yaml
### concentrations of solutions, in mM
solutions:
  A: 3.0
  B: 3.0
  E: 3.0e-3
```

Example of a condition file:
```json
{
  "experiment 1": {"A": 1.0, "B": 2.0,  "E": 1.0, "temperature": 10.0},
  "experiment 2": {"A": 1.0, "B": 1.2,  "E": 1.0, "temperature": 50.0},
  "experiment 3": {"A": 1.0, "B": 1.0,  "E": 1.0, "temperature": 36.6}
}
```

**Note**: substances are mixed, e.g., initial concentration of `A` is
```
    A = A_volume * A_solution_concentration / total_volume
``` 

**Output format** (JSON):

```json
{
  "experiment 1": {
    "timestamps":    [t1, t2, ...],
    "measurements":  [A1, A2, ...]
  },
  ...
}
```

Measurements are concentrations of substrate A (mM) with additive Gaussian noise (std configured via `noise` in the config).

**Example:**

```bash
# 1. Generate parameters
python scripts/generate_parameters.py --seed 42 --output params.yaml

# 2. Run experiment
python scripts/experiment.py \
    --parameters params.yaml \
    --conditions scripts/example.json \
    --output results.json \
    --seed 12345 \
    --plot results.png
```

## Python API

### `enzyme.kinetics(A, B, C, D, E, temperature, parameters)`

Computes the instantaneous reaction rate for the enzymatic conversion A + B → C + D.

**Arguments:**

| Argument | Type | Description |
|---|---|---|
| `A` | array | Concentration of substrate A (mM) |
| `B` | array | Concentration of substrate B (mM) |
| `C` | array | Concentration of product C (mM) |
| `D` | array | Concentration of product D (mM) |
| `E` | array | Total enzyme concentration (mM) |
| `temperature` | array | Temperature (°C) |
| `parameters` | dict | Kinetic parameters (see below) |

## Kinetics model

The rate law is a bi-substrate Michaelis–Menten expression with competitive product inhibition and temperature-dependent terms:

```
rate = k_cat(T) · E_active(T) · A · B / (A + Kapp_A(T)) / (B + Kapp_B(T))
```

where:
- `Kapp_A = K_A(T) · (1 + C / Ki_C(T))` — apparent Michaelis constant for A
- `Kapp_B = K_B(T) · (1 + D / Ki_D(T))` — apparent Michaelis constant for B
- `k_cat(T)` — catalytic rate constant, modeled via Arrhenius
- `E_active(T)` — fraction of active (folded) enzyme, modeled via a Gibbs unfolding equilibrium

**Parameter dictionary keys:**

| Parameter | Description |
|---|---|
| `log_k0_cat` | Log of catalytic rate constant at 0 °C |
| `Q10_cat` | Q10 temperature coefficient for k_cat |
| `log_K0_A` | Log of Michaelis constant for A at 0 °C |
| `Q10_A` | Q10 coefficient for K_A |
| `log_K0_B` | Log of Michaelis constant for B at 0 °C |
| `Q10_B` | Q10 coefficient for K_B |
| `log_K0i_C` | Log of inhibition constant for product C at 0 °C |
| `Q10_C` | Q10 coefficient for Ki_C |
| `log_K0i_D` | Log of inhibition constant for product D at 0 °C |
| `Q10_D` | Q10 coefficient for Ki_D |
| `T_melting` | Enzyme melting temperature (°C) |
| `delta_H` | Enthalpy of unfolding normalized by R (K) |
| `delta_C` | Heat capacity change of unfolding normalized by R (K) |

**Important note: logarithms are natural, not base 10.** 

**Example:**

```python
import jax.numpy as jnp
import enzyme

parameters = {
    'log_k0_cat': 5.0,
    'Q10_cat': 2.0,
    'log_K0_A': -1.0,  'Q10_A': 1.5,
    'log_K0_B': -1.0,  'Q10_B': 1.5,
    'log_K0i_C': -2.0, 'Q10_C': 1.2,
    'log_K0i_D': -2.0, 'Q10_D': 1.2,
    'T_melting': 55.0,
    'delta_H': 60000.0,
    'delta_C': 600.0,
}

rate = enzyme.kinetics(
    A=1.0, B=1.2, C=0.0, D=0.0,
    E=1e-3, temperature=37.0,
    parameters=parameters
)
```

## Scripts

### `scripts/generate_parameters.py` — Sample kinetic parameters

Draws a random parameter set from the prior ranges defined in the config file and saves it to a YAML file.

```bash
python scripts/generate_parameters.py \
    --seed <int>          \  # RNG seed (required)
    --output <path.yaml>  \  # output file (required)
    [--config <path.yaml>]   # config file (default: config/config.yaml)
```

**Output:** YAML file with one sampled value per parameter.

---

## Configuration

Default configuration is in `config/config.yaml`:

```yaml
parameters:
  log_k0_cat:  [5.0,   7.5]      # prior range [low, high]
  log_K0_A:    [-4.5,  0.0]
  log_K0_B:    [-4.5,  0.0]
  log_K0i_C:   [-5.0,  0.0]
  log_K0i_D:   [-5.0,  0.0]
  Q10_cat:     [1.25,  2.5]
  Q10_A:       [1.1,   1.75]
  Q10_B:       [1.1,   1.75]
  Q10_C:       [1.05,  1.5]
  Q10_D:       [1.05,  1.5]
  T_melting:   [40.0,  70.0]     # °C
  delta_H:     [37500, 100000]   # normalized by R, in K
  delta_C:     [375,   1000]     # normalized by R, in K

### concentrations of solutions, in mM
solutions:
  A: 3.0
  B: 3.0
  E: 3.0e-3

experiment:
  duration:     30.0   # seconds
  measurements: 10     # number of time points

noise: 0.025           # Gaussian noise std (mM)
```

A custom config file can be passed to any script via `--config`.