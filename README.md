# Configurable Neural Network in C++ (OOP)

A C++17, object-oriented feedforward neural network whose **architecture**
(number of layers, neurons per layer, and activation function) is loaded
entirely from external text configuration files. The program performs
**forward-pass inference only** — no training/backpropagation — and
benchmarks four different architectures (small, medium, large, wide)
against a fixed 100-sample synthetic dataset, reporting Mean Squared
Error and execution time for each.

This implements the **Configurable Neural Network** mini-project
(MSc Project 06/21 bridge), which links to MSc-level Neural Architecture
Search using The Bees Algorithm: the config-file format used here defines
exactly the kind of search space (layer count, layer width, activation
choice) that a Bees Algorithm optimiser would explore at MSc level.

## Project Structure

```
ConfigurableNN/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── Activation.hpp      # IActivation strategy interface + Sigmoid/ReLU
│   ├── Neuron.hpp          # Single neuron: weights, bias, forward()
│   ├── Layer.hpp           # A layer of fully-connected neurons
│   ├── Network.hpp         # Full feedforward network (forward pass only)
│   ├── ConfigParser.hpp    # Parses "layers:/sizes:/activation:" config files
│   ├── Dataset.hpp         # Hardcoded 100-sample dataset + MSE computation
│   ├── Timer.hpp           # RAII std::chrono stopwatch
│   └── BenchmarkRunner.hpp # Orchestrates running all architectures + report
├── src/
│   ├── main.cpp
│   ├── Activation.cpp
│   ├── Neuron.cpp
│   ├── Layer.cpp
│   ├── Network.cpp
│   ├── ConfigParser.cpp
│   ├── Dataset.cpp
│   └── BenchmarkRunner.cpp
└── config/
    ├── small.txt    # input(4) -> 6 -> output(2),        2 real layers, sigmoid
    ├── medium.txt   # input(4) -> 8 -> 6 -> output(2),    3 real layers, sigmoid
    ├── large.txt    # input(4) -> 16 -> 12 -> 8 -> out(2), 4 real layers, relu
    └── wide.txt     # input(4) -> 64 -> output(2),        1 wide layer,  relu
```

## OOP Design

- **Strategy pattern** (`IActivation`): `Sigmoid` and `ReLU` are interchangeable
  activation strategies selected at runtime via `IActivation::create(name)`.
  Adding a new activation function (e.g. Tanh) requires only a new subclass —
  no changes to `Neuron`, `Layer`, or `Network`.
- **Single Responsibility**: each class does exactly one thing —
  `Neuron` computes one scalar output; `Layer` fans a vector out across
  neurons; `Network` chains layers; `ConfigParser` only parses text;
  `Dataset` only generates data and scores predictions; `BenchmarkRunner`
  only orchestrates and reports.
- **RAII**: `Timer` wraps `std::chrono` cleanly; all containers use
  value semantics / smart pointers — no manual `new`/`delete` anywhere.
- **Const-correctness**: all read-only operations (`forward`, accessors)
  are `const`; mutation is confined to construction.
- **Exception-based validation**: malformed configs, mismatched
  dimensions, and unknown activation names all throw `std::invalid_argument`
  or `std::runtime_error` with specific, actionable messages — verified
  by direct testing.

## Config File Format

```
layers: 3
sizes: 4 6 2
activation: sigmoid
```

- `layers:` — total count of entries in `sizes:` (input dimension + all
  real layers). Used as a self-check against `sizes:`.
- `sizes:` — space-separated layer widths. The **first** value is the
  network's input dimensionality; every value after it defines one real
  (neuron-bearing) layer.
- `activation:` — `sigmoid` or `relu` (case-insensitive). Applied to
  every neuron in the network.
- Lines starting with `#` are comments; blank lines are ignored.

## Building

Requires CMake ≥ 3.15 and a C++17 compiler (GCC, Clang, or MSVC).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

This produces `configurable_nn` (or `configurable_nn.exe` on Windows) and
automatically copies the `config/` directory next to the executable.

### Building without CMake (quick manual compile)

```bash
g++ -std=c++17 -Wall -Wextra -O2 -Iinclude \
    src/main.cpp src/Activation.cpp src/Neuron.cpp src/Layer.cpp \
    src/Network.cpp src/ConfigParser.cpp src/Dataset.cpp \
    src/BenchmarkRunner.cpp -o configurable_nn
```

## Running

```bash
./configurable_nn config
```

The single argument is the path to the directory containing
`small.txt`, `medium.txt`, `large.txt`, and `wide.txt`. It defaults to
`config` if omitted.

### Sample Output

```
======================================================================================================
  CONFIGURABLE NEURAL NETWORK - ARCHITECTURE BENCHMARK REPORT
======================================================================================================
Architecture            Layer Sizes   Activation           MSE     Avg Time (us)      Total (us)
------------------------------------------------------------------------------------------------------
Small (2 layers)        4-6-2         Sigmoid         0.089199              0.29           28.77
Medium (3 layers)       4-8-6-2       Sigmoid         0.096258              0.44           43.52
Large (4 layers)        4-16-12-8-2   ReLU            0.260462              0.58           58.04
Wide (1 large layer)    4-64-2        ReLU            0.265850              0.66           65.67
======================================================================================================
```

(Exact timing values vary by machine; MSE values are deterministic and
reproducible across runs and machines, since weights and the dataset are
generated from fixed seeds.)

## Notes on Reproducibility

Both the network weights/biases and the synthetic dataset are generated
with `std::mt19937` seeded by fixed constants, so the **MSE values are
identical on every run** (timing naturally varies with hardware/load).
This makes the four architectures directly comparable — any MSE
difference is due to architecture/activation choice, not random noise.
This is intentional and documented behavior, not a limitation: the
project is a forward-pass/architecture-comparison study, not a trained
model, so weight initialization is illustrative rather than learned.

## Extending This Project (MSc Bridge)

The `sizes:`/`activation:` config format is exactly the kind of discrete
search space a Bees Algorithm Neural Architecture Search would explore:
each "bee" could represent one candidate config file, with the resulting
MSE (or accuracy/latency tradeoff) serving directly as the fitness
function — no changes needed to `Network`, `Layer`, or `Neuron`.
