#ifndef BENCHMARK_RUNNER_HPP
#define BENCHMARK_RUNNER_HPP

#include <vector>
#include <string>
#include "ConfigParser.hpp"

/**
 * @brief Result of benchmarking a single network architecture.
 */
struct BenchmarkResult {
    std::string label;            ///< e.g. "Small (2 layers)"
    std::string architecture;     ///< e.g. "4-3-2"
    std::string activation;       ///< "Sigmoid" or "ReLU"
    double meanSquaredError;      ///< MSE over the full dataset
    double avgTimeMicroseconds;   ///< Average forward-pass time per sample
    double totalTimeMicroseconds; ///< Total time for all samples
};

/**
 * @brief Orchestrates loading configs, building networks, running the
 *        forward-pass benchmark over the dataset, and reporting results.
 *
 * Responsibility: this class ties together ConfigParser, Network, and
 * Dataset to execute the end-to-end benchmark described in the project
 * specification: for each of several architecture configs, build the
 * corresponding network, run a forward pass over every sample in a
 * 100-sample dataset, measure execution time with std::chrono, compute
 * MSE, and finally print a comparison table.
 *
 * It intentionally contains no neural-network math itself -- that logic
 * lives in Neuron/Layer/Network -- keeping this class focused purely on
 * orchestration and reporting (single-responsibility principle).
 */
class BenchmarkRunner {
public:
    /**
     * @brief Run the benchmark for a single architecture configuration.
     * @param config The parsed network configuration to benchmark.
     * @return A populated BenchmarkResult.
     */
    BenchmarkResult runSingle(const NetworkConfig& config);

    /**
     * @brief Run the benchmark for several configurations in sequence.
     * @param configs The list of configurations to benchmark, in order.
     * @return One BenchmarkResult per input configuration, in the same order.
     */
    std::vector<BenchmarkResult> runAll(const std::vector<NetworkConfig>& configs);

    /**
     * @brief Print a formatted comparison table of benchmark results to stdout.
     * @param results The results to display, typically the output of runAll().
     */
    static void printReport(const std::vector<BenchmarkResult>& results);
};

#endif // BENCHMARK_RUNNER_HPP
