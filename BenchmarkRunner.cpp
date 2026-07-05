#include "BenchmarkRunner.hpp"
#include "Network.hpp"
#include "Dataset.hpp"
#include "Timer.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

BenchmarkResult BenchmarkRunner::runSingle(const NetworkConfig& config) {
    Network network(config.layerSizes, config.activation);

    std::size_t inputSize = config.layerSizes.front();
    std::size_t outputSize = config.layerSizes.back();

    // Fixed sample count and seed per specification: a 100-sample
    // hardcoded dataset, identical across all benchmarked architectures
    // so that MSE values are directly comparable.
    Dataset dataset(inputSize, outputSize, /*sampleCount=*/100, /*seed=*/2024);

    std::vector<std::vector<double>> predictions;
    predictions.reserve(dataset.size());

    Timer timer;
    timer.start();

    for (const auto& sample : dataset.samples()) {
        predictions.push_back(network.forward(sample.input));
    }

    timer.stop();

    double totalMicros = timer.elapsedMicroseconds();
    double avgMicros = totalMicros / static_cast<double>(dataset.size());
    double mse = dataset.computeMSE(predictions);

    BenchmarkResult result;
    result.label = config.label;
    result.architecture = network.architectureString();
    // Network::activationName() already returns the canonical display
    // form ("Sigmoid" or "ReLU") via IActivation::name(), so it is used
    // as-is rather than re-capitalized (naive capitalization would turn
    // "ReLU" into "Relu").
    result.activation = network.activationName();
    result.meanSquaredError = mse;
    result.avgTimeMicroseconds = avgMicros;
    result.totalTimeMicroseconds = totalMicros;

    return result;
}

std::vector<BenchmarkResult> BenchmarkRunner::runAll(const std::vector<NetworkConfig>& configs) {
    std::vector<BenchmarkResult> results;
    results.reserve(configs.size());

    for (const auto& config : configs) {
        results.push_back(runSingle(config));
    }

    return results;
}

void BenchmarkRunner::printReport(const std::vector<BenchmarkResult>& results) {
    const int labelWidth = 24;
    const int archWidth = 14;
    const int actWidth = 10;
    const int mseWidth = 14;
    const int timeWidth = 18;
    const int totalTimeWidth = 16;

    std::cout << "\n";
    std::cout << std::string(labelWidth + archWidth + actWidth + mseWidth +
                              timeWidth + totalTimeWidth + 6, '=') << "\n";
    std::cout << "  CONFIGURABLE NEURAL NETWORK - ARCHITECTURE BENCHMARK REPORT\n";
    std::cout << std::string(labelWidth + archWidth + actWidth + mseWidth +
                              timeWidth + totalTimeWidth + 6, '=') << "\n";

    std::cout << std::left
              << std::setw(labelWidth) << "Architecture"
              << std::setw(archWidth) << "Layer Sizes"
              << std::setw(actWidth) << "Activation"
              << std::right
              << std::setw(mseWidth) << "MSE"
              << std::setw(timeWidth) << "Avg Time (us)"
              << std::setw(totalTimeWidth) << "Total (us)"
              << "\n";

    std::cout << std::string(labelWidth + archWidth + actWidth + mseWidth +
                              timeWidth + totalTimeWidth + 6, '-') << "\n";

    std::cout << std::fixed << std::setprecision(6);

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(labelWidth) << r.label
                  << std::setw(archWidth) << r.architecture
                  << std::setw(actWidth) << r.activation
                  << std::right
                  << std::setw(mseWidth) << r.meanSquaredError
                  << std::setprecision(2)
                  << std::setw(timeWidth) << r.avgTimeMicroseconds
                  << std::setw(totalTimeWidth) << r.totalTimeMicroseconds
                  << std::setprecision(6)
                  << "\n";
    }

    std::cout << std::string(labelWidth + archWidth + actWidth + mseWidth +
                              timeWidth + totalTimeWidth + 6, '=') << "\n";
    std::cout << std::defaultfloat;
}
