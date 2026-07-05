#include "Dataset.hpp"
#include <random>
#include <stdexcept>
#include <string>

Dataset::Dataset(std::size_t inputSize, std::size_t outputSize,
                  std::size_t sampleCount, unsigned int seed) {
    generateSamples(inputSize, outputSize, sampleCount, seed);
}

void Dataset::generateSamples(std::size_t inputSize, std::size_t outputSize,
                               std::size_t sampleCount, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> inputDist(-1.0, 1.0);
    std::uniform_real_distribution<double> targetDist(0.0, 1.0);

    samples_.reserve(sampleCount);

    for (std::size_t s = 0; s < sampleCount; ++s) {
        Sample sample;
        sample.input.reserve(inputSize);
        sample.target.reserve(outputSize);

        for (std::size_t i = 0; i < inputSize; ++i) {
            sample.input.push_back(inputDist(generator));
        }
        for (std::size_t o = 0; o < outputSize; ++o) {
            // Targets are generated in [0, 1] since both Sigmoid and
            // ReLU-with-bounded-inputs networks produce comparable
            // output ranges, keeping MSE values meaningful across
            // architectures that differ only in activation function.
            sample.target.push_back(targetDist(generator));
        }

        samples_.push_back(std::move(sample));
    }
}

double Dataset::computeMSE(const std::vector<std::vector<double>>& predictions) const {
    if (predictions.size() != samples_.size()) {
        throw std::invalid_argument(
            "Dataset::computeMSE - prediction count (" +
            std::to_string(predictions.size()) +
            ") does not match sample count (" +
            std::to_string(samples_.size()) + ")");
    }

    double totalSquaredError = 0.0;
    std::size_t totalElements = 0;

    for (std::size_t s = 0; s < samples_.size(); ++s) {
        const auto& predicted = predictions[s];
        const auto& target = samples_[s].target;

        if (predicted.size() != target.size()) {
            throw std::invalid_argument(
                "Dataset::computeMSE - prediction size (" +
                std::to_string(predicted.size()) +
                ") does not match target size (" +
                std::to_string(target.size()) + ") at sample " +
                std::to_string(s));
        }

        for (std::size_t i = 0; i < predicted.size(); ++i) {
            double error = predicted[i] - target[i];
            totalSquaredError += error * error;
            ++totalElements;
        }
    }

    return totalElements > 0 ? (totalSquaredError / static_cast<double>(totalElements)) : 0.0;
}
