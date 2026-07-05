#include "Neuron.hpp"
#include <random>
#include <stdexcept>

Neuron::Neuron(std::size_t numInputs,
               std::shared_ptr<IActivation> activation,
               unsigned int seed)
    : weights_(numInputs, 0.0),
      bias_(0.0),
      activation_(std::move(activation)) {
    initializeParameters(seed);
}

void Neuron::initializeParameters(unsigned int seed) {
    // Use a fixed-seed Mersenne Twister so that results are reproducible
    // across runs (important for the benchmark table to be stable/comparable).
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> weightDist(-0.5, 0.5);
    std::uniform_real_distribution<double> biasDist(-0.1, 0.1);

    for (auto& w : weights_) {
        w = weightDist(generator);
    }
    bias_ = biasDist(generator);
}

double Neuron::forward(const std::vector<double>& inputs) const {
    if (inputs.size() != weights_.size()) {
        throw std::invalid_argument(
            "Neuron::forward - input size (" + std::to_string(inputs.size()) +
            ") does not match expected weight count (" +
            std::to_string(weights_.size()) + ")");
    }

    double weightedSum = bias_;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        weightedSum += inputs[i] * weights_[i];
    }

    return activation_->activate(weightedSum);
}
