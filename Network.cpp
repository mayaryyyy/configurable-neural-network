#include "Network.hpp"
#include <stdexcept>
#include <sstream>

Network::Network(const std::vector<std::size_t>& layerSizes,
                  const std::string& activationName,
                  unsigned int baseSeed)
    : layerSizes_(layerSizes),
      activationName_(activationName) {
    validateArchitecture();

    activation_ = IActivation::create(activationName_);

    layers_.reserve(layerSizes_.size() - 1);

    // layerSizes_[0] is the input dimensionality (no neurons of its own).
    // Each subsequent entry defines a real layer of neurons whose input
    // width is the previous entry in layerSizes_.
    for (std::size_t i = 1; i < layerSizes_.size(); ++i) {
        std::size_t numNeurons = layerSizes_[i];
        std::size_t numInputsPerNeuron = layerSizes_[i - 1];

        // Derive a distinct, reproducible seed per layer.
        unsigned int layerSeed = baseSeed + static_cast<unsigned int>(i) * 104729u;

        layers_.emplace_back(numNeurons, numInputsPerNeuron, activation_, layerSeed);
    }
}

void Network::validateArchitecture() const {
    if (layerSizes_.size() < 2) {
        throw std::invalid_argument(
            "Network: architecture must specify an input size and at "
            "least one real layer (e.g. {4, 3, 2}).");
    }

    for (std::size_t size : layerSizes_) {
        if (size == 0) {
            throw std::invalid_argument(
                "Network: layer sizes must all be greater than zero.");
        }
    }
}

std::vector<double> Network::forward(const std::vector<double>& input) const {
    if (input.size() != layerSizes_.front()) {
        throw std::invalid_argument(
            "Network::forward - input size (" + std::to_string(input.size()) +
            ") does not match configured input dimensionality (" +
            std::to_string(layerSizes_.front()) + ")");
    }

    std::vector<double> current = input;
    for (const auto& layer : layers_) {
        current = layer.forward(current);
    }

    return current;
}

std::string Network::architectureString() const {
    std::ostringstream oss;
    for (std::size_t i = 0; i < layerSizes_.size(); ++i) {
        if (i > 0) {
            oss << "-";
        }
        oss << layerSizes_[i];
    }
    return oss.str();
}
