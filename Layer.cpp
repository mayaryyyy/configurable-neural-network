#include "Layer.hpp"

Layer::Layer(std::size_t numNeurons,
             std::size_t numInputsPerNeuron,
             std::shared_ptr<IActivation> activation,
             unsigned int layerSeed)
    : activationName_(activation ? activation->name() : "Unknown") {
    neurons_.reserve(numNeurons);

    for (std::size_t i = 0; i < numNeurons; ++i) {
        // Derive a distinct, reproducible seed per neuron so that no two
        // neurons in the same layer get identical weights, while the
        // whole run remains deterministic across program executions.
        unsigned int neuronSeed = layerSeed + static_cast<unsigned int>(i) * 7919u;
        neurons_.emplace_back(numInputsPerNeuron, activation, neuronSeed);
    }
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) const {
    std::vector<double> outputs;
    outputs.reserve(neurons_.size());

    for (const auto& neuron : neurons_) {
        outputs.push_back(neuron.forward(inputs));
    }

    return outputs;
}
