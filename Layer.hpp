#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <memory>
#include "Neuron.hpp"
#include "Activation.hpp"

/**
 * @brief Represents one fully-connected layer of neurons.
 *
 * A Layer owns a collection of Neuron objects, all sharing the same
 * activation strategy and all receiving the same input vector. Its sole
 * responsibility is to run a forward pass: feed the input vector to every
 * neuron in the layer and collect their outputs into the layer's output
 * vector, which becomes the next layer's input.
 */
class Layer {
public:
    /**
     * @brief Construct a layer of fully-connected neurons.
     * @param numNeurons Number of neurons in this layer (its "width").
     * @param numInputsPerNeuron Number of inputs each neuron receives
     *        (the width of the previous layer, or the network input size
     *        for the first layer).
     * @param activation Shared activation strategy used by every neuron
     *        in this layer.
     * @param layerSeed Base seed used to derive distinct, reproducible
     *        per-neuron seeds within this layer.
     */
    Layer(std::size_t numNeurons,
          std::size_t numInputsPerNeuron,
          std::shared_ptr<IActivation> activation,
          unsigned int layerSeed);

    /**
     * @brief Run the forward pass for this layer.
     * @param inputs The input vector (from the previous layer, or the
     *        network's input for the first layer).
     * @return A vector containing each neuron's activated output, in the
     *         same order the neurons were constructed.
     */
    std::vector<double> forward(const std::vector<double>& inputs) const;

    /// @brief Number of neurons in this layer.
    std::size_t size() const { return neurons_.size(); }

    /// @brief Read-only access to the underlying neurons (for inspection/testing).
    const std::vector<Neuron>& neurons() const { return neurons_; }

    /// @brief Name of the activation function used by this layer.
    std::string activationName() const { return activationName_; }

private:
    std::vector<Neuron> neurons_;
    std::string activationName_;
};

#endif // LAYER_HPP
