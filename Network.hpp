#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <memory>
#include <string>
#include "Layer.hpp"
#include "Activation.hpp"

/**
 * @brief Represents a configurable feedforward neural network.
 *
 * A Network owns an ordered sequence of Layer objects. Its architecture
 * (number of layers and number of neurons per layer) is supplied at
 * construction time, typically loaded from a configuration file via
 * ConfigParser. The Network's only computational responsibility is to
 * run a full forward pass: feed an input vector through every layer in
 * sequence and return the final layer's output.
 *
 * No training (backpropagation, gradient descent, etc.) is implemented;
 * this class performs inference only, per the project specification.
 */
class Network {
public:
    /**
     * @brief Construct a network from an explicit architecture description.
     * @param layerSizes Sizes of each layer, e.g. {4, 3, 2} means an
     *        input layer expecting 4 inputs, a hidden layer of 3 neurons,
     *        and an output layer of 2 neurons. The first entry is treated
     *        as the input dimensionality and does NOT create a Layer of
     *        neurons by itself; it only defines how many inputs the
     *        first real layer (layerSizes[1]) receives.
     * @param activationName Name of the activation function to use for
     *        every neuron in the network ("sigmoid" or "relu").
     * @param baseSeed Base seed used to derive reproducible per-layer
     *        seeds, so that the same architecture always produces the
     *        same weights across runs.
     *
     * @throws std::invalid_argument if layerSizes has fewer than 2
     *         entries (an input size and at least one real layer are
     *         required), or if any layer size is zero.
     */
    Network(const std::vector<std::size_t>& layerSizes,
            const std::string& activationName,
            unsigned int baseSeed = 42);

    /**
     * @brief Run a full forward pass through the network.
     * @param input The input feature vector. Its size must equal
     *        layerSizes[0] as supplied at construction.
     * @return The output vector produced by the final layer.
     * @throws std::invalid_argument if input.size() does not match the
     *         expected input dimensionality.
     */
    std::vector<double> forward(const std::vector<double>& input) const;

    /// @brief Total number of real (neuron-bearing) layers in the network.
    std::size_t numLayers() const { return layers_.size(); }

    /// @brief The full architecture description, including the input size.
    const std::vector<std::size_t>& architecture() const { return layerSizes_; }

    /// @brief Canonical display name of the activation function in use (e.g. "Sigmoid", "ReLU").
    std::string activationName() const { return activation_->name(); }

    /**
     * @brief Render the architecture as a compact human-readable string,
     *        e.g. "4-3-2".
     */
    std::string architectureString() const;

private:
    std::vector<std::size_t> layerSizes_;
    std::string activationName_;
    std::shared_ptr<IActivation> activation_;
    std::vector<Layer> layers_;

    /// @brief Validate the architecture vector; throws on invalid input.
    void validateArchitecture() const;
};

#endif // NETWORK_HPP
