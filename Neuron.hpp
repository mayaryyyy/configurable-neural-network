#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <memory>
#include "Activation.hpp"

/**
 * @brief Represents a single artificial neuron.
 *
 * A Neuron owns its own weight vector, bias term, and a reference to an
 * activation strategy (shared across the layer it belongs to, since the
 * activation function is a per-layer configuration, not a per-neuron one).
 *
 * Responsibility: given an input vector, compute the weighted sum plus
 * bias, then pass it through the activation function to produce a single
 * scalar output. This is a pure forward-pass neuron; no training/backprop
 * logic is implemented, per the project specification.
 */
class Neuron {
public:
    /**
     * @brief Construct a neuron with deterministic pseudo-random weights.
     * @param numInputs Number of inputs this neuron receives (i.e. the
     *        size of the previous layer, or the size of the network's
     *        input vector for first-layer neurons).
     * @param activation Shared pointer to the activation strategy used by
     *        this neuron's layer.
     * @param seed A seed value used to deterministically vary the
     *        generated weights between neurons, so that runs are
     *        reproducible across program executions.
     */
    Neuron(std::size_t numInputs,
           std::shared_ptr<IActivation> activation,
           unsigned int seed);

    /**
     * @brief Compute the forward pass output of this neuron.
     * @param inputs The input vector from the previous layer (or the
     *        network's input vector, for first-layer neurons).
     * @return The activated scalar output.
     * @throws std::invalid_argument if inputs.size() does not match the
     *         number of weights this neuron was constructed with.
     */
    double forward(const std::vector<double>& inputs) const;

    /// @brief Return the number of inputs (weights) this neuron expects.
    std::size_t inputSize() const { return weights_.size(); }

    /// @brief Read-only access to this neuron's weights (useful for inspection/testing).
    const std::vector<double>& weights() const { return weights_; }

    /// @brief Read-only access to this neuron's bias term.
    double bias() const { return bias_; }

private:
    std::vector<double> weights_;
    double bias_;
    std::shared_ptr<IActivation> activation_;

    /**
     * @brief Deterministically initialize weights and bias using a
     *        simple, reproducible pseudo-random scheme seeded by
     *        @p seed, scaled to a small range suitable for forward-pass
     *        demonstration (this project performs no training, so the
     *        initialization scheme need only be stable and reasonable,
     *        not statistically optimal like Xavier/He initialization).
     */
    void initializeParameters(unsigned int seed);
};

#endif // NEURON_HPP
