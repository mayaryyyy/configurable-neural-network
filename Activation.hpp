#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>
#include <memory>
#include <string>

/**
 * @brief Abstract base class for activation function strategies.
 *
 * This follows the Strategy design pattern: each concrete activation
 * function (Sigmoid, ReLU, ...) implements the same interface so that
 * a Neuron can be configured with any activation function at runtime
 * without changing the Neuron's own logic.
 */
class IActivation {
public:
    virtual ~IActivation() = default;

    /**
     * @brief Apply the activation function to a raw weighted sum.
     * @param x The pre-activation value (weighted sum + bias).
     * @return The activated value.
     */
    virtual double activate(double x) const = 0;

    /**
     * @brief Return the human-readable name of this activation function.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Factory method: create an activation strategy by name.
     * @param activationName Either "sigmoid" or "relu" (case-insensitive).
     * @return A unique_ptr owning the requested activation strategy.
     * @throws std::invalid_argument if the name is not recognized.
     */
    static std::unique_ptr<IActivation> create(const std::string& activationName);
};

/**
 * @brief Logistic sigmoid activation: f(x) = 1 / (1 + e^-x).
 *
 * Squashes any real input into the open interval (0, 1). Smooth and
 * differentiable everywhere, but suffers from vanishing gradients for
 * large |x|.
 */
class Sigmoid : public IActivation {
public:
    double activate(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }

    std::string name() const override {
        return "Sigmoid";
    }
};

/**
 * @brief Rectified Linear Unit activation: f(x) = max(0, x).
 *
 * Cheap to compute and avoids vanishing gradients for positive inputs,
 * but can produce "dead" neurons if inputs are persistently negative.
 */
class ReLU : public IActivation {
public:
    double activate(double x) const override {
        return x > 0.0 ? x : 0.0;
    }

    std::string name() const override {
        return "ReLU";
    }
};

#endif // ACTIVATION_HPP
