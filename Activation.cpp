#include "Activation.hpp"
#include <stdexcept>
#include <algorithm>

namespace {
    std::string toLower(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(),
                        [](unsigned char c) { return std::tolower(c); });
        return result;
    }
}

std::unique_ptr<IActivation> IActivation::create(const std::string& activationName) {
    std::string lowered = toLower(activationName);

    if (lowered == "sigmoid") {
        return std::make_unique<Sigmoid>();
    }
    if (lowered == "relu") {
        return std::make_unique<ReLU>();
    }

    throw std::invalid_argument(
        "IActivation::create - unknown activation function '" + activationName +
        "'. Supported values are 'sigmoid' and 'relu'.");
}
