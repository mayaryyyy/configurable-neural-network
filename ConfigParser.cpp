#include "ConfigParser.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

namespace {
    /// @brief Trim leading/trailing whitespace from a string.
    std::string trim(const std::string& s) {
        const char* whitespace = " \t\r\n";
        std::size_t start = s.find_first_not_of(whitespace);
        if (start == std::string::npos) {
            return "";
        }
        std::size_t end = s.find_last_not_of(whitespace);
        return s.substr(start, end - start + 1);
    }

    std::string toLower(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(),
                        [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    /// @brief Extract the value portion after a "key:" prefix on a line.
    std::string extractValue(const std::string& line, const std::string& key) {
        std::size_t colonPos = line.find(':');
        if (colonPos == std::string::npos) {
            throw std::runtime_error(
                "ConfigParser: malformed line (missing ':') for key '" + key + "'");
        }
        return trim(line.substr(colonPos + 1));
    }
}

NetworkConfig ConfigParser::parseString(const std::string& content) {
    NetworkConfig config;
    config.activation = "sigmoid"; // sensible default if not specified

    std::istringstream stream(content);
    std::string line;

    bool sawLayers = false;
    bool sawSizes = false;
    std::size_t declaredLayerCount = 0;

    while (std::getline(stream, line)) {
        std::string trimmedLine = trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') {
            continue; // skip blank lines and comments
        }

        std::string lowered = toLower(trimmedLine);

        if (lowered.rfind("layers", 0) == 0) {
            std::string value = extractValue(trimmedLine, "layers");
            try {
                declaredLayerCount = static_cast<std::size_t>(std::stoul(value));
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "ConfigParser: invalid integer for 'layers': '" + value + "'");
            }
            sawLayers = true;
        } else if (lowered.rfind("sizes", 0) == 0) {
            std::string value = extractValue(trimmedLine, "sizes");
            std::istringstream sizeStream(value);
            std::size_t sizeValue;
            while (sizeStream >> sizeValue) {
                config.layerSizes.push_back(sizeValue);
            }
            sawSizes = true;
        } else if (lowered.rfind("activation", 0) == 0) {
            config.activation = toLower(extractValue(trimmedLine, "activation"));
        }
        // Unknown keys are silently ignored to keep the format forgiving.
    }

    if (!sawLayers) {
        throw std::runtime_error("ConfigParser: config is missing required 'layers:' field");
    }
    if (!sawSizes) {
        throw std::runtime_error("ConfigParser: config is missing required 'sizes:' field");
    }
    if (config.layerSizes.empty()) {
        throw std::runtime_error("ConfigParser: 'sizes:' field contained no values");
    }
    if (config.layerSizes.size() != declaredLayerCount) {
        throw std::runtime_error(
            "ConfigParser: declared 'layers: " + std::to_string(declaredLayerCount) +
            "' does not match the number of values in 'sizes:' (" +
            std::to_string(config.layerSizes.size()) + ")");
    }
    if (config.activation != "sigmoid" && config.activation != "relu") {
        throw std::runtime_error(
            "ConfigParser: unsupported activation '" + config.activation +
            "' (expected 'sigmoid' or 'relu')");
    }

    return config;
}

NetworkConfig ConfigParser::parseFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("ConfigParser: could not open config file '" + filePath + "'");
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    NetworkConfig config = parseString(buffer.str());

    std::filesystem::path p(filePath);
    config.label = p.stem().string();

    return config;
}
