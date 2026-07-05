/**
 * @file main.cpp
 * @brief Entry point for the Configurable Neural Network benchmark application.
 *
 * This program demonstrates a feedforward neural network whose architecture
 * (number of layers, neurons per layer, and activation function) is loaded
 * entirely from external text configuration files. No training takes place;
 * the program performs forward-pass inference only.
 *
 * Workflow:
 *   1. Load four architecture configurations from the config/ directory:
 *        - small.txt  (2 real layers)
 *        - medium.txt (3 real layers)
 *        - large.txt  (4 real layers)
 *        - wide.txt   (1 large hidden layer)
 *   2. For each configuration, build a Network and run a forward pass over
 *      a deterministic, hardcoded 100-sample dataset.
 *   3. Measure forward-pass execution time using std::chrono.
 *   4. Compute the mean squared error against the dataset's targets.
 *   5. Print a results table: architecture vs. MSE vs. execution time.
 */

#include <iostream>
#include <vector>
#include <string>
#include <exception>

#include "ConfigParser.hpp"
#include "BenchmarkRunner.hpp"

namespace {

/**
 * @brief Attempt to load a single named configuration file, printing a
 *        clear diagnostic and returning false on failure rather than
 *        crashing the whole benchmark run.
 */
bool loadConfig(const std::string& path, const std::string& displayLabel,
                 std::vector<NetworkConfig>& outConfigs) {
    try {
        NetworkConfig config = ConfigParser::parseFile(path);
        config.label = displayLabel;
        outConfigs.push_back(config);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config '" << path << "': " << e.what() << "\n";
        return false;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::cout << "==============================================\n";
    std::cout << " Configurable Neural Network in C++ (OOP)\n";
    std::cout << " Forward-pass only -- architectures from config\n";
    std::cout << "==============================================\n";

    // Allow an optional config directory to be passed on the command
    // line; defaults to "config" relative to the working directory,
    // which matches how the project is built and run via CMake.
    std::string configDir = (argc > 1) ? argv[1] : "config";

    std::vector<NetworkConfig> configs;

    loadConfig(configDir + "/small.txt", "Small (2 layers)", configs);
    loadConfig(configDir + "/medium.txt", "Medium (3 layers)", configs);
    loadConfig(configDir + "/large.txt", "Large (4 layers)", configs);
    loadConfig(configDir + "/wide.txt", "Wide (1 large layer)", configs);

    if (configs.empty()) {
        std::cerr << "\nNo configuration files could be loaded from '"
                  << configDir << "'. Aborting.\n";
        std::cerr << "Usage: " << argv[0] << " [path-to-config-directory]\n";
        return 1;
    }

    std::cout << "\nLoaded " << configs.size() << " architecture configuration(s) from '"
              << configDir << "'.\n";

    BenchmarkRunner runner;
    std::vector<BenchmarkResult> results = runner.runAll(configs);

    BenchmarkRunner::printReport(results);

    std::cout << "\nAll forward passes completed successfully.\n";

    return 0;
}
