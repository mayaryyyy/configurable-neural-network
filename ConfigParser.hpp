#ifndef CONFIG_PARSER_HPP
#define CONFIG_PARSER_HPP

#include <vector>
#include <string>
#include <cstddef>

/**
 * @brief Plain data structure describing a parsed network configuration.
 */
struct NetworkConfig {
    std::string label;                  ///< Human-readable name, e.g. "Small".
    std::vector<std::size_t> layerSizes; ///< Includes input size as element 0.
    std::string activation;             ///< "sigmoid" or "relu".
};

/**
 * @brief Parses network architecture configuration from text files.
 *
 * Responsibility: read a configuration file in the format
 *
 *     layers: 3
 *     sizes: 4 3 2
 *     activation: sigmoid
 *
 * and produce a NetworkConfig describing the resulting architecture.
 * This class is solely concerned with parsing and validating the file
 * format; it knows nothing about how the Network class uses the data.
 */
class ConfigParser {
public:
    /**
     * @brief Parse a configuration file from disk.
     * @param filePath Path to the configuration text file.
     * @return A populated NetworkConfig. The "label" field is set to the
     *         filename (without extension) by default.
     * @throws std::runtime_error if the file cannot be opened, is
     *         malformed, or the declared layer count does not match the
     *         number of sizes provided.
     */
    static NetworkConfig parseFile(const std::string& filePath);

    /**
     * @brief Parse configuration directly from an in-memory string.
     *
     * Useful for unit testing without needing files on disk, and reused
     * internally by parseFile().
     * @param content The full text content to parse.
     * @return A populated NetworkConfig (label left empty; caller may set it).
     * @throws std::runtime_error on malformed content.
     */
    static NetworkConfig parseString(const std::string& content);
};

#endif // CONFIG_PARSER_HPP
