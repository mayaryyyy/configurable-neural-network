#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <cstddef>

/**
 * @brief A single labeled sample: an input feature vector and its
 *        corresponding target output vector.
 */
struct Sample {
    std::vector<double> input;
    std::vector<double> target;
};

/**
 * @brief Provides a fixed, reproducible 100-sample synthetic dataset and
 *        evaluation utilities (mean squared error).
 *
 * Responsibility: generate hardcoded input/target pairs sized to match a
 * given network's input and output dimensionality, and compute the mean
 * squared error between a network's predictions and the dataset's
 * targets. The generation is deterministic (fixed seed) so that results
 * are reproducible and comparable across the four benchmarked
 * architectures.
 */
class Dataset {
public:
    /**
     * @brief Construct a synthetic dataset of a fixed sample count.
     * @param inputSize Dimensionality of each input vector.
     * @param outputSize Dimensionality of each target vector.
     * @param sampleCount Number of samples to generate (100, per spec).
     * @param seed Seed for the deterministic pseudo-random generator,
     *        ensuring the same dataset is produced on every run.
     */
    Dataset(std::size_t inputSize, std::size_t outputSize,
            std::size_t sampleCount = 100, unsigned int seed = 2024);

    /// @brief Read-only access to all generated samples.
    const std::vector<Sample>& samples() const { return samples_; }

    /// @brief Number of samples in the dataset.
    std::size_t size() const { return samples_.size(); }

    /**
     * @brief Compute the mean squared error between a set of predicted
     *        output vectors and this dataset's target vectors.
     * @param predictions One predicted output vector per sample, in the
     *        same order as samples().
     * @return The mean squared error across all samples and all output
     *         dimensions.
     * @throws std::invalid_argument if predictions.size() does not match
     *         the number of samples, or if any predicted vector's size
     *         does not match the corresponding target's size.
     */
    double computeMSE(const std::vector<std::vector<double>>& predictions) const;

private:
    std::vector<Sample> samples_;

    void generateSamples(std::size_t inputSize, std::size_t outputSize,
                          std::size_t sampleCount, unsigned int seed);
};

#endif // DATASET_HPP
