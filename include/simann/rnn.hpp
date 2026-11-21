#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace simann {
constexpr int VOCAB = 256;

struct Config {
    int hidden = 96;
    int sequence = 64;
    float learning_rate = 0.001f;
    float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
    float clip_norm = 1.0f;
    std::uint64_t seed = 1796;
};

class RNN {
public:
    explicit RNN(Config cfg = {});
    float train_chunk(const std::uint8_t* bytes, int length);
    float evaluate_chunk(const std::uint8_t* bytes, int length) const;
    std::vector<std::uint8_t> generate(const std::string& prompt, int count,
                                       float temperature = 0.85f, int top_k = 32);
    void save(const std::string& path) const;
    static RNN load(const std::string& path);
    int hidden_size() const { return H; }
    std::uint64_t parameters() const;
private:
    int H;
    Config cfg;
    std::vector<float> wxh, whh, why, bh, by;
    std::vector<float> m, v;
    std::uint64_t step = 0;
    std::mt19937_64 rng;

    std::vector<float> forward(std::uint8_t token, const std::vector<float>& prev,
                               std::vector<float>& hidden, std::vector<float>& probs) const;
    void adam_update(const std::vector<float>& grad);
};
}
