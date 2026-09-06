/**
 * @file config.h
 * @brief AudioVAE Configuration
 *
 * Configuration structure for the AudioVAE encoder/decoder.
 */

#ifndef VOXCPM_CONFIG_H
#define VOXCPM_CONFIG_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace voxcpm {

// =============================================================================
// AudioVAE Configuration
// =============================================================================

/**
 * @brief AudioVAE Configuration
 *
 * AudioVAE is a variational autoencoder for audio processing:
 * - Encoder: Audio waveform -> Latent representation (64-dim)
 * - Decoder: Latent representation -> Audio waveform
 *
 * Key parameters:
 * - encoder_rates: Downsampling factors for each encoder block
 *   hop_length = product of encoder_rates (e.g., 2*3*6*7*7 = 1764)
 * - decoder_rates: Upsampling factors for each decoder block (reverse order)
 */
struct AudioVAEConfig {
    // Dimensions
    int encoder_dim = 128;      // Upstream torch default encoder channel dimension
    int latent_dim = 64;        // Latent space dimension
    int decoder_dim = 1536;     // Upstream torch default decoder channel dimension

    // Sampling
    int sample_rate = 16000;    // Upstream torch default sample rate (Hz)
    int out_sample_rate = 0;    // Optional decode/output sample rate (Hz)
    std::vector<int> sr_bin_boundaries = {20000, 30000, 40000};

    // Encoder/Decoder rates (downsampling/upsampling factors)
    // Supports variable number of blocks via std::vector
    std::vector<int> encoder_rates = {2, 5, 8, 8};
    std::vector<int> decoder_rates = {8, 8, 5, 2};

    // Convolution settings
    bool depthwise = true;      // Use depthwise convolution

    // Noise injection (optional)
    bool use_noise_block = false;

    // Decoder-side sample-rate conditioning (AudioVAE v2)
    std::string cond_type = "scale_bias";
    int cond_dim = 128;
    bool cond_out_layer = false;

    /**
     * @brief Get hop length (total downsampling factor)
     * hop_length = product of all encoder_rates
     */
    int hop_length() const {
        int hop = 1;
        for (int r : encoder_rates) hop *= r;
        return hop;
    }

    /**
     * @brief Get decoder output step length (total upsampling factor)
     * decode_hop_length = product of all decoder_rates
     */
    int decode_hop_length() const {
        int hop = 1;
        for (int r : decoder_rates) hop *= r;
        return hop;
    }

    /**
     * @brief Get number of encoder blocks
     */
    int num_encoder_blocks() const { return static_cast<int>(encoder_rates.size()); }

    /**
     * @brief Get number of decoder blocks
     */
    int num_decoder_blocks() const { return static_cast<int>(decoder_rates.size()); }

    int output_sample_rate() const { return out_sample_rate > 0 ? out_sample_rate : sample_rate; }

    int sr_bin_bucket_count() const { return static_cast<int>(sr_bin_boundaries.size()) + 1; }

    int sample_rate_bucket(int sample_rate_hz) const {
        int bucket = 0;
        while (bucket < static_cast<int>(sr_bin_boundaries.size()) &&
               sample_rate_hz > sr_bin_boundaries[static_cast<size_t>(bucket)]) {
            ++bucket;
        }
        return bucket;
    }

    /**
     * @brief Calculate encoder channel progression
     * e.g., 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 for 5 blocks
     */
    std::vector<int> encoder_channels() const {
        std::vector<int> channels;
        channels.push_back(encoder_dim);
        int ch = encoder_dim;
        for (size_t i = 0; i < encoder_rates.size(); i++) {
            ch *= 2;
            channels.push_back(ch);
        }
        return channels;
    }

    /**
     * @brief Calculate decoder channel progression
     * e.g., 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 for 5 blocks
     */
    std::vector<int> decoder_channels() const {
        std::vector<int> channels;
        channels.push_back(decoder_dim);
        int ch = decoder_dim;
        for (size_t i = 0; i < decoder_rates.size(); i++) {
            ch /= 2;
            channels.push_back(ch);
        }
        return channels;
    }
};

}  // namespace voxcpm

#endif  // VOXCPM_CONFIG_H
