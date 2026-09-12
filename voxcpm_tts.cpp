/**
 * @file voxcpm_tts.cpp
 * @brief Non-streaming VoxCPM TTS CLI
 */

#include "voxcpm/audio-vae.h"
#include "voxcpm/audio_io.h"
#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/tokenizer.h"
#include "voxcpm/voxcpm.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace voxcpm {
namespace {

struct Options {
    std::string text;
    std::string output_path;
    std::string stream_dir;
    std::string prompt_audio_path;
    std::string reference_audio_path;
    std::string prompt_text;
    std::string model_path;
    BackendType backend = BackendType::CPU;
    float cfg_value = 2.0f;
    int inference_timesteps = 10;
    int threads = 4;
    bool normalize = false;
    bool stream = false;
    bool retry_badcase = false;
    int retry_badcase_max_times = 3;
    float retry_badcase_ratio_threshold = 6.0f;
    // Pin the CFM sampling noise so two runs are byte-comparable. Unset means
    // seed from std::random_device, which is the historical behaviour.
    unsigned int seed = 0;
    bool has_seed = false;
};

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

struct PreparedInputs {
    std::vector<int32_t> full_text_tokens;
    std::vector<int32_t> text_mask;
    std::vector<int32_t> feat_mask;
    std::vector<float> feat;
    std::vector<float> prompt_feat;
    std::vector<float> reference_feat;
    int prompt_audio_length = 0;
    int reference_audio_length = 0;
    bool has_prompt_audio = false;
    bool has_reference_audio = false;
};

constexpr int32_t kAudioStartToken = 101;
constexpr int32_t kRefAudioStartToken = 103;
constexpr int32_t kRefAudioEndToken = 104;

enum class PaddingMode {
    Left,
    Right,
};

size_t skip_ascii_whitespace(const std::string& text, size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    return pos;
}

std::pair<std::string, bool> strip_hifi_control_prefix(const std::string& text) {
    const size_t start = skip_ascii_whitespace(text, 0);
    if (start >= text.size()) {
        return {text, false};
    }

    size_t content_start = std::string::npos;
    size_t close_pos = std::string::npos;
    size_t close_len = 0;
    if (text.compare(start, 1, "(") == 0) {
        content_start = start + 1;
        close_pos = text.find(')', content_start);
        close_len = 1;
    } else if (text.compare(start, 3, "（") == 0) {
        content_start = start + 3;
        close_pos = text.find("）", content_start);
        close_len = 3;
    }
    if (close_pos == std::string::npos) {
        return {text, false};
    }

    const size_t next = skip_ascii_whitespace(text, close_pos + close_len);
    return {text.substr(next), true};
}

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

const char* backend_type_name(BackendType type) {
    switch (type) {
        case BackendType::CPU:
            return "cpu";
        default:
            return "unknown";
    }
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return false;
    }

    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

int env_int_or_default(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }

    try {
        return std::max(1, std::stoi(raw));
    } catch (const std::exception&) {
        return default_value;
    }
}

int env_nonnegative_int_or_default(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }

    try {
        return std::max(0, std::stoi(raw));
    } catch (const std::exception&) {
        return default_value;
    }
}

double bytes_to_mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

size_t decode_state_kv_bytes(const VoxCPMDecodeState& state) {
    size_t total = 0;
    if (state.base_lm_cache) {
        total += state.base_lm_cache->buffer_size();
    }
    if (state.residual_lm_cache) {
        total += state.residual_lm_cache->buffer_size();
    }
    return total;
}

void log_memory_breakdown(bool enabled,
                          const char* stage,
                          const VoxCPMWeightStore& store,
                          const VoxCPMBackend& backend,
                          const VoxCPMDecodeState* state) {
    if (!enabled) {
        return;
    }

    const size_t weights = store.buffer_size();
    const size_t compute = backend.compute_buffer_size();
    const size_t kv = state ? decode_state_kv_bytes(*state) : 0;
    const size_t tracked_total = weights + compute + kv;

    std::cerr << std::fixed << std::setprecision(2)
              << "[memory] stage=" << stage
              << " weights=" << bytes_to_mib(weights) << " MiB"
              << " compute_arena=" << bytes_to_mib(compute) << " MiB"
              << " kv_cache=" << bytes_to_mib(kv) << " MiB"
              << " tracked_total=" << bytes_to_mib(tracked_total) << " MiB"
              << "\n";
}

std::string format_duration_compact(double seconds) {
    const int total_seconds = static_cast<int>(std::max(0.0, std::floor(seconds)));
    const int hours = total_seconds / 3600;
    const int minutes = (total_seconds % 3600) / 60;
    const int secs = total_seconds % 60;

    std::ostringstream oss;
    if (hours > 0) {
        oss << hours << ":" << std::setw(2) << std::setfill('0') << minutes
            << ":" << std::setw(2) << secs;
    } else {
        oss << minutes << ":" << std::setw(2) << std::setfill('0') << secs;
    }
    return oss.str();
}

class DecodeProgressPrinter {
public:
    explicit DecodeProgressPrinter(int total_steps)
        : total_steps_(std::max(1, total_steps)),
          start_time_(std::chrono::steady_clock::now()) {}

    void clear_line() {
        if (!has_rendered_) {
            return;
        }
        std::cerr << '\r' << std::string(last_width_, ' ') << '\r' << std::flush;
        has_rendered_ = false;
        last_width_ = 0;
    }

    void render(int completed_steps) {
        const int clamped_steps = std::clamp(completed_steps, 0, total_steps_);
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start_time_).count();
        const double safe_elapsed = std::max(elapsed, 1e-9);
        const double it_per_sec = clamped_steps > 0 ? static_cast<double>(clamped_steps) / safe_elapsed : 0.0;
        const double sec_per_it = clamped_steps > 0 ? safe_elapsed / static_cast<double>(clamped_steps) : 0.0;

        std::ostringstream oss;
        oss << "[decode] "
            << clamped_steps << "/" << total_steps_
            << " | elapsed " << format_duration_compact(elapsed)
            << " | " << std::fixed << std::setprecision(2) << it_per_sec << " it/s"
            << " | " << std::setprecision(3) << sec_per_it << " s/it";

        const std::string line = oss.str();
        last_width_ = std::max(last_width_, line.size());
        std::cerr << '\r' << line;
        if (line.size() < last_width_) {
            std::cerr << std::string(last_width_ - line.size(), ' ');
        }
        std::cerr << std::flush;
        has_rendered_ = true;
    }

    void finish(int completed_steps) {
        render(completed_steps);
        std::cerr << "\n";
        has_rendered_ = false;
        last_width_ = 0;
    }

private:
    int total_steps_ = 0;
    std::chrono::steady_clock::time_point start_time_;
    size_t last_width_ = 0;
    bool has_rendered_ = false;
};

BackendType parse_backend_type(const std::string& value) {
    if (value == "cpu") {
        return BackendType::CPU;
    }
    fail("Unsupported backend: " + value + " (expected cpu)");
}

void print_usage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0 << " --text TEXT --output OUTPUT --model-path MODEL.gguf [options]\n\n"
              << "Options:\n"
              << "  --text, -t TEXT\n"
              << "  --output, -o OUTPUT\n"
              << "  --prompt-audio, -pa PROMPT_AUDIO\n"
              << "  --prompt-text, -pt PROMPT_TEXT\n"
              << "  --reference-audio, -ra REFERENCE_AUDIO\n"
              << "  --cfg-value FLOAT (default: 2.0)\n"
              << "  --inference-timesteps INT (default: 10)\n"
              << "  --retry-badcase\n"
              << "  --retry-badcase-max-times INT (default: 3)\n"
              << "  --retry-badcase-ratio-threshold FLOAT (default: 6.0)\n"
              << "  --backend {cpu} (default: cpu; CPU-only build)\n"
              << "  --threads INT (default: 4)\n"
              << "  --seed UINT (default: random; pin it to make runs reproducible)\n"
              << "  --stream\n"
              << "  --stream-dir DIR\n"
              << "  --normalize\n"
              << "  --model-path GGUF\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--text" || arg == "-t") {
            options.text = require_value("--text");
        } else if (arg == "--output" || arg == "-o") {
            options.output_path = require_value("--output");
        } else if (arg == "--prompt-audio" || arg == "-pa") {
            options.prompt_audio_path = require_value("--prompt-audio");
        } else if (arg == "--prompt-text" || arg == "-pt") {
            options.prompt_text = require_value("--prompt-text");
        } else if (arg == "--reference-audio" || arg == "-ra" || arg == "--reference-wav-path") {
            options.reference_audio_path = require_value("--reference-audio");
        } else if (arg == "--cfg-value") {
            options.cfg_value = std::stof(require_value("--cfg-value"));
        } else if (arg == "--inference-timesteps") {
            options.inference_timesteps = std::stoi(require_value("--inference-timesteps"));
        } else if (arg == "--retry-badcase") {
            options.retry_badcase = true;
        } else if (arg == "--retry-badcase-max-times") {
            options.retry_badcase_max_times = std::stoi(require_value("--retry-badcase-max-times"));
        } else if (arg == "--retry-badcase-ratio-threshold") {
            options.retry_badcase_ratio_threshold = std::stof(require_value("--retry-badcase-ratio-threshold"));
        } else if (arg == "--backend") {
            options.backend = parse_backend_type(require_value("--backend"));
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--seed") {
            options.seed = static_cast<unsigned int>(std::stoul(require_value("--seed")));
            options.has_seed = true;
        } else if (arg == "--stream") {
            options.stream = true;
        } else if (arg == "--stream-dir") {
            options.stream_dir = require_value("--stream-dir");
        } else if (arg == "--normalize") {
            options.normalize = true;
        } else if (arg == "--model-path") {
            options.model_path = require_value("--model-path");
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fail("Unknown argument: " + arg);
        }
    }

    if (options.text.empty()) {
        fail("--text is required");
    }
    if (options.output_path.empty()) {
        fail("--output is required");
    }
    if (options.model_path.empty()) {
        fail("--model-path is required");
    }
    if ((options.prompt_audio_path.empty()) != (options.prompt_text.empty())) {
        fail("--prompt-audio and --prompt-text must be provided together");
    }
    if (!(0.1f <= options.cfg_value && options.cfg_value <= 10.0f)) {
        fail("--cfg-value must be between 0.1 and 10.0");
    }
    if (!(1 <= options.inference_timesteps && options.inference_timesteps <= 100)) {
        fail("--inference-timesteps must be between 1 and 100");
    }
    if (options.threads < 1) {
        fail("--threads must be >= 1");
    }
    if (options.retry_badcase_max_times < 1) {
        fail("--retry-badcase-max-times must be >= 1");
    }
    if (options.retry_badcase_ratio_threshold <= 0.0f) {
        fail("--retry-badcase-ratio-threshold must be > 0");
    }
    if (options.normalize) {
        fail("C++ text normalization not implemented");
    }
    if (options.stream && options.stream_dir.empty()) {
        fail("--stream requires --stream-dir");
    }

    const std::filesystem::path model_path(options.model_path);
    if (!std::filesystem::exists(model_path) || !std::filesystem::is_regular_file(model_path)) {
        fail("--model-path must point to an existing GGUF file");
    }
    if (!options.prompt_audio_path.empty() && !std::filesystem::exists(options.prompt_audio_path)) {
        fail("Prompt audio file does not exist: " + options.prompt_audio_path);
    }
    if (!options.reference_audio_path.empty() && !std::filesystem::exists(options.reference_audio_path)) {
        fail("Reference audio file does not exist: " + options.reference_audio_path);
    }
    if (options.retry_badcase && options.stream) {
        std::cerr << "Warning: retry_badcase is not supported in streaming mode; disabling retries.\n";
        options.retry_badcase = false;
    }

    return options;
}

uint16_t read_le_u16(std::istream& in) {
    uint8_t bytes[2] = {0, 0};
    in.read(reinterpret_cast<char*>(bytes), 2);
    return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t read_le_u32(std::istream& in) {
    uint8_t bytes[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(bytes), 4);
    return static_cast<uint32_t>(bytes[0] |
                                 (bytes[1] << 8) |
                                 (bytes[2] << 16) |
                                 (bytes[3] << 24));
}

WavData read_wav_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        fail("Failed to open WAV file: " + path);
    }

    char riff[4] = {0};
    char wave[4] = {0};
    in.read(riff, 4);
    const uint32_t riff_size = read_le_u32(in);
    (void)riff_size;
    in.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        fail("Invalid WAV header: " + path);
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<uint8_t> data_chunk;

    while (in && (!sample_rate || data_chunk.empty())) {
        char chunk_id[4] = {0};
        in.read(chunk_id, 4);
        if (in.gcount() != 4) {
            break;
        }
        const uint32_t chunk_size = read_le_u32(in);
        const std::string id(chunk_id, 4);

        if (id == "fmt ") {
            audio_format = read_le_u16(in);
            num_channels = read_le_u16(in);
            sample_rate = read_le_u32(in);
            const uint32_t byte_rate = read_le_u32(in);
            const uint16_t block_align = read_le_u16(in);
            (void)byte_rate;
            (void)block_align;
            bits_per_sample = read_le_u16(in);
            if (chunk_size > 16) {
                in.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
            }
        } else if (id == "data") {
            data_chunk.resize(chunk_size);
            in.read(reinterpret_cast<char*>(data_chunk.data()), static_cast<std::streamsize>(chunk_size));
        } else {
            in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }
        if (chunk_size % 2 != 0) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (sample_rate == 0 || num_channels == 0 || data_chunk.empty()) {
        fail("Incomplete WAV file: " + path);
    }
    if (audio_format != 1 && audio_format != 3) {
        fail("Unsupported WAV format in " + path + " (only PCM/float supported)");
    }

    const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample) / 8;
    if (bytes_per_sample == 0) {
        fail("Invalid bits-per-sample in WAV file: " + path);
    }

    const size_t frame_count = data_chunk.size() / (bytes_per_sample * num_channels);
    std::vector<float> samples(frame_count * num_channels, 0.0f);

    size_t offset = 0;
    for (size_t i = 0; i < frame_count * num_channels; ++i) {
        if (audio_format == 3 && bits_per_sample == 32) {
            float value = 0.0f;
            std::memcpy(&value, data_chunk.data() + offset, sizeof(float));
            samples[i] = value;
        } else if (audio_format == 1 && bits_per_sample == 16) {
            const int16_t value = static_cast<int16_t>(data_chunk[offset] | (data_chunk[offset + 1] << 8));
            samples[i] = static_cast<float>(value) / 32768.0f;
        } else if (audio_format == 1 && bits_per_sample == 24) {
            int32_t value = (static_cast<int32_t>(data_chunk[offset]) |
                             (static_cast<int32_t>(data_chunk[offset + 1]) << 8) |
                             (static_cast<int32_t>(data_chunk[offset + 2]) << 16));
            if (value & 0x800000) {
                value |= ~0xFFFFFF;
            }
            samples[i] = static_cast<float>(value) / 8388608.0f;
        } else if (audio_format == 1 && bits_per_sample == 32) {
            int32_t value = 0;
            std::memcpy(&value, data_chunk.data() + offset, sizeof(int32_t));
            samples[i] = static_cast<float>(value) / 2147483648.0f;
        } else {
            fail("Unsupported WAV bit depth in " + path);
        }
        offset += bytes_per_sample;
    }

    return WavData{
        static_cast<int>(sample_rate),
        static_cast<int>(num_channels),
        std::move(samples),
    };
}

std::vector<float> convert_to_mono(const WavData& wav) {
    if (wav.channels == 1) {
        return wav.samples;
    }

    const size_t frame_count = wav.samples.size() / static_cast<size_t>(wav.channels);
    std::vector<float> mono(frame_count, 0.0f);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        float sum = 0.0f;
        for (int channel = 0; channel < wav.channels; ++channel) {
            sum += wav.samples[frame * static_cast<size_t>(wav.channels) + static_cast<size_t>(channel)];
        }
        mono[frame] = sum / static_cast<float>(wav.channels);
    }
    return mono;
}

std::vector<float> linear_resample(const std::vector<float>& input, int src_rate, int dst_rate) {
    if (src_rate == dst_rate || input.empty()) {
        return input;
    }

    const double scale = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
    const size_t out_size = std::max<size_t>(1, static_cast<size_t>(std::llround(input.size() * scale)));
    std::vector<float> out(out_size, 0.0f);

    for (size_t i = 0; i < out_size; ++i) {
        const double src_pos = static_cast<double>(i) / scale;
        const size_t left = static_cast<size_t>(std::floor(src_pos));
        const size_t right = std::min(left + 1, input.size() - 1);
        const double frac = src_pos - static_cast<double>(left);
        out[i] = static_cast<float>((1.0 - frac) * input[left] + frac * input[right]);
    }

    return out;
}

void pad_audio_for_patch_alignment(std::vector<float>& audio, size_t patch_len, PaddingMode mode) {
    if (patch_len == 0 || audio.empty() || (audio.size() % patch_len) == 0) {
        return;
    }
    const size_t padding = patch_len - (audio.size() % patch_len);
    if (mode == PaddingMode::Left) {
        audio.insert(audio.begin(), padding, 0.0f);
    } else {
        audio.insert(audio.end(), padding, 0.0f);
    }
}

void write_wav_pcm16(const std::string& path, const std::vector<float>& audio, int sample_rate) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fail("Failed to open output WAV file: " + path);
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
    const uint16_t block_align = channels * (bits_per_sample / 8);
    const uint32_t data_size = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), 4);
    out.write("WAVE", 4);

    const uint32_t fmt_size = 16;
    const uint16_t audio_format = 1;
    out.write("fmt ", 4);
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);
    for (float sample : audio) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
        out.write(reinterpret_cast<const char*>(&pcm), sizeof(int16_t));
    }
}

std::string chunk_output_path(const std::string& stream_dir, int index) {
    std::ostringstream oss;
    oss << "chunk_" << std::setw(4) << std::setfill('0') << index << ".wav";
    return (std::filesystem::path(stream_dir) / oss.str()).string();
}

std::vector<float> extract_prompt_features(AudioVAE& audio_vae,
                                           VoxCPMBackend& backend,
                                           std::vector<float> audio,
                                           int sample_rate,
                                           int patch_size,
                                           int feat_dim,
                                           const char* label) {
    std::cerr << "Encoding " << label << " audio...\n";
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = audio_vae.encode(graph_ctx, backend, audio, sample_rate);
    if (!latent) {
        fail("Failed to build AudioVAE encode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, latent);
    backend.reserve_compute_memory(graph, "tts.audio_vae.encode");
    backend.alloc_graph(graph, "tts.audio_vae.encode");
    const auto& preprocessed = audio_vae.last_preprocessed_audio();
    backend.tensor_set(audio_vae.last_input_tensor(), preprocessed.data(), 0, preprocessed.size() * sizeof(float));
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE encode failed");
    }

    const int total_patches = static_cast<int>(latent->ne[0]);
    const int latent_dim = static_cast<int>(latent->ne[1]);
    if (latent_dim != feat_dim) {
        fail("Prompt latent dim mismatch");
    }
    if (total_patches % patch_size != 0) {
        fail("Prompt latent patches are not divisible by patch size");
    }

    std::vector<float> encoded(static_cast<size_t>(total_patches) * latent_dim);
    backend.tensor_get(latent, encoded.data(), 0, encoded.size() * sizeof(float));

    const int audio_length = total_patches / patch_size;
    std::vector<float> features(static_cast<size_t>(audio_length) * patch_size * feat_dim, 0.0f);
    for (int t = 0; t < audio_length; ++t) {
        for (int p = 0; p < patch_size; ++p) {
            const int patch_index = t * patch_size + p;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = static_cast<size_t>(d) * total_patches + patch_index;
                const size_t dst = (static_cast<size_t>(t) * patch_size + p) * feat_dim + d;
                features[dst] = encoded[src];
            }
        }
    }
    return features;
}

std::vector<float> load_audio_features(const std::string& path,
                                       PaddingMode padding_mode,
                                       AudioVAE& audio_vae,
                                       VoxCPMBackend& backend,
                                       int patch_size,
                                       int feat_dim,
                                       int patch_len,
                                       const char* label) {
    const WavData wav = read_wav_file(path);
    std::vector<float> mono = convert_to_mono(wav);
    mono = linear_resample(mono, wav.sample_rate, audio_vae.config().sample_rate);
    mono = trim_audio_silence_vad(mono, audio_vae.config().sample_rate);
    pad_audio_for_patch_alignment(mono, static_cast<size_t>(patch_len), padding_mode);
    return extract_prompt_features(audio_vae,
                                   backend,
                                   mono,
                                   audio_vae.config().sample_rate,
                                   patch_size,
                                   feat_dim,
                                   label);
}

void patch_major_to_latent(const std::vector<float>& frames,
                           int patch_size,
                           int feat_dim,
                           std::vector<float>& latent);

std::vector<float> patch_major_to_latent(const std::vector<float>& frames,
                                         int patch_size,
                                         int feat_dim);

std::vector<float> decode_audio(AudioVAE& audio_vae,
                                VoxCPMBackend& backend,
                                const std::vector<float>& features,
                                int total_patches,
                                int feat_dim) {
    std::cerr << "Decoding waveform from " << total_patches << " latent patches...\n";
    VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
    ggml_tensor* latent = graph_ctx.new_tensor_2d(GGML_TYPE_F32, total_patches, feat_dim);
    ggml_set_input(latent);
    ggml_tensor* audio = audio_vae.decode(graph_ctx, backend, latent);
    if (!audio) {
        fail("Failed to build AudioVAE decode graph");
    }

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, audio);
    backend.reserve_compute_memory(graph, "tts.audio_vae.decode");
    backend.alloc_graph(graph, "tts.audio_vae.decode");
    backend.tensor_set(latent, features.data(), 0, features.size() * sizeof(float));
    audio_vae.prepare_decode_inputs(backend);
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        fail("AudioVAE decode failed");
    }

    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

int stateful_audio_decode_patch_threshold(const AudioVAE& audio_vae) {
    constexpr int kDefaultCudaAudioDecodePatchThreshold = 2048;
    constexpr int kConditionedCudaAudioDecodePatchThreshold = 1024;
    const bool has_sr_conditioning = std::any_of(audio_vae.weights().decoder_blocks.begin(),
                                                 audio_vae.weights().decoder_blocks.end(),
                                                 [](const DecoderBlockWeights& block) {
                                                     return block.sr_cond.active();
                                                 });
    if (has_sr_conditioning || audio_vae.config().num_decoder_blocks() >= 6) {
        return kConditionedCudaAudioDecodePatchThreshold;
    }
    return kDefaultCudaAudioDecodePatchThreshold;
}

int stateful_audio_decode_max_window_patches(const AudioVAE& audio_vae) {
    constexpr int kDefaultMaxWindowPatches = 1024;
    constexpr int kConditionedMaxWindowPatches = 1024;
    const bool has_sr_conditioning = std::any_of(audio_vae.weights().decoder_blocks.begin(),
                                                 audio_vae.weights().decoder_blocks.end(),
                                                 [](const DecoderBlockWeights& block) {
                                                     return block.sr_cond.active();
                                                 });
    if (has_sr_conditioning || audio_vae.config().num_decoder_blocks() >= 6) {
        return kConditionedMaxWindowPatches;
    }
    return kDefaultMaxWindowPatches;
}

int stateful_audio_decode_chunk_frames(const AudioVAE& audio_vae, int patch_size) {
    const int max_window_frames =
        std::max(1, stateful_audio_decode_max_window_patches(audio_vae) / std::max(1, patch_size));
    const int requested_chunk_frames = env_int_or_default("VOXCPM_AUDIO_DECODE_CHUNK_FRAMES", 0);
    if (requested_chunk_frames > 0) {
        return std::min(requested_chunk_frames, max_window_frames);
    }
    return max_window_frames;
}

bool should_use_stateful_audio_decode(const VoxCPMBackend& backend,
                                      const AudioVAE& audio_vae,
                                      int total_patches) {
    return backend.is_gpu() &&
           total_patches >= stateful_audio_decode_patch_threshold(audio_vae) &&
           audio_vae.supports_streaming_decode(backend);
}

bool should_use_output_pool_final_decode(const VoxCPMDecodeState& state,
                                         bool has_reference_audio,
                                         int seq_len) {
    constexpr int kOutputPoolSeqLimit = 256;
    return state.output_pool != nullptr &&
           state.output_pool->is_initialized() &&
           !has_reference_audio &&
           seq_len <= kOutputPoolSeqLimit;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
    }
    return max_diff;
}

void append_stateful_decoded_chunk(std::vector<float>& waveform,
                                   std::vector<float>& chunk_waveform,
                                   int chunk_start,
                                   int new_frames,
                                   int skip_frames,
                                   int patch_len) {
    const int discard_frames = std::max(0, std::min(new_frames, skip_frames - chunk_start));
    const size_t discard = static_cast<size_t>(discard_frames) * static_cast<size_t>(patch_len);
    if (chunk_waveform.size() > discard) {
        chunk_waveform.erase(chunk_waveform.begin(),
                             chunk_waveform.begin() + static_cast<std::ptrdiff_t>(discard));
    } else {
        chunk_waveform.clear();
    }

    const int keep_frames = std::max(0, new_frames - discard_frames);
    const size_t keep = static_cast<size_t>(keep_frames) * static_cast<size_t>(patch_len);
    if (chunk_waveform.size() > keep) {
        chunk_waveform.resize(keep);
    }
    waveform.insert(waveform.end(), chunk_waveform.begin(), chunk_waveform.end());
}

std::vector<float> decode_audio_stateful_from_patch_major_frames(AudioVAE& audio_vae,
                                                                 VoxCPMBackend& backend,
                                                                 const std::vector<float>& frames,
                                                                 int skip_frames,
                                                                 int chunk_frames,
                                                                 int patch_size,
                                                                 int feat_dim,
                                                                 int patch_len) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    if (frames.empty() || (frames.size() % frame_stride) != 0 || skip_frames < 0 || skip_frames > total_frames) {
        return {};
    }

    AudioVAEStreamingDecodeState stream_state;
    if (!audio_vae.initialize_streaming_decode_state(backend, stream_state)) {
        return {};
    }

    std::vector<float> waveform;
    waveform.reserve(static_cast<size_t>(std::max(0, total_frames - skip_frames)) * static_cast<size_t>(patch_len));
    for (int chunk_start = 0; chunk_start < total_frames; chunk_start += chunk_frames) {
        const int new_frames = std::min(chunk_frames, total_frames - chunk_start);
        const int total_patches = new_frames * patch_size;
        const size_t begin = static_cast<size_t>(chunk_start) * frame_stride;
        const size_t end = static_cast<size_t>(chunk_start + new_frames) * frame_stride;
        std::vector<float> chunk_frames_host(frames.begin() + static_cast<std::ptrdiff_t>(begin),
                                             frames.begin() + static_cast<std::ptrdiff_t>(end));

        VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);
        ggml_tensor* patch_major = graph_ctx.new_tensor_2d(GGML_TYPE_F32, feat_dim, total_patches);
        ggml_set_input(patch_major);
        ggml_tensor* latent = ggml_cont(graph_ctx.raw_context(), ggml_transpose(graph_ctx.raw_context(), patch_major));
        ggml_tensor* audio = audio_vae.decode_streaming(graph_ctx, backend, latent, stream_state);
        if (!audio) {
            fail("Failed to build stateful AudioVAE decode graph from patch-major frames");
        }

        ggml_cgraph* graph = graph_ctx.new_graph();
        graph_ctx.build_forward(graph, audio);
        stream_state.build_update_graph(graph);
        backend.reserve_compute_memory(graph, "tts.audio_vae.decode.stateful.patch_major");
        backend.alloc_graph(graph, "tts.audio_vae.decode.stateful.patch_major");
        backend.tensor_set(patch_major, chunk_frames_host.data(), 0, chunk_frames_host.size() * sizeof(float));
        audio_vae.prepare_decode_inputs(backend);
        if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
            fail("Stateful AudioVAE decode from patch-major frames failed");
        }
        stream_state.publish_updates(backend);

        std::vector<float> chunk_waveform(static_cast<size_t>(ggml_nelements(audio)));
        backend.tensor_get(audio, chunk_waveform.data(), 0, chunk_waveform.size() * sizeof(float));
        append_stateful_decoded_chunk(waveform, chunk_waveform, chunk_start, new_frames, skip_frames, patch_len);
    }
    return waveform;
}

std::vector<float> decode_audio_stateful_from_output_pool(AudioVAE& audio_vae,
                                                          VoxCPMBackend& backend,
                                                          const VoxCPMOutputPool& output_pool,
                                                          int frame_offset,
                                                          int frame_count,
                                                          int skip_frames,
                                                          int chunk_frames,
                                                          int patch_size,
                                                          int feat_dim,
                                                          int patch_len) {
    if (frame_count <= 0 || skip_frames < 0 || skip_frames > frame_count) {
        return {};
    }
    if (output_pool.shape().feat_dim != feat_dim || output_pool.shape().patch_size != patch_size) {
        fail("Output pool shape does not match stateful AudioVAE decode request");
    }

    AudioVAEStreamingDecodeState stream_state;
    if (!audio_vae.initialize_streaming_decode_state(backend, stream_state)) {
        return {};
    }

    std::vector<float> waveform;
    waveform.reserve(static_cast<size_t>(std::max(0, frame_count - skip_frames)) * static_cast<size_t>(patch_len));
    for (int chunk_start = 0; chunk_start < frame_count; chunk_start += chunk_frames) {
        const int new_frames = std::min(chunk_frames, frame_count - chunk_start);

        VoxCPMContext graph_ctx(ContextType::Graph, 65536, 262144);
        ggml_tensor* latent =
            output_pool.make_audio_vae_latent_view(graph_ctx.raw_context(), frame_offset + chunk_start, new_frames);
        if (latent == nullptr) {
            return {};
        }
        ggml_tensor* audio = audio_vae.decode_streaming(graph_ctx, backend, latent, stream_state);
        if (!audio) {
            fail("Failed to build stateful AudioVAE decode graph from output pool");
        }

        ggml_cgraph* graph = graph_ctx.new_graph();
        graph_ctx.build_forward(graph, audio);
        stream_state.build_update_graph(graph);
        backend.reserve_compute_memory(graph, "tts.audio_vae.decode.stateful.output_pool");
        backend.alloc_graph(graph, "tts.audio_vae.decode.stateful.output_pool");
        audio_vae.prepare_decode_inputs(backend);
        if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
            fail("Stateful AudioVAE decode from output pool failed");
        }
        stream_state.publish_updates(backend);

        std::vector<float> chunk_waveform(static_cast<size_t>(ggml_nelements(audio)));
        backend.tensor_get(audio, chunk_waveform.data(), 0, chunk_waveform.size() * sizeof(float));
        append_stateful_decoded_chunk(waveform, chunk_waveform, chunk_start, new_frames, skip_frames, patch_len);
    }
    return waveform;
}

void fill_noise(std::vector<float>& noise, int patch_size, int feat_dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    noise.resize(static_cast<size_t>(patch_size) * feat_dim);
    for (float& value : noise) {
        value = dist(rng);
    }
}

std::vector<float> build_decode_feature_sequence(const std::vector<float>& prompt_feat,
                                                 int prompt_audio_length,
                                                 const std::vector<float>& generated_steps,
                                                 int streaming_prefix_len,
                                                 int patch_size,
                                                 int feat_dim,
                                                 int* prepended_context_frames) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;

    int context_frames = 0;
    if (!prompt_feat.empty() && prompt_audio_length > 0 && streaming_prefix_len > 1) {
        context_frames = std::min(streaming_prefix_len - 1, prompt_audio_length);
    }

    std::vector<float> decode_frames;
    decode_frames.reserve(static_cast<size_t>(context_frames) * frame_stride + generated_steps.size());
    if (context_frames > 0) {
        const size_t context_offset = static_cast<size_t>(prompt_audio_length - context_frames) * frame_stride;
        decode_frames.insert(decode_frames.end(),
                             prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                             prompt_feat.end());
    }
    decode_frames.insert(decode_frames.end(), generated_steps.begin(), generated_steps.end());

    if (prepended_context_frames != nullptr) {
        *prepended_context_frames = context_frames;
    }
    return decode_frames;
}

void patch_major_to_latent(const std::vector<float>& frames,
                           int patch_size,
                           int feat_dim,
                           std::vector<float>& latent) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const int total_frames = static_cast<int>(frames.size() / frame_stride);
    const int total_patches = total_frames * patch_size;
    latent.assign(static_cast<size_t>(total_patches) * feat_dim, 0.0f);
    for (int frame = 0; frame < total_frames; ++frame) {
        for (int patch = 0; patch < patch_size; ++patch) {
            const int time_index = frame * patch_size + patch;
            for (int d = 0; d < feat_dim; ++d) {
                const size_t src = (static_cast<size_t>(frame) * patch_size + patch) * feat_dim + d;
                const size_t dst = static_cast<size_t>(d) * total_patches + time_index;
                latent[dst] = frames[src];
            }
        }
    }
}

std::vector<float> patch_major_to_latent(const std::vector<float>& frames,
                                         int patch_size,
                                         int feat_dim) {
    std::vector<float> latent;
    patch_major_to_latent(frames, patch_size, feat_dim, latent);
    return latent;
}

void append_stream_frame(std::vector<float>& recent_frames,
                         const std::vector<float>& patch,
                         int max_frames,
                         int patch_size,
                         int feat_dim) {
    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    recent_frames.insert(recent_frames.end(), patch.begin(), patch.end());
    const size_t max_elems = static_cast<size_t>(max_frames) * frame_stride;
    if (recent_frames.size() > max_elems) {
        recent_frames.erase(recent_frames.begin(),
                            recent_frames.begin() + static_cast<std::ptrdiff_t>(recent_frames.size() - max_elems));
    }
}

void maybe_profile_front_half(bool enabled,
                              VoxCPMRuntime& runtime,
                              const VoxCPMDecodeState& state,
                              int patch_size,
                              int feat_dim,
                              int inference_timesteps,
                              float cfg_value) {
    if (!enabled) {
        return;
    }

    constexpr int kWarmupIters = 1;
    constexpr int kMeasureIters = 3;

    std::mt19937 rng(1234);
    std::vector<float> noise;
    fill_noise(noise, patch_size, feat_dim, rng);

    volatile float checksum = 0.0f;
    const auto measure_ms = [&](auto&& fn) {
        for (int i = 0; i < kWarmupIters; ++i) {
            checksum += fn();
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kMeasureIters; ++i) {
            checksum += fn();
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() /
            static_cast<double>(kMeasureIters);
    };

    const double lm_proj_ms = measure_ms([&]() {
        const std::vector<float> out = runtime.benchmark_run_lm_to_dit_projection(state.lm_hidden);
        return out.empty() ? 0.0f : out.front();
    });

    const double res_proj_ms = measure_ms([&]() {
        const std::vector<float> out = runtime.benchmark_run_res_to_dit_projection(state.residual_hidden);
        return out.empty() ? 0.0f : out.front();
    });

    const std::vector<float> lm_proj = runtime.benchmark_run_lm_to_dit_projection(state.lm_hidden);
    const std::vector<float> res_proj = runtime.benchmark_run_res_to_dit_projection(state.residual_hidden);

    const double host_add_ms = measure_ms([&]() {
        std::vector<float> mu = lm_proj;
        for (size_t i = 0; i < mu.size(); ++i) {
            mu[i] += res_proj[i];
        }
        return mu.empty() ? 0.0f : mu.front();
    });

    std::vector<float> mu = lm_proj;
    for (size_t i = 0; i < mu.size(); ++i) {
        mu[i] += res_proj[i];
    }

    const double unified_cfm_ms = measure_ms([&]() {
        const std::vector<float> out = runtime.benchmark_run_unified_cfm(
            noise, mu, state.prefix_feat_cond, inference_timesteps, cfg_value);
        return out.empty() ? 0.0f : out.front();
    });

    const double fused_front_half_ms = measure_ms([&]() {
        const std::vector<float> out = runtime.benchmark_run_decode_front_half(
            noise,
            state.lm_hidden,
            state.residual_hidden,
            state.prefix_feat_cond,
            inference_timesteps,
            cfg_value);
        return out.empty() ? 0.0f : out.front();
    });

    const double split_total_ms = lm_proj_ms + res_proj_ms + host_add_ms + unified_cfm_ms;

    std::cerr << std::fixed << std::setprecision(3)
              << "\n=== Front-Half Profile ===\n"
              << "  lm_to_dit_proj:      " << lm_proj_ms << " ms\n"
              << "  res_to_dit_proj:     " << res_proj_ms << " ms\n"
              << "  host_add(mu):        " << host_add_ms << " ms\n"
              << "  unified_cfm:         " << unified_cfm_ms << " ms\n"
              << "  -----------------------------\n"
              << "  split_total:         " << split_total_ms << " ms\n"
              << "  fused_front_half:    " << fused_front_half_ms << " ms\n"
              << "  fused_vs_split_gap:  " << (fused_front_half_ms - split_total_ms) << " ms\n";

    if (checksum == std::numeric_limits<float>::infinity()) {
        std::cerr << "[front_half_profile] checksum overflow\n";
    }
}

PreparedInputs prepare_inputs(const Options& options,
                              const std::string& effective_text,
                              ChineseCharSplitTokenizer& split_tokenizer,
                              AudioVAE& audio_vae,
                              VoxCPMBackend& backend,
                              int patch_size,
                              int feat_dim,
                              int patch_len) {
    PreparedInputs prepared;

    if (!options.reference_audio_path.empty()) {
        prepared.has_reference_audio = true;
        prepared.reference_feat = load_audio_features(options.reference_audio_path,
                                                      PaddingMode::Right,
                                                      audio_vae,
                                                      backend,
                                                      patch_size,
                                                      feat_dim,
                                                      patch_len,
                                                      "reference");
        prepared.reference_audio_length =
            static_cast<int>(prepared.reference_feat.size() / static_cast<size_t>(patch_size * feat_dim));
    }
    if (!options.prompt_audio_path.empty()) {
        prepared.has_prompt_audio = true;
        prepared.prompt_feat = load_audio_features(options.prompt_audio_path,
                                                   PaddingMode::Left,
                                                   audio_vae,
                                                   backend,
                                                   patch_size,
                                                   feat_dim,
                                                   patch_len,
                                                   "prompt");
        prepared.prompt_audio_length =
            static_cast<int>(prepared.prompt_feat.size() / static_cast<size_t>(patch_size * feat_dim));
    }

    const std::string main_text = prepared.has_prompt_audio ? options.prompt_text + effective_text : effective_text;
    std::vector<int32_t> text_tokens = split_tokenizer.encode(main_text, false);
    text_tokens.push_back(kAudioStartToken);

    const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
    const size_t total_frames = static_cast<size_t>(text_tokens.size()) +
                                static_cast<size_t>(prepared.prompt_audio_length) +
                                (prepared.has_reference_audio
                                     ? static_cast<size_t>(prepared.reference_audio_length) + 2
                                     : 0);
    prepared.full_text_tokens.reserve(total_frames);
    prepared.text_mask.reserve(total_frames);
    prepared.feat_mask.reserve(total_frames);
    prepared.feat.reserve(total_frames * frame_stride);

    const auto append_zero_frame = [&]() {
        prepared.feat.insert(prepared.feat.end(), frame_stride, 0.0f);
    };
    const auto append_feat_frames = [&](const std::vector<float>& frames, int frame_count) {
        prepared.feat.insert(prepared.feat.end(),
                             frames.begin(),
                             frames.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(frame_count) * frame_stride));
    };

    if (prepared.has_reference_audio) {
        prepared.full_text_tokens.push_back(kRefAudioStartToken);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();

        for (int i = 0; i < prepared.reference_audio_length; ++i) {
            prepared.full_text_tokens.push_back(0);
            prepared.text_mask.push_back(0);
            prepared.feat_mask.push_back(1);
        }
        append_feat_frames(prepared.reference_feat, prepared.reference_audio_length);

        prepared.full_text_tokens.push_back(kRefAudioEndToken);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();
    }

    for (int32_t token : text_tokens) {
        prepared.full_text_tokens.push_back(token);
        prepared.text_mask.push_back(1);
        prepared.feat_mask.push_back(0);
        append_zero_frame();
    }

    if (prepared.has_prompt_audio) {
        for (int i = 0; i < prepared.prompt_audio_length; ++i) {
            prepared.full_text_tokens.push_back(0);
            prepared.text_mask.push_back(0);
            prepared.feat_mask.push_back(1);
        }
        append_feat_frames(prepared.prompt_feat, prepared.prompt_audio_length);
    }
    return prepared;
}

}  // namespace
}  // namespace voxcpm

int main(int argc, char** argv) {
    using namespace voxcpm;

    try {
        const Options options = parse_args(argc, argv);
        constexpr int kStreamingPrefixLen = 4;
        const bool log_memory = env_flag_enabled("VOXCPM_LOG_MEMORY_BREAKDOWN");
        const bool log_decode_memory = env_flag_enabled("VOXCPM_LOG_DECODE_MEMORY");
        const int log_decode_memory_every = env_int_or_default("VOXCPM_LOG_DECODE_MEMORY_EVERY", 1);
        const bool profile_front_half = env_flag_enabled("VOXCPM_PROFILE_FRONT_HALF");

        VoxCPMBackend backend(options.backend, options.threads);
        std::cerr << "Using backend: " << backend_type_name(backend.type())
                  << " (" << backend.backend_name();
        if (std::strlen(backend.backend_description()) > 0) {
            std::cerr << " | " << backend.backend_description();
        }
        std::cerr << ")\n";
        std::cerr << "Loading GGUF from " << options.model_path << " with " << options.threads
                  << " threads...\n";
        auto store = std::make_shared<VoxCPMWeightStore>();
        if (!store->load_from_file(options.model_path, backend)) {
            fail("Failed to load GGUF: " + options.model_path);
        }

        VoxCPMRuntime runtime;
        if (!runtime.load_from_store(store, backend)) {
            fail("Failed to initialize VoxCPM runtime from GGUF");
        }

        AudioVAE audio_vae;
        if (!audio_vae.load_from_store(store)) {
            fail("Failed to initialize AudioVAE from GGUF");
        }

        VoxCPMTokenizer tokenizer;
        if (!tokenizer.load_from_store(*store)) {
            fail("Failed to load tokenizer metadata from GGUF");
        }
        ChineseCharSplitTokenizer split_tokenizer(tokenizer);
        std::cerr << "Tokenizer loaded from GGUF metadata.\n";
        log_memory_breakdown(log_memory, "post_load", *store, backend, nullptr);

        const int patch_size = runtime.config().patch_size;
        const int feat_dim = runtime.config().feat_dim;
        const int encode_patch_len = patch_size * audio_vae.config().hop_length();
        const int decode_patch_len = patch_size * audio_vae.config().decode_hop_length();

        // Detailed timing breakdown
        const auto encode_start = std::chrono::steady_clock::now();
        std::string effective_text = options.text;
        if (!options.prompt_audio_path.empty()) {
            const auto [stripped_text, stripped] = strip_hifi_control_prefix(options.text);
            if (stripped) {
                std::cerr << "Hi-Fi mode ignores control instructions; stripping the leading parenthesized prefix.\n";
                effective_text = stripped_text;
            }
        }
        const PreparedInputs prepared = prepare_inputs(
            options, effective_text, split_tokenizer, audio_vae, backend, patch_size, feat_dim, encode_patch_len);
        const auto encode_end = std::chrono::steady_clock::now();
        const double vae_encode_time = std::chrono::duration<double>(encode_end - encode_start).count();

        log_memory_breakdown(log_memory, "post_prompt_encode", *store, backend, nullptr);
        const int target_text_token_count =
            std::max<int>(1, static_cast<int>(split_tokenizer.tokenize(effective_text).size()));
        const int benchmark_decode_steps = env_nonnegative_int_or_default("VOXCPM_BENCHMARK_DECODE_STEPS", 0);
        const bool benchmark_ignore_stop = env_flag_enabled("VOXCPM_BENCHMARK_IGNORE_STOP");
        const bool disable_output_pool_final_decode = env_flag_enabled("VOXCPM_DISABLE_OUTPUT_POOL_FINAL_DECODE");
        const bool compare_output_pool_latent = env_flag_enabled("VOXCPM_COMPARE_OUTPUT_POOL_LATENT");
        const int natural_max_len =
            std::min(static_cast<int>(target_text_token_count * options.retry_badcase_ratio_threshold + 10.0f), 2000);
        const int max_len = benchmark_decode_steps > 0 ? benchmark_decode_steps : natural_max_len;
        constexpr int kMinLen = 2;
        const auto model_start = std::chrono::steady_clock::now();
        std::vector<float> waveform;
        VoxCPMDecodeState final_state;
        double vae_decode_time = 0.0;
        int generated_frames = 0;
        int prepended_context_frames = 0;

        const int max_attempts = options.retry_badcase ? std::max(1, options.retry_badcase_max_times) : 1;
        // max_attempts =1 
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            const int seq_len = static_cast<int>(prepared.full_text_tokens.size());
            std::cerr << "Running prefill, seq_len=" << seq_len << " (attempt " << (attempt + 1)
                      << "/" << max_attempts << ")...\n";
            // 预填充
            VoxCPMDecodeState state = runtime.prefill(prepared.full_text_tokens,
                                                     prepared.text_mask,
                                                     prepared.feat,
                                                     prepared.feat_mask,
                                                     seq_len,
                                                     kStreamingPrefixLen);
            log_memory_breakdown(log_memory, "post_prefill", *store, backend, &state);
            if (attempt == 0) {
                maybe_profile_front_half(profile_front_half,
                                         runtime,
                                         state,
                                         patch_size,
                                         feat_dim,
                                         options.inference_timesteps,
                                         options.cfg_value);
            }

            std::mt19937 rng(options.has_seed ? options.seed : std::random_device{}());
            std::vector<float> generated_steps;
            generated_steps.reserve(static_cast<size_t>(max_len) * patch_size * feat_dim);
            std::vector<float> noise;
            std::vector<float> stream_recent_frames;
            std::vector<float> stream_latent;
            if (options.stream) {
                std::filesystem::create_directories(options.stream_dir);
                const size_t frame_stride = static_cast<size_t>(patch_size) * feat_dim;
                const int context_frames = (prepared.has_prompt_audio && prepared.prompt_audio_length > 0 && kStreamingPrefixLen > 1)
                    ? std::min(kStreamingPrefixLen - 1, prepared.prompt_audio_length)
                    : 0;
                if (context_frames > 0) {
                    const size_t context_offset = static_cast<size_t>(prepared.prompt_audio_length - context_frames) * frame_stride;
                    stream_recent_frames.insert(stream_recent_frames.end(),
                                                prepared.prompt_feat.begin() + static_cast<std::ptrdiff_t>(context_offset),
                                                prepared.prompt_feat.end());
                }
            }
            // 解码循环
            std::cerr << "Running decode loop, max_len=" << max_len;
            if (benchmark_decode_steps > 0) {
                std::cerr << " (benchmark_steps=" << benchmark_decode_steps
                          << ", ignore_stop=" << (benchmark_ignore_stop ? 1 : 0) << ")";
            }
            std::cerr << "...\n";
            DecodeProgressPrinter decode_progress(max_len);
            int stop_step_observed = -1;
            for (int step = 0; step < max_len; ++step) {
                fill_noise(noise, patch_size, feat_dim, rng);
                VoxCPMDecodeResult result = runtime.decode(std::move(state),
                                                           noise,
                                                           options.inference_timesteps,
                                                           options.cfg_value);
                generated_steps.insert(generated_steps.end(), result.output_0.begin(), result.output_0.end());
                state = std::move(result.output_1);
                decode_progress.render(step + 1);

                if (log_decode_memory && ((step + 1) % log_decode_memory_every == 0 || result.output_2)) {
                    decode_progress.clear_line();
                    const std::string stage = "decode_step_" + std::to_string(step + 1);
                    log_memory_breakdown(true, stage.c_str(), *store, backend, &state);
                    decode_progress.render(step + 1);
                }

                if (options.stream) {
                    append_stream_frame(stream_recent_frames,
                                        result.output_0,
                                        kStreamingPrefixLen,
                                        patch_size,
                                        feat_dim);
                    const int recent_frame_count =
                        static_cast<int>(stream_recent_frames.size() / static_cast<size_t>(patch_size * feat_dim));
                    const int recent_patches = recent_frame_count * patch_size;
                    if (recent_patches > 0) {
                        patch_major_to_latent(stream_recent_frames, patch_size, feat_dim, stream_latent);
                        std::vector<float> chunk_waveform = decode_audio(audio_vae, backend, stream_latent, recent_patches, feat_dim);
                        if (chunk_waveform.size() > static_cast<size_t>(decode_patch_len)) {
                            chunk_waveform.erase(chunk_waveform.begin(),
                                                 chunk_waveform.end() - static_cast<std::ptrdiff_t>(decode_patch_len));
                        }
                        write_wav_pcm16(chunk_output_path(options.stream_dir, step),
                                        chunk_waveform,
                                        audio_vae.config().output_sample_rate());
                    }
                }

                if (step > kMinLen && result.output_2) {
                    const bool first_stop_observation = stop_step_observed < 0;
                    if (first_stop_observation) {
                        stop_step_observed = step;
                    }
                    if (benchmark_decode_steps > 0 && benchmark_ignore_stop) {
                        if (first_stop_observation) {
                            decode_progress.clear_line();
                            std::cerr << "Stop token observed at step " << step
                                      << " (continuing due to benchmark mode).\n";
                            decode_progress.render(step + 1);
                        }
                        continue;
                    }
                    decode_progress.clear_line();
                    std::cerr << "Stop token triggered at step " << step << ".\n";
                    break;
                }
            }
            decode_progress.finish(static_cast<int>(generated_steps.size() / static_cast<size_t>(patch_size * feat_dim)));
            if (benchmark_decode_steps > 0 && benchmark_ignore_stop && stop_step_observed >= 0) {
                std::cerr << "First stop token was observed at step " << stop_step_observed << ".\n";
            }

            generated_frames = static_cast<int>(generated_steps.size() / static_cast<size_t>(patch_size * feat_dim));
            if (options.retry_badcase &&
                generated_frames >= static_cast<int>(target_text_token_count * options.retry_badcase_ratio_threshold) &&
                attempt + 1 < max_attempts) {
                std::cerr << "Badcase detected, audio_text_ratio="
                          << (static_cast<float>(generated_frames) / static_cast<float>(target_text_token_count))
                          << ", retrying attempt " << (attempt + 2) << "/" << max_attempts << "...\n";
                continue;
            }

            const std::vector<float> decode_frames = build_decode_feature_sequence(prepared.prompt_feat,
                                                                                   prepared.prompt_audio_length,
                                                                                   generated_steps,
                                                                                   kStreamingPrefixLen,
                                                                                   patch_size,
                                                                                   feat_dim,
                                                                                   &prepended_context_frames);
            const int total_frames = static_cast<int>(decode_frames.size() / static_cast<size_t>(patch_size * feat_dim));
            const int total_patches = total_frames * patch_size;
            if (generated_frames == 0 || total_patches == 0) {
                fail("Model generated no audio patches");
            }

            const auto decode_start = std::chrono::steady_clock::now();
            const bool use_stateful_audio_decode =
                should_use_stateful_audio_decode(backend, audio_vae, total_patches);
            const int decode_stateful_chunk_frames = stateful_audio_decode_chunk_frames(audio_vae, patch_size);
            const bool use_output_pool_final_audio_decode =
                !disable_output_pool_final_decode &&
                should_use_output_pool_final_decode(state, prepared.has_reference_audio, seq_len) &&
                state.audio_frame_count >= prepared.prompt_audio_length + generated_frames;
            if (compare_output_pool_latent &&
                state.output_pool != nullptr &&
                state.output_pool->is_initialized() &&
                state.audio_frame_count >= prepared.prompt_audio_length + generated_frames) {
                const int frame_offset = std::max(0, prepared.prompt_audio_length - prepended_context_frames);
                const std::vector<float> output_pool_frames =
                    state.output_pool->export_latent_seq_range_to_host(backend, frame_offset, total_frames);
                const float latent_max_abs_diff = max_abs_diff(output_pool_frames, decode_frames);
                std::cerr << "[output_pool_compare]"
                          << " seq_len=" << seq_len
                          << " total_frames=" << total_frames
                          << " generated_frames=" << generated_frames
                          << " frame_offset=" << frame_offset
                          << " max_abs_diff=" << latent_max_abs_diff
                          << "\n";
            }
            if (use_stateful_audio_decode) {
                std::cerr << "Using stateful AudioVAE streaming decode for "
                          << total_patches << " latent patches"
                          << " with " << decode_stateful_chunk_frames << " frame chunks...\n";
                if (use_output_pool_final_audio_decode) {
                    const int frame_offset = std::max(0, prepared.prompt_audio_length - prepended_context_frames);
                    waveform = decode_audio_stateful_from_output_pool(audio_vae,
                                                                      backend,
                                                                      *state.output_pool,
                                                                      frame_offset,
                                                                      total_frames,
                                                                      prepended_context_frames,
                                                                      decode_stateful_chunk_frames,
                                                                      patch_size,
                                                                      feat_dim,
                                                                      decode_patch_len);
                } else {
                    waveform = decode_audio_stateful_from_patch_major_frames(audio_vae,
                                                                             backend,
                                                                             decode_frames,
                                                                             prepended_context_frames,
                                                                             decode_stateful_chunk_frames,
                                                                             patch_size,
                                                                             feat_dim,
                                                                              decode_patch_len);
                }
                if (waveform.empty()) {
                    fail("Stateful AudioVAE streaming decode failed");
                }
                prepended_context_frames = 0;
            }
            if (waveform.empty()) {
                std::vector<float> latent;
                if (use_output_pool_final_audio_decode) {
                    const int frame_offset = std::max(0, prepared.prompt_audio_length - prepended_context_frames);
                    latent = state.output_pool->export_audio_vae_latent_to_host(backend, frame_offset, total_frames);
                } else {
                    latent = patch_major_to_latent(decode_frames, patch_size, feat_dim);
                }
                waveform = decode_audio(audio_vae, backend, latent, total_patches, feat_dim);
            }
            const auto decode_end = std::chrono::steady_clock::now();
            vae_decode_time = std::chrono::duration<double>(decode_end - decode_start).count();
            final_state = std::move(state);
            break;
        }

        const auto model_end = std::chrono::steady_clock::now();
        const double model_time = std::chrono::duration<double>(model_end - model_start).count();
        log_memory_breakdown(log_memory, "post_waveform_decode", *store, backend, &final_state);
        if (prepared.has_prompt_audio) {
            const size_t trim = static_cast<size_t>(decode_patch_len) * static_cast<size_t>(prepended_context_frames);
            if (waveform.size() > trim) {
                waveform.erase(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(trim));
            }
        }
        if (waveform.empty()) {
            fail("Retry loop exhausted without producing an accepted sample");
        }

        write_wav_pcm16(options.output_path, waveform, audio_vae.config().output_sample_rate());

        const double audio_seconds =
            static_cast<double>(waveform.size()) / static_cast<double>(audio_vae.config().output_sample_rate());

        // Calculate RTF values
        const double total_synth_time = vae_encode_time + model_time + vae_decode_time;
        const double rtf_total = audio_seconds > 0.0 ? (total_synth_time / audio_seconds) : 0.0;
        const double rtf_model_only = audio_seconds > 0.0 ? (model_time / audio_seconds) : 0.0;
        const double rtf_without_encode = audio_seconds > 0.0
            ? ((model_time + vae_decode_time) / audio_seconds) : 0.0;

        std::cerr << "Saved audio to " << options.output_path
                  << " (" << std::fixed << std::setprecision(3) << audio_seconds << "s)\n";
        std::cerr << std::fixed << std::setprecision(3)
                  << "\n=== Timing Breakdown ===\n"
                  << "  AudioVAE encode:   " << vae_encode_time << "s\n"
                  << "  Model inference:   " << model_time << "s  (prefill + decode loop)\n"
                  << "  AudioVAE decode:   " << vae_decode_time << "s\n"
                  << "  -------------------------\n"
                  << "  Total:             " << total_synth_time << "s\n"
                  << "\n=== RTF (Real-Time Factor) ===\n"
                  << "  Without AudioVAE:        " << rtf_model_only << "\n"
                  << "  Without AudioVAE Encode: " << rtf_without_encode << "  (model + decode)\n"
                  << "  Full pipeline:           " << rtf_total << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
