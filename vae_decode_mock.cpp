/**
 * @file vae_decode_mock.cpp
 * @brief AudioVAE 独立解码（VAE Decode）重放工具（仅 CPU 后端）
 *
 *
 * 输入数据来自流式 mock 捕获（mock/，推理参考姊妹仓
 * voxcpm-vae 的 tools/decode_mock.py）：
 *   - manifest.json            元信息（sample_rate / patch_len_samples / calls）
 *   - captures/decoder_input_NNNN.npz   每步 decoder 输入 z=[1, 64, 6]
 *     （6 latent 帧 = 3 patch × 2 帧/patch）
 *   - reference_streaming.wav  逐 patch_len 取尾拼接的参考波形，用于对拍
 *
 * 重放逻辑与 decode_mock.py 一致：对每个捕获 z 独立做一次全序列 VAE decode
 * （非增量），每步取输出尾部 patch_len_samples 个采样拼进结果，最后写出
 * 16bit 单声道 WAV，并与参考 WAV 对拍 max|Δ|。
 *
 * 用法：
 *   build/examples/voxcpm-vae-decode-mock \
 *       [--model-path models/voxcpm-0.5b-audio-vae-q4_0.gguf] \
 *       [--mock-dir mock] \
 *       [--output assets/out/output_streaming.wav] \
 *       [--threads 4] \
 *       [--compare-wav assets/out/cpu_baseline.wav]
 *
 * 环境变量：
 *   VOXCPM_PROFILE_OPS=1  逐算子计时画像（经单 backend 临时 sched 的 eval
 *                         callback），全部步累计后输出 top-20 耗时算子
 */

#include "voxcpm/audio-vae.h"
#include "voxcpm/backend.h"
#include "voxcpm/context.h"
#include "voxcpm/weight-store.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxcpm {
namespace {

// 统一的报错出口：抛异常，由 main 外层打印并结束进程
[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

// 命令行参数集合（默认路径对齐 decode_mock.py）
struct Options {
  std::string model_path = "models/voxcpm-0.5b-audio-vae-q4_0.gguf";
  std::string mock_dir = "mock";
  std::string output_path = "assets/out/output_streaming.wav";
  std::string compare_wav;
  int threads = 4;
  int max_steps = 0;  // >0 时只重放前 N 步（调试/对拍用）
};

void print_usage(const char *argv0) {
  std::cerr << "Usage:\n"
            << "  " << argv0 << " [options]\n\n"
            << "Options:\n"
            << "  --model-path GGUF   (default: models/voxcpm-0.5b-audio-vae-q4_0.gguf)\n"
            << "  --mock-dir DIR      (default: mock)\n"
            << "  --output OUTPUT     (default: assets/out/output_streaming.wav)\n"
            << "  --threads INT       (default: 4)\n"
            << "  --max-steps INT     只重放前 N 步（0=全部，调试用）\n"
            << "  --compare-wav WAV   对拍指定 WAV（max|delta| + SNR），如 CPU 基线输出\n";
}

Options parse_args(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto require_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        fail(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--model-path") {
      options.model_path = require_value("--model-path");
    } else if (arg == "--mock-dir") {
      options.mock_dir = require_value("--mock-dir");
    } else if (arg == "--output" || arg == "-o") {
      options.output_path = require_value("--output");
    } else if (arg == "--threads") {
      options.threads = std::stoi(require_value("--threads"));
    } else if (arg == "--max-steps") {
      options.max_steps = std::stoi(require_value("--max-steps"));
    } else if (arg == "--compare-wav") {
      options.compare_wav = require_value("--compare-wav");
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      fail("Unknown argument: " + arg);
    }
  }

  if (options.threads < 1) {
    fail("--threads must be >= 1");
  }
  if (!std::filesystem::is_regular_file(options.model_path)) {
    fail("Model GGUF does not exist: " + options.model_path);
  }
  if (!std::filesystem::is_directory(options.mock_dir)) {
    fail("Mock dir does not exist: " + options.mock_dir);
  }
  return options;
}

// =============================================================================
// 极简 JSON 解析器（只用于读 manifest.json，无外部依赖）
// =============================================================================

struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;

  // 便捷取值：对象成员（不存在或类型不对时返回 nullptr）
  const JsonValue *find(const char *key) const {
    if (type != Type::Object) {
      return nullptr;
    }
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }
};

class MiniJsonParser {
public:
  explicit MiniJsonParser(const std::string &src) : src_(src) {}

  JsonValue parse() {
    skip_whitespace();
    JsonValue value = parse_value();
    skip_whitespace();
    if (pos_ != src_.size()) {
      fail("JSON: trailing content at offset " + std::to_string(pos_));
    }
    return value;
  }

private:
  const std::string &src_;
  size_t pos_ = 0;

  [[noreturn]] void error(const std::string &message) const {
    fail("JSON: " + message + " (offset " + std::to_string(pos_) + ")");
  }

  void skip_whitespace() {
    while (pos_ < src_.size()) {
      const char c = src_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  char peek() const {
    if (pos_ >= src_.size()) {
      error("unexpected end of input");
    }
    return src_[pos_];
  }

  void expect(char c) {
    if (peek() != c) {
      error(std::string("expected '") + c + "'");
    }
    ++pos_;
  }

  bool consume_literal(const char *literal) {
    const size_t len = std::strlen(literal);
    if (src_.compare(pos_, len, literal) == 0) {
      pos_ += len;
      return true;
    }
    return false;
  }

  JsonValue parse_value() {
    const char c = peek();
    if (c == '{') {
      return parse_object();
    }
    if (c == '[') {
      return parse_array();
    }
    if (c == '"') {
      JsonValue value;
      value.type = JsonValue::Type::String;
      value.string = parse_string();
      return value;
    }
    if (consume_literal("true")) {
      JsonValue value;
      value.type = JsonValue::Type::Bool;
      value.boolean = true;
      return value;
    }
    if (consume_literal("false")) {
      JsonValue value;
      value.type = JsonValue::Type::Bool;
      value.boolean = false;
      return value;
    }
    if (consume_literal("null")) {
      return JsonValue{};
    }
    return parse_number();
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue value;
    value.type = JsonValue::Type::Object;
    skip_whitespace();
    if (peek() == '}') {
      ++pos_;
      return value;
    }
    while (true) {
      skip_whitespace();
      std::string key = parse_string();
      skip_whitespace();
      expect(':');
      skip_whitespace();
      value.object.emplace(std::move(key), parse_value());
      skip_whitespace();
      const char c = peek();
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == '}') {
        ++pos_;
        return value;
      }
      error("expected ',' or '}' in object");
    }
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue value;
    value.type = JsonValue::Type::Array;
    skip_whitespace();
    if (peek() == ']') {
      ++pos_;
      return value;
    }
    while (true) {
      skip_whitespace();
      value.array.push_back(parse_value());
      skip_whitespace();
      const char c = peek();
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == ']') {
        ++pos_;
        return value;
      }
      error("expected ',' or ']' in array");
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= src_.size()) {
        error("unterminated string");
      }
      const char c = src_[pos_++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      // 转义序列
      if (pos_ >= src_.size()) {
        error("unterminated escape");
      }
      const char esc = src_[pos_++];
      switch (esc) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case 'u': {
        // \uXXXX（代理对合并后转 4 字节 UTF-8）
        const uint32_t cp = decode_unicode_escape(parse_hex4());
        append_utf8(out, cp);
        break;
      }
      default:
        error("invalid escape sequence");
      }
    }
  }

  uint32_t parse_hex4() {
    if (pos_ + 4 > src_.size()) {
      error("truncated \\u escape");
    }
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = src_[pos_++];
      value <<= 4;
      if (c >= '0' && c <= '9') {
        value |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        error("invalid hex digit in \\u escape");
      }
    }
    return value;
  }

  // 处理 UTF-16 代理对：高代理后跟 \uDC00-\uDFFF 时合并成完整码点；
  // 不是代理对则原样返回
  uint32_t decode_unicode_escape(uint32_t cp) {
    if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 6 <= src_.size() &&
        src_[pos_] == '\\' && src_[pos_ + 1] == 'u') {
      const size_t saved = pos_;
      pos_ += 2;
      const uint32_t low = parse_hex4();
      if (low >= 0xDC00 && low <= 0xDFFF) {
        return 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
      }
      pos_ = saved;  // 回退：下一个 \u 由后续调用方按普通转义处理
    }
    return cp;
  }

  static void append_utf8(std::string &out, uint32_t cp) {
    if (cp <= 0x7F) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  JsonValue parse_number() {
    const size_t start = pos_;
    if (pos_ < src_.size() && (src_[pos_] == '-' || src_[pos_] == '+')) {
      ++pos_;
    }
    while (pos_ < src_.size() &&
           (std::isdigit(static_cast<unsigned char>(src_[pos_])) != 0 ||
            src_[pos_] == '.' || src_[pos_] == 'e' || src_[pos_] == 'E' ||
            src_[pos_] == '-' || src_[pos_] == '+')) {
      ++pos_;
    }
    if (pos_ == start) {
      error("invalid value");
    }
    JsonValue value;
    value.type = JsonValue::Type::Number;
    value.number = std::stod(src_.substr(start, pos_ - start));
    return value;
  }
};

// =============================================================================
// manifest.json（mock 捕获清单）
// =============================================================================

struct MockCall {
  int index = 0;                // 捕获序号（决定拼接顺序）
  std::string file;             // npz 相对路径（相对 mock-dir）
  std::vector<int64_t> z_shape; // 捕获的 z 形状（用于校验，如 [1, 64, 6]）
};

struct MockManifest {
  int sample_rate = 0;         // 输出采样率（Hz）
  int patch_len_samples = 0;   // 每步拼接长度（一个 patch 的输出采样数）
  std::vector<MockCall> calls; // 全部捕获，按 index 升序
};

MockManifest load_manifest(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    fail("Failed to open manifest: " + path);
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  const JsonValue root = MiniJsonParser(oss.str()).parse();

  MockManifest manifest;
  if (const JsonValue *v = root.find("sample_rate")) {
    manifest.sample_rate = static_cast<int>(v->number);
  }
  if (const JsonValue *v = root.find("patch_len_samples")) {
    manifest.patch_len_samples = static_cast<int>(v->number);
  }
  if (const JsonValue *calls = root.find("calls")) {
    if (calls->type != JsonValue::Type::Array) {
      fail("Manifest 'calls' is not an array: " + path);
    }
    for (const JsonValue &call : calls->array) {
      MockCall entry;
      if (const JsonValue *v = call.find("index")) {
        entry.index = static_cast<int>(v->number);
      }
      if (const JsonValue *v = call.find("file")) {
        entry.file = v->string;
      }
      if (const JsonValue *v = call.find("z_shape")) {
        for (const JsonValue &dim : v->array) {
          entry.z_shape.push_back(static_cast<int64_t>(dim.number));
        }
      }
      if (entry.file.empty()) {
        fail("Manifest call without file: " + path);
      }
      manifest.calls.push_back(std::move(entry));
    }
  }
  if (manifest.calls.empty()) {
    fail("Manifest has no calls: " + path);
  }
  std::stable_sort(manifest.calls.begin(), manifest.calls.end(),
                   [](const MockCall &a, const MockCall &b) {
                     return a.index < b.index;
                   });
  if (manifest.sample_rate <= 0 || manifest.patch_len_samples <= 0) {
    fail("Manifest missing sample_rate / patch_len_samples: " + path);
  }
  return manifest;
}

// =============================================================================
// npz（zip 容器）+ npy 读取器
// 只支持 numpy savez 写出的 STORED（不压缩）float32 数组
// =============================================================================

struct NpyArray {
  std::vector<int64_t> shape; // 逻辑形状（如 {1, 64, 6}）
  bool fortran_order = false; // true = 列主序存储
  std::vector<float> data;    // 按文件存储顺序的原始数据
};

uint16_t read_le_u16(const uint8_t *p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                               (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_le_u32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

struct ZipEntry {
  std::string name;
  uint16_t method = 0;
  uint32_t compressed_size = 0;
  uint32_t uncompressed_size = 0;
  uint32_t local_header_offset = 0;
};

// 解析 zip central directory（npz 是标准 zip 容器）
std::vector<ZipEntry> read_zip_entries(const std::vector<uint8_t> &file,
                                       const std::string &path) {
  constexpr uint32_t kEocdSig = 0x06054b50;  // "PK\x05\x06"
  constexpr uint32_t kCdirSig = 0x02014b50;  // "PK\x01\x02"

  if (file.size() < 22) {
    fail("npz too small to be a zip: " + path);
  }
  // 从尾部向前扫 EOCD（zip 注释最长 65535 字节）
  size_t eocd = SIZE_MAX;
  const size_t lowest =
      file.size() > (22 + 65535) ? file.size() - 22 - 65535 : 0;
  for (size_t i = file.size() - 22 + 1; i-- > lowest;) {
    if (read_le_u32(file.data() + i) == kEocdSig) {
      eocd = i;
      break;
    }
  }
  if (eocd == SIZE_MAX) {
    fail("npz has no zip end-of-central-directory: " + path);
  }

  const uint16_t entry_count = read_le_u16(file.data() + eocd + 10);
  const uint32_t cd_offset = read_le_u32(file.data() + eocd + 16);
  if (static_cast<size_t>(cd_offset) + 46 > file.size()) {
    fail("npz central directory out of range: " + path);
  }

  std::vector<ZipEntry> entries;
  entries.reserve(entry_count);
  size_t pos = cd_offset;
  for (uint16_t i = 0; i < entry_count; ++i) {
    if (pos + 46 > file.size() || read_le_u32(file.data() + pos) != kCdirSig) {
      fail("npz central directory corrupted: " + path);
    }
    ZipEntry entry;
    entry.method = read_le_u16(file.data() + pos + 10);
    entry.compressed_size = read_le_u32(file.data() + pos + 20);
    entry.uncompressed_size = read_le_u32(file.data() + pos + 24);
    const uint16_t name_len = read_le_u16(file.data() + pos + 28);
    const uint16_t extra_len = read_le_u16(file.data() + pos + 30);
    const uint16_t comment_len = read_le_u16(file.data() + pos + 32);
    entry.local_header_offset = read_le_u32(file.data() + pos + 42);
    pos += 46;
    if (pos + name_len > file.size()) {
      fail("npz entry name out of range: " + path);
    }
    entry.name.assign(reinterpret_cast<const char *>(file.data() + pos),
                      name_len);
    pos += static_cast<size_t>(name_len) + extra_len + comment_len;
    entries.push_back(std::move(entry));
  }
  return entries;
}

// 取出 zip 中指定成员的原始字节（npz 的 savez 用 STORED，无需解压）
std::vector<uint8_t> extract_zip_member(const std::vector<uint8_t> &file,
                                        const ZipEntry &entry,
                                        const std::string &path) {
  if (entry.method != 0) {
    fail("npz member is compressed (method " +
         std::to_string(entry.method) + "): " + entry.name + " in " + path);
  }
  if (entry.local_header_offset + 30 > file.size() ||
      read_le_u32(file.data() + entry.local_header_offset) != 0x04034b50) {
    fail("npz local header corrupted: " + entry.name + " in " + path);
  }
  const uint8_t *local = file.data() + entry.local_header_offset;
  const uint16_t name_len = read_le_u16(local + 26);
  const uint16_t extra_len = read_le_u16(local + 28);
  const size_t data_offset =
      entry.local_header_offset + 30 + name_len + extra_len;
  if (data_offset + entry.uncompressed_size > file.size()) {
    fail("npz member data out of range: " + entry.name + " in " + path);
  }
  return std::vector<uint8_t>(
      file.begin() + static_cast<std::ptrdiff_t>(data_offset),
      file.begin() +
          static_cast<std::ptrdiff_t>(data_offset + entry.uncompressed_size));
}

// 解析 npy 头部（python dict 字面量）中的 descr / fortran_order / shape
void parse_npy_header(const std::string &header, std::string &descr,
                      bool &fortran_order, std::vector<int64_t> &shape) {
  const auto find_quoted = [&](const char *key) -> std::string {
    const size_t key_pos = header.find(key);
    if (key_pos == std::string::npos) {
      fail("npy header missing " + std::string(key));
    }
    const size_t q1 = header.find('\'', key_pos + std::strlen(key));
    const size_t q2 =
        q1 == std::string::npos ? std::string::npos : header.find('\'', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) {
      fail("npy header malformed near " + std::string(key));
    }
    return header.substr(q1 + 1, q2 - q1 - 1);
  };

  descr = find_quoted("'descr'");

  fortran_order = false;
  const size_t order_pos = header.find("'fortran_order':");
  if (order_pos != std::string::npos) {
    if (header.find("True", order_pos) != std::string::npos &&
        header.find("True", order_pos) < header.find(')', order_pos)) {
      fortran_order = true;
    }
  }

  const size_t shape_key_pos = header.find("'shape':");
  if (shape_key_pos == std::string::npos) {
    fail("npy header missing 'shape'");
  }
  const size_t open = header.find('(', shape_key_pos);
  const size_t close = header.find(')', shape_key_pos);
  if (open == std::string::npos || close == std::string::npos || close < open) {
    fail("npy header malformed 'shape'");
  }
  std::istringstream iss(header.substr(open + 1, close - open - 1));
  std::string token;
  while (std::getline(iss, token, ',')) {
    // 去空白；"(64,)" 这类单元素尾逗号会留一个空 token，跳过
    size_t b = 0;
    size_t e = token.size();
    while (b < e && std::isspace(static_cast<unsigned char>(token[b])) != 0) {
      ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(token[e - 1])) != 0) {
      --e;
    }
    if (b == e) {
      continue;
    }
    shape.push_back(std::stoll(token.substr(b, e - b)));
  }
  if (shape.empty()) {
    fail("npy header has empty shape");
  }
}

// 逻辑形状转字符串（日志/报错用）
std::string shape_to_string(const std::vector<int64_t> &shape) {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << ")";
  return oss.str();
}

// 读取 npz 中的指定 .npy 成员（要求 '<f4' float32）
NpyArray read_npz_array(const std::string &path, const std::string &member) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    fail("Failed to open npz: " + path);
  }
  std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());

  const std::vector<ZipEntry> entries = read_zip_entries(file, path);
  const auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const ZipEntry &entry) {
                                 return entry.name == member;
                               });
  if (it == entries.end()) {
    fail("npz member not found: " + member + " in " + path);
  }
  const std::vector<uint8_t> raw = extract_zip_member(file, *it, path);

  // npy 格式：\x93NUMPY + 主/次版本 + 头长 + 头部 dict + 数据
  if (raw.size() < 10 || std::memcmp(raw.data(), "\x93NUMPY", 6) != 0) {
    fail("npz member is not npy: " + member + " in " + path);
  }
  const uint8_t major = raw[6];
  size_t header_len = 0;
  size_t data_offset = 0;
  if (major == 1) {
    header_len = read_le_u16(raw.data() + 8);
    data_offset = 10;
  } else if (major == 2) {
    if (raw.size() < 12) {
      fail("npy truncated: " + member + " in " + path);
    }
    header_len = read_le_u32(raw.data() + 8);
    data_offset = 12;
  } else {
    fail("unsupported npy version " + std::to_string(major) + ": " + member +
         " in " + path);
  }
  if (data_offset + header_len > raw.size()) {
    fail("npy header out of range: " + member + " in " + path);
  }
  const std::string header(
      reinterpret_cast<const char *>(raw.data()) + data_offset, header_len);

  NpyArray array;
  std::string descr;
  parse_npy_header(header, descr, array.fortran_order, array.shape);
  if (descr != "<f4" && descr != "float32") {
    fail("npy dtype is not float32 ('" + descr + "'): " + member + " in " +
         path);
  }

  size_t total = 1;
  for (int64_t dim : array.shape) {
    total *= static_cast<size_t>(dim);
  }
  if (data_offset + header_len + total * sizeof(float) > raw.size()) {
    fail("npy data out of range: " + member + " in " + path);
  }
  array.data.resize(total);
  std::memcpy(array.data.data(), raw.data() + data_offset + header_len,
              total * sizeof(float));
  return array;
}

// 把 mock 捕获的 z（PyTorch [1, C, T] 通道优先）重排成 AudioVAE::decode
// 期望的 ggml 布局：张量 ne=[T, C]，contiguous 内存下标 data[c*T + t]
// （时间维最快）。同时兼容 C / Fortran 两种存储序。
std::vector<float> npy_channels_to_latent(const NpyArray &array,
                                          int expected_channels) {
  if (array.shape.size() != 3 || array.shape[0] != 1) {
    fail("Unexpected latent shape (expected [1, C, T]): " +
         shape_to_string(array.shape));
  }
  const int64_t channels = array.shape[1];
  const int64_t time = array.shape[2];
  if (channels != expected_channels) {
    fail("Latent channel mismatch: npz has " + std::to_string(channels) +
         ", model expects " + std::to_string(expected_channels));
  }

  std::vector<float> latent(static_cast<size_t>(channels) *
                            static_cast<size_t>(time));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < time; ++t) {
      const size_t src =
          array.fortran_order
              // F order strides: [1, 1, C]（第 0 维长度为 1）
              ? static_cast<size_t>(c) + static_cast<size_t>(t) *
                                             static_cast<size_t>(channels)
              : static_cast<size_t>(c) * static_cast<size_t>(time) +
                    static_cast<size_t>(t);
      latent[static_cast<size_t>(c) * static_cast<size_t>(time) +
             static_cast<size_t>(t)] = array.data[src];
    }
  }
  return latent;
}

// =============================================================================
// WAV 读写（与 tts_offline.cpp 保持一致的最小实现）
// =============================================================================

// 从 WAV 文件读出的原始数据（多声道采样点交织存放）
struct WavData {
  int sample_rate = 0;
  int channels = 0;
  std::vector<float> samples; // 已归一化到 [-1, 1]
};

uint16_t read_stream_le_u16(std::istream &in) {
  uint8_t bytes[2] = {0, 0};
  in.read(reinterpret_cast<char *>(bytes), 2);
  return static_cast<uint16_t>(static_cast<uint16_t>(bytes[0]) |
                               (static_cast<uint16_t>(bytes[1]) << 8));
}

uint32_t read_stream_le_u32(std::istream &in) {
  uint8_t bytes[4] = {0, 0, 0, 0};
  in.read(reinterpret_cast<char *>(bytes), 4);
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

// 轻量 WAV 解析器：支持 PCM 16/24/32bit 与 32bit float，输出 [-1, 1] float
WavData read_wav_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    fail("Failed to open WAV file: " + path);
  }

  // 文件头： "RIFF" + 总大小 + "WAVE"
  char riff[4] = {0};
  char wave[4] = {0};
  in.read(riff, 4);
  (void)read_stream_le_u32(in); // riff_size，不做严格校验
  in.read(wave, 4);
  if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
    fail("Invalid WAV header: " + path);
  }

  uint16_t audio_format = 0; // 1 = PCM, 3 = IEEE float
  uint16_t num_channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  std::vector<uint8_t> data_chunk;

  // 逐 chunk 扫描，直到拿齐 fmt 信息和 data 数据为止
  while (in && (!sample_rate || data_chunk.empty())) {
    char chunk_id[4] = {0};
    in.read(chunk_id, 4);
    if (in.gcount() != 4) {
      break;
    }
    const uint32_t chunk_size = read_stream_le_u32(in);
    const std::string id(chunk_id, 4);

    if (id == "fmt ") {
      audio_format = read_stream_le_u16(in);
      num_channels = read_stream_le_u16(in);
      sample_rate = read_stream_le_u32(in);
      (void)read_stream_le_u32(in); // byte_rate
      (void)read_stream_le_u16(in); // block_align
      bits_per_sample = read_stream_le_u16(in);
      if (chunk_size > 16) {
        in.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
      }
    } else if (id == "data") {
      data_chunk.resize(chunk_size);
      in.read(reinterpret_cast<char *>(data_chunk.data()),
              static_cast<std::streamsize>(chunk_size));
    } else {
      in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
    }
    if (chunk_size % 2 != 0) {
      in.seekg(1, std::ios::cur); // RIFF chunk 按 2 字节对齐
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

  const size_t frame_count =
      data_chunk.size() / (bytes_per_sample * num_channels);
  std::vector<float> samples(frame_count * num_channels, 0.0f);

  // 逐采样点解码并归一化到 [-1, 1]
  size_t offset = 0;
  for (size_t i = 0; i < frame_count * num_channels; ++i) {
    if (audio_format == 3 && bits_per_sample == 32) {
      float value = 0.0f;
      std::memcpy(&value, data_chunk.data() + offset, sizeof(float));
      samples[i] = value;
    } else if (audio_format == 1 && bits_per_sample == 16) {
      const int16_t value =
          static_cast<int16_t>(data_chunk[offset] | (data_chunk[offset + 1] << 8));
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
      fail("Unsupported WAV bit depth in WAV file: " + path);
    }
    offset += bytes_per_sample;
  }

  return WavData{static_cast<int>(sample_rate), static_cast<int>(num_channels),
                 std::move(samples)};
}

// 多声道 -> 单声道：逐帧对所有声道取平均
std::vector<float> convert_to_mono(const WavData &wav) {
  if (wav.channels == 1) {
    return wav.samples;
  }
  const size_t frame_count =
      wav.samples.size() / static_cast<size_t>(wav.channels);
  std::vector<float> mono(frame_count, 0.0f);
  for (size_t frame = 0; frame < frame_count; ++frame) {
    float sum = 0.0f;
    for (int channel = 0; channel < wav.channels; ++channel) {
      sum += wav.samples[frame * static_cast<size_t>(wav.channels) +
                         static_cast<size_t>(channel)];
    }
    mono[frame] = sum / static_cast<float>(wav.channels);
  }
  return mono;
}

// 把 float 波形写成标准 16bit PCM 单声道 WAV（44 字节头 + 数据）
void write_wav_pcm16(const std::string &path, const std::vector<float> &audio,
                     int sample_rate) {
  const std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    fail("Failed to open output WAV file: " + path);
  }

  // ---- 组装并写入 RIFF/WAVE 头 ----
  const uint16_t channels = 1;
  const uint16_t bits_per_sample = 16;
  const uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
  const uint16_t block_align = channels * (bits_per_sample / 8);
  const uint32_t data_size =
      static_cast<uint32_t>(audio.size() * sizeof(int16_t));
  const uint32_t riff_size = 36 + data_size;

  out.write("RIFF", 4);
  out.write(reinterpret_cast<const char *>(&riff_size), 4);
  out.write("WAVE", 4);

  const uint32_t fmt_size = 16;
  const uint16_t audio_format = 1;
  out.write("fmt ", 4);
  out.write(reinterpret_cast<const char *>(&fmt_size), 4);
  out.write(reinterpret_cast<const char *>(&audio_format), 2);
  out.write(reinterpret_cast<const char *>(&channels), 2);
  out.write(reinterpret_cast<const char *>(&sample_rate), 4);
  out.write(reinterpret_cast<const char *>(&byte_rate), 4);
  out.write(reinterpret_cast<const char *>(&block_align), 2);
  out.write(reinterpret_cast<const char *>(&bits_per_sample), 2);

  // data chunk：float clamp 到 [-1,1] 后量化成 int16
  out.write("data", 4);
  out.write(reinterpret_cast<const char *>(&data_size), 4);
  for (float sample : audio) {
    const float clamped = std::max(-1.0f, std::min(1.0f, sample));
    const int16_t pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
    out.write(reinterpret_cast<const char *>(&pcm), sizeof(int16_t));
  }
}

// =============================================================================
// AudioVAE 解码（自 tts_offline.cpp 的 decode_audio() 抽取，布局约定一致）
// =============================================================================

// 用 AudioVAE 解码器把 latent 特征还原成 PCM 波形。
// latent_host 的布局：ggml 张量 ne=[total_patches, feat_dim] 的 contiguous
// 数据，即 latent_host[c * total_patches + t]（时间维最快）。

// =============================================================================
// 逐算子计时画像（VOXCPM_PROFILE_OPS=1 时启用）
//
// 生产路径不变；画像走一条独立的单 CPU backend 临时 sched，经 eval callback
// 在每个节点执行前打点。执行 kernel 与生产路径完全一致（同为 CPU backend），
// 仅调度壳不同，数值零差异。
// =============================================================================

struct OpProfiler {
  // key = op 名 + 输出形状 + src0 op 名；value = 累计微秒 / 次数
  std::map<std::string, double> us_by_key;
  std::map<std::string, int64_t> count_by_key;
  const ggml_tensor *prev = nullptr;
  std::chrono::steady_clock::time_point last{};
  int64_t total_us = 0;
  bool dump_nodes = false;   // VOXCPM_DUMP_OPS=1：每节点打印 name/ne/前 3 值
  int dump_seq = 0;

  static void dump_node(const ggml_tensor *t, int seq) {
    std::ostringstream oss;
    oss << "[node " << seq << "] " << ggml_op_desc(t) << " '" << (t->name[0] ? t->name : "-")
        << "' ne=[";
    for (int i = 0; i < 4; ++i) {
      if (i > 0) oss << ",";
      oss << t->ne[i];
    }
    oss << "]";
    // 前 3 个元素的校验值（需 host 可读：CPU 路径下 t->data 直接可解引用）
    if (t->type == GGML_TYPE_F32 && t->buffer != nullptr &&
        ggml_backend_buffer_is_host(t->buffer) && t->data != nullptr) {
      const float *v = static_cast<const float *>(t->data);
      oss << " v[0..2] =";
      for (int i = 0; i < 3 && i < ggml_nelements(t); ++i) {
        oss << " " << v[i];
      }
      // 语义锚点：v[6]（L1 布局 = (t=1,c=0) / L2 布局 = (c=6,·)）、v[1536]
      for (int idx : {6, 1536, 1537}) {
        if (idx < ggml_nelements(t)) {
          oss << " v[" << idx << "]=" << v[idx];
        }
      }
      double sum = 0.0;
      const int64_t n = ggml_nelements(t);
      for (int64_t i = 0; i < n; ++i) {
        sum += v[i];
      }
      oss << " sum=" << sum;
    }
    for (int s_i = 0; s_i < 2; ++s_i) {
      const ggml_tensor *s = t->src[s_i];
      if (s != nullptr && s->type == GGML_TYPE_F32 && s->data != nullptr &&
          ggml_nelements(s) > 0) {
        const float *sv = static_cast<const float *>(s->data);
        oss << " src" << s_i << "[0..2] =";
        for (int i = 0; i < 3 && i < ggml_nelements(s); ++i) {
          oss << " " << sv[i];
        }
      }
    }
    std::cerr << oss.str() << "\n";
  }

  static std::string key_of(const ggml_tensor *t) {
    std::ostringstream oss;
    oss << ggml_op_desc(t) << " ne=[";
    for (int i = 0; i < 4; ++i) {
      if (i > 0) oss << ",";
      oss << t->ne[i];
    }
    oss << "]";
    if (t->src[0] != nullptr) {
      oss << " src0=" << ggml_op_name(t->src[0]->op);
    }
    return oss.str();
  }

  static bool callback(ggml_tensor *t, bool ask, void *user_data) {
    auto *self = static_cast<OpProfiler *>(user_data);
    if (ask) {
      return true;  // 观测全部节点
    }
    const auto now = std::chrono::steady_clock::now();
    // sched 在节点计算完成后才回调 ask=false：(now - last) 覆盖的是 t 自身的
    // 计算 + synchronize，因此归属给当前节点 t（此前记给 prev 导致整体错位一位）
    if (self->prev != nullptr) {
      const double us = std::chrono::duration<double, std::micro>(now - self->last).count();
      const std::string key = key_of(t);
      self->us_by_key[key] += us;
      self->count_by_key[key] += 1;
      self->total_us += static_cast<int64_t>(us);
    }
    self->prev = t;
    self->last = now;
    if (self->dump_nodes && self->dump_seq < 400) {
      dump_node(t, self->dump_seq++);
    }
    return true;
  }

  void flush_tail() {
    // 最后一个节点的执行时长在 compute 结束后补记
    if (prev != nullptr) {
      const double us =
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - last)
              .count();
      const std::string key = key_of(prev);
      us_by_key[key] += us;
      count_by_key[key] += 1;
      total_us += static_cast<int64_t>(us);
      prev = nullptr;
    }
  }

  void dump_top(int top_n, int n_calls) const {
    std::vector<std::pair<std::string, double>> items(us_by_key.begin(), us_by_key.end());
    std::sort(items.begin(), items.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    std::cerr << "\n[profile] top " << std::min<size_t>(top_n, items.size())
              << " ops by total time (" << n_calls << " calls, total "
              << std::fixed << std::setprecision(1)
              << static_cast<double>(total_us) / 1000.0 << " ms)\n";
    std::cerr << std::string(96, '-') << "\n";
    double cumulative = 0.0;
    for (size_t i = 0; i < items.size() && i < static_cast<size_t>(top_n); ++i) {
      cumulative += items[i].second;
      const int64_t n = count_by_key.at(items[i].first);
      std::cerr << std::fixed << std::setprecision(1) << std::setw(10)
                << items[i].second / 1000.0 << " ms | " << std::setw(7) << n
                << "x | " << std::setprecision(2) << std::setw(5)
                << (total_us > 0 ? 100.0 * items[i].second / static_cast<double>(total_us) : 0.0)
                << "% | cum " << std::setprecision(1) << std::setw(5)
                << (total_us > 0 ? 100.0 * cumulative / static_cast<double>(total_us) : 0.0)
                << "% | " << items[i].first << "\n";
    }
  }
};

// 画像用的单 CPU backend sched：首次调用时创建并 reserve，之后各步
// reset + alloc（CPU-only，无跨后端拷贝；仅供打点，生产路径不用它）。
// 进程退出时随进程回收（工具进程，刻意不管理生命周期）。
struct ProfileSched {
  ggml_backend_sched_t sched = nullptr;
  OpProfiler profiler;
};

std::vector<float> decode_waveform(AudioVAE &audio_vae, VoxCPMBackend &backend,
                                   const std::vector<float> &latent_host,
                                   int total_patches, int feat_dim,
                                   ProfileSched *profile, AudioVAEL2Plan *l2plan) {
  // 派生权重（convT 拆半 / F32 化 / inv）幂等构建：HTP 模式落在 HTP buffer
  if (!audio_vae.ensure_derived_weights(backend)) {
    fail("Failed to build derived decoder weights");
  }

  // Stage 2d：HTP 模式走拆段计划（9 张单后端图 + 段间显式搬运，图复用）
  if (l2plan != nullptr) {
    if (!l2plan->build(audio_vae, backend, total_patches)) {
      fail("Failed to build L2 decode plan");
    }
    ggml_tensor *lin = l2plan->latent_input();
    if (!lin) {
      fail("L2 plan has no latent input");
    }
    backend.tensor_set(lin, latent_host.data(), 0,
                       latent_host.size() * sizeof(float));
    ggml_status st = l2plan->run(backend);
    if (st != GGML_STATUS_SUCCESS) {
      fail("L2 plan decode failed");
    }
    ggml_tensor *audio = l2plan->audio_output();
    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
    backend.tensor_get(audio, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
  }

  // 新建临时计算图：输入 latent [total_patches, feat_dim]，输出波形
  VoxCPMContext graph_ctx(ContextType::Graph, 32768, 262144);
  ggml_tensor *latent =
      graph_ctx.new_tensor_2d(GGML_TYPE_F32, total_patches, feat_dim);
  ggml_set_input(latent);
  ggml_tensor *audio = audio_vae.decode(graph_ctx, latent);
  if (!audio) {
    fail("Failed to build AudioVAE decode graph");
  }

  // 搭建并执行解码计算图：reserve(预估内存) -> alloc(分配) -> 填输入 -> compute
  ggml_cgraph *graph = graph_ctx.new_graph();
  graph_ctx.build_forward(graph, audio);
  if (profile != nullptr) {
    // 画像路径：单 CPU backend 的临时 sched + eval callback 打点
    if (profile->sched == nullptr) {
      ggml_backend_t backends[1] = {backend.raw_backend()};
      // graph_size 容量只需 >= n_nodes + n_leafs；这里给充足余量
      profile->sched = ggml_backend_sched_new(
          backends, nullptr, 1, 8192,
          /*parallel=*/false, /*op_offload=*/false);
      if (profile->sched == nullptr) {
        fail("Failed to create profiling sched");
      }
      ggml_backend_sched_set_eval_callback(profile->sched, OpProfiler::callback,
                                           &profile->profiler);
      if (!ggml_backend_sched_reserve(profile->sched, graph)) {
        fail("Failed to reserve profiling sched buffers");
      }
    }
    ggml_backend_sched_reset(profile->sched);
    if (!ggml_backend_sched_alloc_graph(profile->sched, graph)) {
      fail("Failed to alloc profiling sched graph");
    }
  } else {
    backend.reserve_compute_memory(graph, "vae_mock.audio_vae.decode");
    backend.alloc_graph(graph, "vae_mock.audio_vae.decode");
  }
  backend.tensor_set(latent, latent_host.data(), 0,
                     latent_host.size() * sizeof(float));
  // 准备解码器所需的其他输入（如 sample-rate conditioning），再执行整个图
  audio_vae.prepare_decode_inputs(backend);
  ggml_status status =
      profile != nullptr ? ggml_backend_sched_graph_compute(profile->sched, graph)
                         : backend.compute(graph);
  if (profile != nullptr) {
    profile->profiler.flush_tail();
  }
  if (status != GGML_STATUS_SUCCESS) {
    fail("AudioVAE decode failed");
  }

  // 解码出的波形拷回 host：形状 [samples, 1, 1]
  std::vector<float> waveform(static_cast<size_t>(ggml_nelements(audio)));
  backend.tensor_get(audio, waveform.data(), 0,
                     waveform.size() * sizeof(float));
  return waveform;
}

} // namespace
} // namespace voxcpm

int main(int argc, char **argv) {
  using namespace voxcpm;

  try {
    const Options options = parse_args(argc, argv);

    // ============ 1. 初始化后端并加载独立 VAE GGUF ============
    // VOXCPM_BACKEND=htp 启用 CPU+HTP 异构；未设置/=cpu 走纯 CPU（默认回退）
    BackendType backend_type = BackendType::CPU;
    if (const char* be = std::getenv("VOXCPM_BACKEND")) {
        if (std::strcmp(be, "htp") == 0) {
            backend_type = BackendType::HTP;
        }
    }
    VoxCPMBackend backend(backend_type, options.threads);
    std::cerr << "Using backend: " << backend.backend_name();
    if (std::strlen(backend.backend_description()) > 0) {
      std::cerr << " (" << backend.backend_description() << ")";
    }
    std::cerr << "\n";
    std::cerr << "Loading GGUF from " << options.model_path << " with "
              << options.threads << " threads...\n";
    auto store = std::make_shared<VoxCPMWeightStore>();
    // 权重分桶（Stage 2b 保守版）。GGUF 实测：dw 与 final conv 是 F16 [7,1,C]，
    // pw 是 Q4_0 [Cin,Cout]，convT 是 Q4_0 [K*Cout,Cin]（k 主序折叠，2d 已展开）。
    // 仅 pw 的 Q4_0 上 HTP——其唯一消费者是 MUL_MAT，可整体留在 HTP 侧。
    // convT 当前仍被 CPU 的 unfold/conv_transpose_1d 消费，CPU 算子直接解引用
    // HTP buffer 会读到垃圾（Stage 1 教训），2c lowering 后再迁移。
    // VOXCPM_HTP_WEIGHTS=1：pw Q4_0 权重上 HTP。默认关——当前 im2col 混合图上
    // HTP 子图数据流仍有问题（SNR<0），2c 图重写（HTP 内聚形态）后再默认开。
    // 注意：WeightGroupFn 是函数指针，谓词必须无捕获 lambda。
    const auto htp_group = [](const ggml_tensor *t) -> bool {
      if (const char *e = std::getenv("VOXCPM_HTP_WEIGHTS"); e == nullptr || e[0] != '1') {
        return false;
      }
      if (t->type != GGML_TYPE_Q4_0) {
        return false;
      }
      if (std::strncmp(t->name, "audio_vae.decoder.", 18) != 0) {
        return false;
      }
      // convT 名形如 model.N.block.1.weight（Q4_0 且以此结尾的只有 convT；
      // res unit 的 block.N.block.1.weight 是 F16 dw，已被类型排除）
      const char *suf = std::strstr(t->name, ".block.1.weight");
      const bool is_convt = suf != nullptr && std::strcmp(suf, ".block.1.weight") == 0;
      return !is_convt;
    };
    if (!store->load_from_file(options.model_path, backend, htp_group)) {
      fail("Failed to load GGUF: " + options.model_path);
    }
    std::cerr << "Weights buffer: " << std::fixed << std::setprecision(1)
              << static_cast<double>(store->buffer_size()) / (1024.0 * 1024.0)
              << " MiB CPU + " << std::setprecision(1)
              << static_cast<double>(store->htp_buffer_size()) / (1024.0 * 1024.0)
              << " MiB HTP (weights kept in native GGUF dtypes)\n";

    AudioVAE audio_vae;
    if (!audio_vae.load_from_store(store)) {
      fail("Failed to initialize AudioVAE from GGUF");
    }

    const AudioVAEConfig &config = audio_vae.config();
    const int latent_dim = config.latent_dim;
    const int decode_hop = config.decode_hop_length();
    const int output_sample_rate = config.output_sample_rate();
    std::cerr << "AudioVAE config: latent_dim=" << latent_dim
              << " decode_hop=" << decode_hop
              << " output_sample_rate=" << output_sample_rate << "\n";

    // ============ 2. 读取 mock 捕获清单 ============
    const std::string manifest_path =
        (std::filesystem::path(options.mock_dir) / "manifest.json").string();
    const MockManifest manifest = load_manifest(manifest_path);
    std::cerr << "Mock dir: " << options.mock_dir << " ("
              << manifest.calls.size() << " calls, patch="
              << manifest.patch_len_samples << " samples @ "
              << manifest.sample_rate << " Hz)\n";
    if (manifest.sample_rate != output_sample_rate) {
      fail("Manifest sample rate " + std::to_string(manifest.sample_rate) +
           " != model output sample rate " +
           std::to_string(output_sample_rate));
    }

    // ============ 3. 逐捕获重放 VAE decode，每步取尾 patch_len 拼接 ============
    ProfileSched profile_sched;
    ProfileSched *profile = nullptr;
    if (const char *env = std::getenv("VOXCPM_PROFILE_OPS");
        env != nullptr && env[0] == '1') {
      profile = &profile_sched;
      std::cerr << "Op profiling enabled (VOXCPM_PROFILE_OPS=1)\n";
      if (const char *e = std::getenv("VOXCPM_DUMP_OPS"); e != nullptr && e[0] == '1') {
        profile_sched.profiler.dump_nodes = true;
        std::cerr << "Node dump enabled (VOXCPM_DUMP_OPS=1)\n";
      }
    }

    std::vector<float> full_waveform;
    full_waveform.reserve(static_cast<size_t>(manifest.calls.size()) *
                          static_cast<size_t>(manifest.patch_len_samples));

    // Stage 2d：HTP 模式启用拆段解码计划（profile 路径仍走单图打点）
    AudioVAEL2Plan *l2plan_ptr =
        backend.is_htp_active() && profile == nullptr ? new AudioVAEL2Plan : nullptr;
    if (l2plan_ptr != nullptr) {
      std::cerr << "L2 decode plan enabled (HTP segmented graphs)\n";
    }

    const auto decode_start = std::chrono::steady_clock::now();
    size_t n_calls = manifest.calls.size();
    if (options.max_steps > 0) {
      n_calls = std::min(n_calls, static_cast<size_t>(options.max_steps));
      std::cerr << "Max steps: " << n_calls << " (of " << manifest.calls.size()
                << ")\n";
    }
    for (size_t i = 0; i < n_calls; ++i) {
      const MockCall &call = manifest.calls[i];
      const std::string npz_path =
          (std::filesystem::path(options.mock_dir) / call.file).string();

      const NpyArray z = read_npz_array(npz_path, "z.npy");
      // 校验捕获形状与 manifest 记录一致
      if (!call.z_shape.empty() && call.z_shape != z.shape) {
        fail("Capture shape mismatch for " + npz_path + ": npz " +
             shape_to_string(z.shape) + " vs manifest " +
             shape_to_string(call.z_shape));
      }

      const std::vector<float> latent = npy_channels_to_latent(z, latent_dim);
      const int total_patches = static_cast<int>(z.shape[2]);
      std::vector<float> chunk = decode_waveform(audio_vae, backend, latent,
                                                 total_patches, latent_dim,
                                                 profile, l2plan_ptr);

      // 每步取输出尾部一个 patch（流式语义：decoder 是因果卷积，尾部
      // patch_len_samples 个采样才是本步新增的音频）
      const size_t tail =
          std::min(chunk.size(), static_cast<size_t>(manifest.patch_len_samples));
      full_waveform.insert(
          full_waveform.end(),
          chunk.end() - static_cast<std::ptrdiff_t>(tail), chunk.end());

      if ((i + 1) % 16 == 0 || i + 1 == n_calls) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          decode_start)
                .count();
        std::cerr << "\r[mock] decoded " << (i + 1) << "/"
                  << n_calls << " | elapsed " << std::fixed
                  << std::setprecision(1) << elapsed << "s" << std::flush;
      }
    }
    std::cerr << "\n";

    const double decode_time = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - decode_start)
                                   .count();
    const double audio_seconds =
        static_cast<double>(full_waveform.size()) /
        static_cast<double>(output_sample_rate);
    std::cerr << std::fixed << std::setprecision(2) << "Decoded "
              << full_waveform.size() << " samples (" << audio_seconds
              << "s) in " << decode_time << "s ("
              << std::setprecision(3)
              << decode_time / static_cast<double>(
                                   std::max<size_t>(1, n_calls))
              << "s/call, RTF " << std::setprecision(2)
              << decode_time / std::max(audio_seconds, 1e-9) << ")\n";

    if (profile != nullptr) {
      profile->profiler.dump_top(20, static_cast<int>(n_calls));
    }

    // ============ 4. 写出 WAV ============
    write_wav_pcm16(options.output_path, full_waveform, output_sample_rate);
    std::cerr << "Saved audio to " << options.output_path << "\n";

    // ============ 5. 与参考波形对拍 max|Δ| ============
    const std::string reference_path =
        (std::filesystem::path(options.mock_dir) / "reference_streaming.wav")
            .string();
    if (std::filesystem::is_regular_file(reference_path)) {
      const WavData ref = read_wav_file(reference_path);
      const std::vector<float> ref_mono = convert_to_mono(ref);
      if (ref.sample_rate != output_sample_rate) {
        std::cerr << "Warning: reference sample rate " << ref.sample_rate
                  << " != model output sample rate " << output_sample_rate
                  << "\n";
      }
      const size_t n = std::min(ref_mono.size(), full_waveform.size());
      float max_abs = 0.0f;
      for (size_t k = 0; k < n; ++k) {
        max_abs = std::max(max_abs, std::fabs(ref_mono[k] - full_waveform[k]));
      }
      std::cerr << "Compare with reference_streaming.wav: length "
                << ref_mono.size() << " vs " << full_waveform.size()
                << ", max|delta| = " << std::scientific << max_abs << " -> "
                << (max_abs == 0.0f ? "PASS" : "DIFF") << "\n";
    } else {
      std::cerr << "No reference_streaming.wav found, skip comparison.\n";
    }

    // ============ 6. 与 --compare-wav 指定的 WAV 对拍（max|Δ| + SNR）============
    if (!options.compare_wav.empty()) {
      if (!std::filesystem::is_regular_file(options.compare_wav)) {
        fail("Compare WAV does not exist: " + options.compare_wav);
      }
      const WavData base = read_wav_file(options.compare_wav);
      const std::vector<float> base_mono = convert_to_mono(base);
      const size_t n = std::min(base_mono.size(), full_waveform.size());
      double max_abs = 0.0;
      double sig_energy = 0.0;
      double err_energy = 0.0;
      for (size_t k = 0; k < n; ++k) {
        const double d = static_cast<double>(base_mono[k]) -
                         static_cast<double>(full_waveform[k]);
        max_abs = std::max(max_abs, std::fabs(d));
        sig_energy += static_cast<double>(base_mono[k]) *
                      static_cast<double>(base_mono[k]);
        err_energy += d * d;
      }
      const double snr_db =
          err_energy > 0.0 ? 10.0 * std::log10(sig_energy / err_energy)
                           : std::numeric_limits<double>::infinity();
      std::cerr << "Compare with " << options.compare_wav << ": length "
                << base_mono.size() << " vs " << full_waveform.size()
                << ", max|delta| = " << std::scientific << max_abs
                << ", SNR = " << std::fixed << std::setprecision(1) << snr_db
                << " dB\n";
    }
    delete l2plan_ptr;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
