#pragma once

// Suprnova wire contract: https://prl.suprnova.cc/stratum-spec.html
// The specification is CC-BY-4.0. This is an independent implementation.
#include <boost/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tnn::pearl::stratum {

inline constexpr size_t max_frame_bytes = 4'000'000;
inline constexpr size_t max_proof_base64 = 2'500'000;
using Clock = std::chrono::steady_clock;

inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

inline std::string string_field(const boost::json::object& object, const char* key) {
    const auto* value = object.if_contains(key);
    require(value && value->is_string(), "Missing or invalid string field");
    return std::string(value->as_string());
}

inline unsigned hex_digit(char value) {
    if (value >= '0' && value <= '9') return unsigned(value - '0');
    if (value >= 'a' && value <= 'f') return unsigned(value - 'a' + 10);
    if (value >= 'A' && value <= 'F') return unsigned(value - 'A' + 10);
    throw std::invalid_argument("Invalid hexadecimal byte");
}

template <size_t Size>
std::array<uint8_t, Size> decode_hex(std::string_view text) {
    require(text.size() == Size * 2, "Incorrect hexadecimal field length");
    std::array<uint8_t, Size> result{};
    for (size_t i = 0; i < Size; ++i) {
        result[i] = uint8_t((hex_digit(text[2 * i]) << 4) | hex_digit(text[2 * i + 1]));
    }
    return result;
}

inline bool meets_target(const std::array<uint8_t, 32>& digest_le,
                         const std::array<uint8_t, 32>& target_le) {
    for (size_t i = 32; i-- > 0;) {
        if (digest_le[i] != target_le[i]) return digest_le[i] < target_le[i];
    }
    return true;
}

struct Job {
    std::string id;
    std::array<uint8_t, 76> header{};
    std::array<uint8_t, 32> target_le{};
    uint64_t height = 0;
    uint64_t generation = 0;
    uint32_t cert_version = 0;
};

// Suprnova's observed share-difficulty scale (the captured d=200 target
// equals floor(2^238 / 200)). Display only: never reconstruct a validation
// target from this floating-point value or interpret the opaque job ID.
inline double share_difficulty(const std::array<uint8_t, 32>& target_le) {
    double target = 0;
    for (size_t i = target_le.size(); i-- > 0;) {
        target = std::ldexp(target, 8) + target_le[i];
    }
    require(target > 0, "Zero share target");
    return std::ldexp(1.0, 238) / target;
}

inline std::string authorization_password(bool is_dev, const std::string& user_password) {
    return is_dev ? "x" : user_password;
}

inline Job parse_job(const boost::json::object& packet, uint64_t generation, uint32_t fallback = 0) {
    require(string_field(packet, "method") == "mining.notify", "Not a Pearl notification");
    const auto* params = packet.if_contains("params");
    require(params && params->is_object(), "Pearl params must be an object");
    const auto& fields = params->as_object();
    Job result;
    const auto* version = fields.if_contains("cert_version");
    require(fallback <= 2, "Invalid certificate fallback");
    if (version) {
        require(version->is_uint64() || version->is_int64(), "Invalid certificate version type");
        const auto value = version->to_number<int64_t>();
        require(value >= 1 && value <= 3, "Unsupported certificate version");
        result.cert_version = uint32_t(value);
    } else {
        require(fallback == 1 || fallback == 2, "Missing certificate version; explicit legacy fallback required");
        result.cert_version = fallback;
    }
    result.id = string_field(fields, "job_id");
    require(!result.id.empty(), "Empty job ID");
    result.header = decode_hex<76>(string_field(fields, "header"));
    result.target_le = decode_hex<32>(string_field(fields, "target"));
    std::reverse(result.target_le.begin(), result.target_le.end());
    require(std::any_of(result.target_le.begin(), result.target_le.end(),
                        [](uint8_t byte) { return byte != 0; }), "Zero share target");
    const auto* height = fields.if_contains("height");
    require(height && (height->is_uint64() || (height->is_int64() && height->as_int64() >= 0)),
            "Missing or invalid height");
    result.height = height->to_number<uint64_t>();
    result.generation = generation;
    return result;
}

inline boost::json::object authorize(const std::string& wallet, const std::string& worker,
                                     const std::string& password, const std::string& agent) {
    require(!wallet.empty(), "Missing Pearl wallet");
    return {{"id", 1}, {"method", "mining.authorize"},
            {"params", {{"wallet", wallet}, {"worker", worker}, {"pass", password}, {"agent", agent}}}};
}

inline void validate_base64(std::string_view text) {
    require(!text.empty() && text.size() <= max_proof_base64 && text.size() % 4 == 0,
            "Invalid proof size");
    size_t padding = 0;
    if (text.back() == '=') ++padding;
    if (text.size() > 1 && text[text.size() - 2] == '=') ++padding;
    for (size_t i = 0; i < text.size() - padding; ++i) {
        const char c = text[i];
        require((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9') || c == '+' || c == '/', "Invalid proof base64");
    }
}

inline boost::json::object submit(uint64_t request_id, const Job& job, const std::string& proof) {
    require(request_id > 1 && !job.id.empty(), "Invalid submission identity");
    validate_base64(proof);
    return {{"id", request_id}, {"method", "mining.submit"},
            {"params", {{"job_id", job.id}, {"plain_proof", proof}}}};
}

// Per-connection state, accessed by the owning session strand only.
class Jobs {
public:
    explicit Jobs(uint64_t generation) : generation_(generation) {}

    bool update(Job job, Clock::time_point now) {
        require(job.generation == generation_, "Wrong connection generation");
        if (current_ && current_->id == job.id && current_->header == job.header &&
            current_->target_le == job.target_le && current_->height == job.height &&
            current_->cert_version == job.cert_version) return false;

        // One previous difficulty generation, not an unbounded stale-job cache.
        previous_.reset();
        if (current_ && current_->header == job.header && current_->height == job.height &&
            current_->target_le != job.target_le && current_->cert_version == job.cert_version) {
            previous_ = current_;
            expires_ = now + std::chrono::seconds(30);
        }
        current_ = std::move(job);
        return true;
    }

    bool eligible(const Job& job, Clock::time_point now) const {
        auto same = [&](const std::optional<Job>& candidate) {
            return candidate && candidate->generation == job.generation &&
                   candidate->id == job.id && candidate->header == job.header &&
                   candidate->target_le == job.target_le && candidate->height == job.height &&
                   candidate->cert_version == job.cert_version;
        };
        return same(current_) || (now < expires_ && same(previous_));
    }

    const std::optional<Job>& current() const { return current_; }

private:
    uint64_t generation_;
    std::optional<Job> current_;
    std::optional<Job> previous_;
    Clock::time_point expires_{};
};

// Accept split and coalesced TCP reads without accepting oversized partial lines.
class Frames {
public:
    std::vector<boost::json::object> feed(std::string_view bytes) {
        std::vector<boost::json::object> packets;
        while (!bytes.empty()) {
            const auto newline = bytes.find('\n');
            const auto count = newline == std::string_view::npos ? bytes.size() : newline;
            require(pending_.size() + count + 1 <= max_frame_bytes, "Oversized Stratum frame");
            pending_.append(bytes.data(), count);
            bytes.remove_prefix(count);
            if (newline == std::string_view::npos) break;
            bytes.remove_prefix(1);
            auto value = boost::json::parse(pending_);
            require(value.is_object(), "Stratum frame must be an object");
            packets.push_back(std::move(value.as_object()));
            pending_.clear();
        }
        return packets;
    }

private:
    std::string pending_;
};

struct Reply {
    uint64_t id = 0;
    bool accepted = false;
    std::optional<int64_t> error_code;
    std::string error_message;
};

inline Reply parse_reply(const boost::json::object& packet) {
    Reply reply;
    const auto* id = packet.if_contains("id");
    require(id && (id->is_uint64() || (id->is_int64() && id->as_int64() >= 0)),
            "Invalid response ID");
    reply.id = id->to_number<uint64_t>();
    const auto* error = packet.if_contains("error");
    const auto* result = packet.if_contains("result");
    require(error && result, "Incomplete Stratum response");
    if (!error->is_null()) {
        require(error->is_object() && result->is_null(), "Invalid error response");
        const auto& fields = error->as_object();
        const auto* code = fields.if_contains("code");
        require(code && code->is_int64(), "Invalid error code");
        reply.error_code = code->as_int64();
        reply.error_message = string_field(fields, "msg");
    } else {
        require(result->is_bool(), "Invalid success response");
        reply.accepted = result->as_bool();
    }
    return reply;
}

// Wire-state bookkeeping; the transport must serialize calls on its strand.
// A new instance is required on reconnect: requests are never replayed.
class Connection {
public:
    explicit Connection(uint64_t generation) : jobs(generation) {}

    void begin_submit(uint64_t id, const Job& job, int device, Clock::time_point now) {
        require(authorized_, "Submit before authorization");
        require(jobs.eligible(job, now), "Stale submission");
        require(pending_.empty(), "Only one outstanding submit per connection");
        require(id > last_submit_id_ && id > 1, "Reused submission ID");
        pending_.emplace(id, device);
        last_submit_id_ = id;
    }

    // Returns device attribution only for an acknowledged submission.
    std::optional<int> acknowledge(const Reply& reply) {
        if (reply.id == 1) {
            require(!auth_answered_, "Duplicate authorization response");
            auth_answered_ = true;
            authorized_ = reply.accepted;
            return std::nullopt;
        }
        const auto found = pending_.find(reply.id);
        require(found != pending_.end(), "Unsolicited or duplicate response");
        const int device = found->second;
        pending_.erase(found);
        return device;
    }

    bool authorized() const { return authorized_; }
    bool submission_pending() const { return !pending_.empty(); }
    Jobs jobs;

private:
    bool auth_answered_ = false;
    bool authorized_ = false;
    uint64_t last_submit_id_ = 1;
    std::map<uint64_t, int> pending_;
};

} // namespace tnn::pearl::stratum
