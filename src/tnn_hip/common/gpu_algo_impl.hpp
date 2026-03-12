#pragma once
#include "gpu_algo.hpp"
#include "gpu_rtc.hpp"
#include "tnn_log.hpp"
#include <coins/miners.hpp>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <thread>
#include <ctime>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
    #include <climits>
#endif

namespace fs = std::filesystem;

inline std::mutex g_tune_output_mutex;

class TuneOutputBuffer {
public:
    explicit TuneOutputBuffer(int device_id = -1) : device_id_(device_id) {}
    
    // Printf-style append to buffer
    template<typename... Args>
    TuneOutputBuffer& printf(const char* fmt, Args... args) {
        char buf[2048];
        int len = snprintf(buf, sizeof(buf), fmt, args...);
        if (len > 0) {
            buffer_.append(buf, std::min((size_t)len, sizeof(buf) - 1));
        }
        return *this;
    }
    
    // Append string directly
    TuneOutputBuffer& append(const std::string& s) {
        buffer_ += s;
        return *this;
    }
    
    TuneOutputBuffer& append(const char* s) {
        buffer_ += s;
        return *this;
    }
    
    // Add a horizontal separator
    TuneOutputBuffer& separator(char c = '=', int len = 60) {
        buffer_.append(len, c);
        buffer_ += '\n';
        return *this;
    }
    
    // Add blank line
    TuneOutputBuffer& newline() {
        buffer_ += '\n';
        return *this;
    }
    
    // Flush buffer to stdout under lock
    void flush() {
        if (buffer_.empty()) return;
        
        std::lock_guard<std::mutex> lock(g_tune_output_mutex);
        ::printf("%s", buffer_.c_str());
        ::fflush(stdout);
        buffer_.clear();
    }
    
    // Auto-flush on destruction
    ~TuneOutputBuffer() {
        flush();
    }
    
    // Prevent copying (could cause double-flush issues)
    TuneOutputBuffer(const TuneOutputBuffer&) = delete;
    TuneOutputBuffer& operator=(const TuneOutputBuffer&) = delete;
    
    // Allow moving
    TuneOutputBuffer(TuneOutputBuffer&& other) noexcept 
        : buffer_(std::move(other.buffer_)), device_id_(other.device_id_) {
        other.buffer_.clear();
    }
    
    TuneOutputBuffer& operator=(TuneOutputBuffer&& other) noexcept {
        if (this != &other) {
            flush();  // Flush our current content first
            buffer_ = std::move(other.buffer_);
            device_id_ = other.device_id_;
            other.buffer_.clear();
        }
        return *this;
    }
    
    bool empty() const { return buffer_.empty(); }
    
private:
    std::string buffer_;
    int device_id_;
};

class GPUAlgorithm : public IGPUAlgorithm
{
public:
    explicit GPUAlgorithm(const AlgoConfig &config)
        : config_(config), initialized_(false) {}

    ~GPUAlgorithm() override
    {
        cleanup();
    }

    TuningResult get_tuning_result() const override {
        return tuning_result_;
    }
    
    bool set_batch_size_override(uint32_t batch_size) override {
        if (!initialized_) return false;

        // Ensure correct device context
        hipSetDevice(device_id_);

        batch_size = (batch_size / block_size_) * block_size_;
        if (batch_size == 0) batch_size = block_size_;

        size_t required = batch_size * config_.scratch_per_hash;
        size_t free_mem, total_mem;
        hipMemGetInfo(&free_mem, &total_mem);
        
        if (required > free_mem * config_.memory_usage_factor) {
            return false;
        }
        
        if (batch_size != batch_size_) {
            cleanup_batch_buffers();
            batch_size_ = batch_size;
            num_blocks_ = batch_size_ / block_size_;
            
            if (!allocate_batch_buffers()) {
                return false;
            }
        }
        
        return true;
    }

    bool initialize(int device_id = 0) override
    {
        printf("[TRACE] GPUAlgorithm::initialize: Entry for device %d\n", device_id);
        fflush(stdout);

        device_id_ = device_id;
        
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            printf("[ERROR] hipSetDevice(%d) failed: %s\n", device_id, hipGetErrorString(err));
            return false;
        }

#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDACC_RTC__)
        {
            void *dummy = nullptr;
            err = hipMalloc(&dummy, 256);
            if (err == hipSuccess) {
                hipFree(dummy);
            } else {
                printf("[ERROR] Failed to initialize CUDA context: %s\n", hipGetErrorString(err));
                return false;
            }
        }
#endif

        hipGetDeviceProperties(&device_props_, device_id);
        compute_units_ = device_props_.multiProcessorCount;
        
        TNN_LOG_INFO("[INFO] GPU %d: %s (%d CUs)\n", device_id, device_props_.name, compute_units_);
        fflush(stdout);

        if (!compile_kernel()) {
            return false;
        }

        if (!configure_batch()) {
            return false;
        }

        if (!allocate_batch_buffers()) {
            return false;
        }

        hipEventCreate(&start_event_);
        hipEventCreate(&stop_event_);

        initialized_ = true;
        
        TNN_LOG_INFO("[INFO] GPU %d initialized: %s\n", device_id, tuning_result_.describe().c_str());
        fflush(stdout);

        return true;
    }

    void cleanup() override
    {
        // Ensure correct device context for cleanup
        if (device_id_ >= 0) hipSetDevice(device_id_);

        cleanup_batch_buffers();
        if (start_event_) { hipEventDestroy(start_event_); start_event_ = nullptr; }
        if (stop_event_) { hipEventDestroy(stop_event_); stop_event_ = nullptr; }
        initialized_ = false;
    }

    void set_work(const uint8_t *work_template, uint64_t difficulty) override
    {
        // Ensure correct device context
        hipSetDevice(device_id_);

        // Save host-side copy for solution verification (prevents race with job updates)
        h_work_template_.resize(config_.template_size);
        memcpy(h_work_template_.data(), work_template, config_.template_size);

        hipMemcpy(d_input_, work_template, config_.template_size, hipMemcpyHostToDevice);
        uint64_t target[4];
        compute_target(difficulty, target);
        hipMemcpy(d_difficulty_target_, target, 32, hipMemcpyHostToDevice);
    }

    // Get the currently-active work template for this miner (for solution verification)
    const uint8_t* get_current_work_template() const {
        return h_work_template_.empty() ? nullptr : h_work_template_.data();
    }

    BatchResult mine_batch(uint64_t nonce_start, uint32_t count = 0) override
    {
        // Ensure correct device context for mining
        hipSetDevice(device_id_);

        if (count == 0) count = batch_size_;

        hipMemset(d_solutions_, 0, 24);

        // Build launch context
        KernelLaunchContext ctx;
        ctx.d_input = d_input_;
        ctx.d_outputs = d_outputs_;
        ctx.d_scratch = d_scratch_;
        ctx.d_difficulty_target = d_difficulty_target_;
        ctx.d_solutions = d_solutions_;
        ctx.nonce_start = nonce_start;
        ctx.batch_size = count;
        ctx.block_size = block_size_;
        ctx.num_blocks = num_blocks_;
        ctx.strategy = tuning_result_.strategy;
        ctx.config = &config_;
        ctx.stream = nullptr;  // Default stream

        hipEventRecord(start_event_);

        // Execute using strategy (custom or default)
        bool success;
        if (config_.execute_fn) {
            success = config_.execute_fn(kernels_, ctx);
        } else {
            success = default_monolithic_execute(kernels_, ctx);
        }

        hipEventRecord(stop_event_);
        hipError_t sync_err = hipEventSynchronize(stop_event_);

        // Check for async kernel errors (illegal memory access, stack overflow, etc.)
        hipError_t last_err = hipGetLastError();

        if (!success) {
            fprintf(stderr, "[ERROR] GPU %d: Kernel launch reported failure\n", device_id_);
        }
        if (sync_err != hipSuccess) {
            fprintf(stderr, "[ERROR] GPU %d: hipEventSynchronize failed: %s\n",
                    device_id_, hipGetErrorString(sync_err));
        }
        if (last_err != hipSuccess) {
            fprintf(stderr, "[ERROR] GPU %d: Async kernel error: %s\n",
                    device_id_, hipGetErrorString(last_err));
        }

        float ms;
        hipEventElapsedTime(&ms, start_event_, stop_event_);
        last_hashrate_ = (count * 1000.0) / ms;

        // Rest unchanged - extract solutions
        BatchResult result;
        result.nonce_start = nonce_start;
        result.count = count;

        uint64_t solution_count = 0;
        hipMemcpy(&solution_count, d_solutions_, sizeof(uint64_t), hipMemcpyDeviceToHost);

        if (solution_count > count) solution_count = 0;
        if (solution_count > 1024) solution_count = 1024;

        result.num_valid = (uint32_t)solution_count;

        if (solution_count > 0) {
            size_t solution_bytes = solution_count * 40;
            std::vector<uint64_t> raw_solutions(solution_count * 5);

            hipMemcpy(raw_solutions.data(), d_solutions_ + 1, solution_bytes, hipMemcpyDeviceToHost);

            result.valid_nonces.reserve(solution_count);
            result.valid_hashes.resize(solution_count * config_.hash_size);

            for (uint32_t i = 0; i < solution_count; i++) {
                result.valid_nonces.push_back(raw_solutions[i * 5]);
                uint8_t *hash_dest = result.valid_hashes.data() + i * config_.hash_size;
                memcpy(hash_dest, &raw_solutions[i * 5 + 1], config_.hash_size);
            }
        }

        result.hashes.clear();
        return result;
    }

    uint32_t get_batch_size() const override { return batch_size_; }
    double get_hashrate() const override { return last_hashrate_; }
    const AlgoConfig &get_config() const override { return config_; }

private:
    // ========================================================================
    // Difficulty Target Calculation
    // ========================================================================
    
    void compute_target(uint64_t difficulty, uint64_t *target_out)
    {
        Num target_bigint = ConvertDifficultyToBig(difficulty, config_.algo_id);
        uint8_t target_bytes[32] = {0};
        size_t num_words = std::min(target_bigint.words.size(), (size_t)4);
        memcpy(target_bytes, target_bigint.words.data(), num_words * sizeof(uint64_t));
        uint8_t *out_bytes = (uint8_t *)target_out;
        for (int i = 0; i < 32; i++) {
            out_bytes[i] = target_bytes[31 - i];
        }
    }

    // ========================================================================
    // Kernel Compilation
    // ========================================================================
    
    bool compile_kernel() {
        TNN_LOG_INFO("[INFO] GPU %d: Starting kernel compilation\n", device_id_);
        fflush(stdout);

        try {
            auto& compiler = RTCCompiler::instance();

            for (const auto &header : config_.rtc_headers) {
                compiler.add_header_source(
                    std::string(header.name),
                    std::string(header.source));
            }

            std::vector<std::string> options;

    #if defined(__HIP_PLATFORM_AMD__)
          options = {"-O3", "-mno-cumode", "-ffast-math"};

          options.push_back("-DXELIS_MIN_WG=" + std::to_string(config_.amd_blocks.block_min));
          options.push_back("-DXELIS_MAX_WG=" + std::to_string(config_.amd_blocks.block_max));

          if (device_props_.gcnArchName[0] != '\0') {
              options.push_back(std::string("--gpu-architecture=") + device_props_.gcnArchName);
          }
    #elif defined(__HIP_PLATFORM_NVIDIA__)
            options = {"--dopt=on", "--use_fast_math"};
          #ifdef __linux__
            options.push_back("--device-int128");
          #endif

            options.push_back("-DXELIS_MIN_WG=" + std::to_string(config_.nvidia_blocks.block_min));
            options.push_back("-DXELIS_MAX_WG=" + std::to_string(config_.nvidia_blocks.block_max));

            // Per-kernel register limits via -D defines (arch-dependent)
            {
                const int major = device_props_.major;
                int s3_nreg = 40; // (major <= 7) ? 56 : 40;  // Turing/Volta and older: 56, Ampere+: 40
                options.push_back("-DXELIS_S3_NREG=" + std::to_string(s3_nreg));
            }

            // Per-device module key (NVIDIA modules are bound to a CUDA context)
            options.push_back("-DDEVICE_ID=" + std::to_string(device_id_));
    #endif

            // Compile module once
            RTCCompiler::CompiledKernel compiled;
            std::string primary_kernel = config_.get_primary_kernel();
            
            if (!config_.source.empty()) {
                compiled = compiler.compile_from_source(
                    std::string(config_.source),
                    config_.source_path,
                    primary_kernel,
                    options);
            } else {
                compiled = compiler.compile(
                    config_.source_path,
                    primary_kernel,
                    options);
            }
            
            module_ = compiled.module;
            
            // Load all kernels from the module
            for (const auto& kernel_name : config_.get_kernel_names()) {
                hipFunction_t func = nullptr;
                hipError_t err = hipModuleGetFunction(&func, module_, kernel_name.c_str());
                
                if (err == hipSuccess && func != nullptr) {
                    kernels_[kernel_name] = func;
                    TNN_LOG_INFO("[INFO] GPU %d: Loaded kernel '%s'\n", device_id_, kernel_name.c_str());
                } else {
                    printf("[WARN] GPU %d: Could not load kernel '%s': %s\n", 
                           device_id_, kernel_name.c_str(), hipGetErrorString(err));
                }
            }
            
            if (kernels_.empty()) {
                fprintf(stderr, "[ERROR] No kernels loaded!\n");
                return false;
            }

            // Debug: Print loaded kernels for this device
            TNN_LOG_INFO("[INFO] GPU %d: Loaded %zu kernel(s):\n", device_id_, kernels_.size());
            for (const auto& kv : kernels_) {
                TNN_LOG_INFO("[INFO] GPU %d:   - %s\n", device_id_, kv.first.c_str());
            }
            if (config_.execute_fn) {
                TNN_LOG_INFO("[INFO] GPU %d: Using custom execution strategy\n", device_id_);
            } else {
                TNN_LOG_INFO("[INFO] GPU %d: Using default monolithic execution\n", device_id_);
            }

            return true;
        }
        catch (const std::exception &e) {
            fprintf(stderr, "[ERROR] RTC compilation failed: %s\n", e.what());
            return false;
        }
    }

    // ========================================================================
    // Buffer Management
    // ========================================================================
    
    bool allocate_batch_buffers() {
        // Ensure correct device context for allocation
        hipError_t set_err = hipSetDevice(device_id_);
        if (set_err != hipSuccess) {
            fprintf(stderr, "[ERROR] GPU %d: hipSetDevice failed before allocation: %s\n",
                    device_id_, hipGetErrorString(set_err));
            return false;
        }

        hipError_t err;

        size_t scratch_size = batch_size_ * config_.scratch_per_hash;
        TNN_LOG_INFO("[INFO] GPU %d: Allocating buffers (batch=%u, scratch=%zu MB)\n",
               device_id_, batch_size_, scratch_size / (1024*1024));
        fflush(stdout);

        err = hipMalloc(&d_input_, config_.template_size);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR] GPU %d: hipMalloc d_input_ (%zu bytes) failed: %s\n",
                    device_id_, config_.template_size, hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(&d_outputs_, batch_size_ * config_.hash_size);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR] GPU %d: hipMalloc d_outputs_ (%zu bytes) failed: %s\n",
                    device_id_, batch_size_ * config_.hash_size, hipGetErrorString(err));
            cleanup_batch_buffers();
            return false;
        }

        err = hipMalloc(&d_scratch_, scratch_size);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR] GPU %d: hipMalloc d_scratch_ (%zu MB) failed: %s\n",
                    device_id_, scratch_size / (1024*1024), hipGetErrorString(err));
            cleanup_batch_buffers();
            return false;
        }

        err = hipMalloc(&d_difficulty_target_, 32);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR] GPU %d: hipMalloc d_difficulty_target_ failed: %s\n",
                    device_id_, hipGetErrorString(err));
            cleanup_batch_buffers();
            return false;
        }

        size_t solutions_size = 8 + 1024 * 40 + 16;
        err = hipMalloc(&d_solutions_, solutions_size);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR] GPU %d: hipMalloc d_solutions_ failed: %s\n",
                    device_id_, hipGetErrorString(err));
            cleanup_batch_buffers();
            return false;
        }

        TNN_LOG_INFO("[INFO] GPU %d: Buffer allocation successful\n", device_id_);
        fflush(stdout);
        return true;
    }
    
    void cleanup_batch_buffers() {
        // Ensure correct device context for deallocation
        if (device_id_ >= 0) hipSetDevice(device_id_);

        if (d_input_) { hipFree(d_input_); d_input_ = nullptr; }
        if (d_outputs_) { hipFree(d_outputs_); d_outputs_ = nullptr; }
        if (d_scratch_) { hipFree(d_scratch_); d_scratch_ = nullptr; }
        if (d_difficulty_target_) { hipFree(d_difficulty_target_); d_difficulty_target_ = nullptr; }
        if (d_solutions_) { hipFree(d_solutions_); d_solutions_ = nullptr; }
    }

    // ========================================================================
    // Batch Configuration
    // ========================================================================
    
    bool configure_batch() {
        auto batch_override = g_tuning_overrides.get_batch_override(device_id_);
        auto block_override = g_tuning_overrides.get_block_override(device_id_);
        
        if (batch_override.has_value()) {
            block_size_ = block_override.value_or(config_.preferred_block_size);
            batch_size_ = batch_override.value();
            batch_size_ = (batch_size_ / block_size_) * block_size_;
            num_blocks_ = batch_size_ / block_size_;
            
            TNN_LOG_INFO("[INFO] GPU %d: Using CLI override batch_size=%u, block_size=%d\n",
                   device_id_, batch_size_, block_size_);
            
            tuning_result_.block_size = block_size_;
            tuning_result_.num_blocks = num_blocks_;
            tuning_result_.batch_size = batch_size_;
            tuning_result_.valid = true;
            return true;
        }
        
        if (config_.enable_autotune && !g_tuning_overrides.disable_autotune) {
            return run_autotune();
        } else {
            return calculate_static_batch();
        }
    }
    
    bool calculate_static_batch() {
        // Ensure correct device context for memory queries
        hipSetDevice(device_id_);

        block_size_ = config_.preferred_block_size;

        size_t free_mem, total_mem;
        hipMemGetInfo(&free_mem, &total_mem);

        size_t reserved = (size_t)(config_.memory_reserve_mb * 1024 * 1024);
        size_t available = (size_t)((free_mem - reserved) * config_.memory_usage_factor);

        uint32_t max_by_mem = available / config_.scratch_per_hash;

#ifdef __HIP_PLATFORM_AMD__
        const int occupancy_factor = 4;
#else
        const int occupancy_factor = 2;
#endif

        uint32_t max_concurrent = compute_units_ * occupancy_factor * block_size_;

        batch_size_ = std::min(max_by_mem, max_concurrent);
        batch_size_ = (batch_size_ / block_size_) * block_size_;
        num_blocks_ = batch_size_ / block_size_;
        
        tuning_result_.block_size = block_size_;
        tuning_result_.num_blocks = num_blocks_;
        tuning_result_.batch_size = batch_size_;
        tuning_result_.valid = true;
        
        return batch_size_ > 0;
    }

    // ========================================================================
    // Tuning Cache (Disk Persistence)
    // ========================================================================

    static fs::path get_executable_dir() {
        static fs::path cached_path;
        static bool initialized = false;

        if (initialized) {
            return cached_path;
        }

#ifdef _WIN32
        std::vector<wchar_t> path_buf(MAX_PATH);
        DWORD len;

        do {
            len = GetModuleFileNameW(nullptr, path_buf.data(), (DWORD)path_buf.size());
            if (len == 0) {
                // Failed, fallback to current directory
                cached_path = fs::current_path();
                initialized = true;
                return cached_path;
            }
            if (len < path_buf.size()) {
                break;
            }
            path_buf.resize(path_buf.size() * 2);
        } while (true);

        cached_path = fs::path(path_buf.data()).parent_path();
#else
        // Linux/Unix
        char path_buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", path_buf, sizeof(path_buf) - 1);

        if (len != -1) {
            path_buf[len] = '\0';
            cached_path = fs::path(path_buf).parent_path();
        } else {
            // Fallback to current directory if readlink fails
            cached_path = fs::current_path();
        }
#endif

        initialized = true;
        return cached_path;
    }

    std::string get_tune_cache_path() const {
        std::string vendor;
        std::string arch;
        
#ifdef __HIP_PLATFORM_AMD__
        vendor = "amd";
        arch = device_props_.gcnArchName;
#else
        vendor = "nvidia";
        char buf[32];
        snprintf(buf, sizeof(buf), "sm_%d%d", device_props_.major, device_props_.minor);
        arch = buf;
#endif
        
        // Round VRAM to nearest GB
        size_t vram_gb = (device_props_.totalGlobalMem + 512ULL * 1024 * 1024) / (1024ULL * 1024 * 1024);
        
        // Sanitize arch string
        for (char& c : arch) {
            if (!isalnum(c) && c != '_' && c != '-') c = '_';
        }

        fs::path cache_dir = get_executable_dir() / "tunes" / vendor;
        
        std::error_code ec;
        fs::create_directories(cache_dir, ec);
        
        char filename[128];
        snprintf(filename, sizeof(filename), "%s_%zugb_%s.txt", 
                 arch.c_str(), vram_gb, config_.name.c_str());
        
        return (cache_dir / filename).string();
    }

    bool load_cached_tune() {
        // Ensure correct device context for memory validation
        hipSetDevice(device_id_);

        std::string path = get_tune_cache_path();
        std::ifstream f(path);
        if (!f.is_open()) return false;

        try {
            std::string line;
            TuningResult cached;
            int cached_compute_units = 0;
            
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#') continue;
                
                size_t eq = line.find('=');
                if (eq == std::string::npos) continue;
                
                std::string key = line.substr(0, eq);
                std::string val = line.substr(eq + 1);
                
                if (key == "block_size") cached.block_size = std::stoi(val);
                else if (key == "num_blocks") cached.num_blocks = std::stoi(val);
                else if (key == "batch_size") cached.batch_size = std::stoul(val);
                else if (key == "hashrate") cached.hashrate = std::stod(val);
                else if (key == "batch_time_ms") cached.batch_time_ms = std::stod(val);
                else if (key == "compute_units") cached_compute_units = std::stoi(val);
                else if (key == "strategy") cached.strategy = (uint8_t)std::stoi(val);
            }
            
            // Validate
            if (cached_compute_units != 0 && cached_compute_units != compute_units_) {
                TuneOutputBuffer out(device_id_);
                out.printf("[AUTOTUNE] GPU %d: Cache CU mismatch (%d vs %d), re-tuning\n", 
                           device_id_, cached_compute_units, compute_units_);
                return false;
            }
            
            if (cached.block_size > 0 && cached.batch_size > 0) {
                size_t required = cached.batch_size * config_.scratch_per_hash;
                size_t free_mem, total_mem;
                hipMemGetInfo(&free_mem, &total_mem);
                
                if (required > free_mem * 0.95) {
                    TuneOutputBuffer out(device_id_);
                    out.printf("[AUTOTUNE] GPU %d: Cache batch_size too large for memory, re-tuning\n",
                               device_id_);
                    return false;
                }
                
                cached.valid = true;
                tuning_result_ = cached;
                block_size_ = cached.block_size;
                num_blocks_ = cached.num_blocks;
                batch_size_ = cached.batch_size;
                
                return true;  // Caller will print the success message
            }
        } catch (const std::exception& e) {
            TuneOutputBuffer out(device_id_);
            out.printf("[AUTOTUNE] GPU %d: Cache parse error: %s\n", device_id_, e.what());
        }
        
        return false;
    }

    void save_tune_cache() const {
        std::string path = get_tune_cache_path();
        std::ofstream f(path);
        if (!f.is_open()) {
            TuneOutputBuffer out(device_id_);
            out.printf("[AUTOTUNE] GPU %d: Warning: Could not save cache to %s\n", 
                       device_id_, path.c_str());
            return;
        }
        
        time_t now = time(nullptr);
        char time_buf[64];
        strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
        
        f << "# Tuning cache for " << config_.name << "\n";
        f << "# Device: " << device_props_.name << "\n";
        f << "# Generated: " << time_buf << "\n";
        f << "compute_units=" << compute_units_ << "\n";
        f << "strategy=" << (int)tuning_result_.strategy << "\n";
        f << "block_size=" << tuning_result_.block_size << "\n";
        f << "num_blocks=" << tuning_result_.num_blocks << "\n";
        f << "batch_size=" << tuning_result_.batch_size << "\n";
        f << "hashrate=" << tuning_result_.hashrate << "\n";
        f << "batch_time_ms=" << tuning_result_.batch_time_ms << "\n";
        
        TuneOutputBuffer out(device_id_);
        out.printf("[AUTOTUNE] GPU %d: Saved cache to %s\n", device_id_, path.c_str());
    }

    // ========================================================================
    // Auto-Tuning with Timeout Support
    // ========================================================================
    
    struct TuneTestResult {
        bool valid = false;
        bool timed_out = false;
        double time_ms = 0;
        double hashrate = 0;
    };

private:
    TuneTestResult run_timed_kernel_test(
        uint32_t test_batch,
        int test_block_size,
        int test_num_blocks,
        uint8_t* test_input,
        uint8_t* test_outputs,
        uint64_t* test_scratch,
        uint64_t* test_target,
        uint64_t* test_solutions,
        hipStream_t stream,
        double timeout_ms,
        uint8_t test_strategy = 0
    ) {
        TuneTestResult result;

        // CRITICAL: Ensure device context is correct for this test
        // Events and kernel launches must happen on the same device
        hipError_t dev_err = hipSetDevice(device_id_);
        if (dev_err != hipSuccess) {
            return result;  // Invalid result
        }

        // Build context
        KernelLaunchContext ctx;
        ctx.d_input = test_input;
        ctx.d_outputs = test_outputs;
        ctx.d_scratch = test_scratch;
        ctx.d_difficulty_target = test_target;
        ctx.d_solutions = test_solutions;
        ctx.nonce_start = 0;
        ctx.batch_size = test_batch;
        ctx.block_size = test_block_size;
        ctx.num_blocks = test_num_blocks;
        ctx.strategy = test_strategy;
        ctx.config = &config_;
        ctx.stream = stream;
        
        hipMemsetAsync(test_solutions, 0, 8, stream);

        hipEvent_t start_ev, stop_ev;
        hipEventCreate(&start_ev);
        hipEventCreate(&stop_ev);

        hipEventRecord(start_ev, stream);

        // Check if kernels are loaded
        if (kernels_.empty()) {
            TNN_LOG_DEBUG("[TUNE DEBUG] GPU %d: No kernels loaded!\n", device_id_);
            hipEventDestroy(start_ev);
            hipEventDestroy(stop_ev);
            return result;
        }

        // Execute using strategy
        bool success;
        if (config_.execute_fn) {
            success = config_.execute_fn(kernels_, ctx);
        } else {
            success = default_monolithic_execute(kernels_, ctx);
        }

        if (!success) {
            TNN_LOG_DEBUG("[TUNE DEBUG] GPU %d: Kernel launch returned failure\n", device_id_);
            hipEventDestroy(start_ev);
            hipEventDestroy(stop_ev);
            return result;
        }

        // Check for kernel launch errors (asynchronous errors)
        hipError_t launch_err = hipGetLastError();
        if (launch_err != hipSuccess) {
            TNN_LOG_DEBUG("[TUNE DEBUG] GPU %d: Kernel launch error: %s\n",
                    device_id_, hipGetErrorString(launch_err));
            hipEventDestroy(start_ev);
            hipEventDestroy(stop_ev);
            return result;
        }

        hipEventRecord(stop_ev, stream);
        
        // Poll for completion with timeout
        auto wall_start = std::chrono::steady_clock::now();
        const int poll_interval_ms = 10;
        
        while (true) {
            hipError_t query = hipEventQuery(stop_ev);
            if (query == hipSuccess) {
                break;
            }
            
            if (query != hipErrorNotReady) {
                hipEventDestroy(start_ev);
                hipEventDestroy(stop_ev);
                return result;
            }
            
            auto elapsed = std::chrono::steady_clock::now() - wall_start;
            double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
            
            if (elapsed_ms > timeout_ms) {
                result.timed_out = true;
                hipStreamSynchronize(stream);
                
                float actual_ms;
                hipEventElapsedTime(&actual_ms, start_ev, stop_ev);
                
                result.valid = true;
                result.time_ms = actual_ms;
                result.hashrate = (test_batch * 1000.0) / actual_ms;
                
                hipEventDestroy(start_ev);
                hipEventDestroy(stop_ev);
                return result;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
        }
        
        // Check for async kernel errors after completion
        hipError_t post_err = hipGetLastError();
        if (post_err != hipSuccess) {
            fprintf(stderr, "[AUTOTUNE] GPU %d: Kernel execution error: %s (batch=%u, block=%d, strategy=%u)\n",
                    device_id_, hipGetErrorString(post_err), test_batch, test_block_size, test_strategy);
            fflush(stderr);
            hipEventDestroy(start_ev);
            hipEventDestroy(stop_ev);
            return result;
        }

        float ms;
        hipError_t time_err = hipEventElapsedTime(&ms, start_ev, stop_ev);

        if (time_err != hipSuccess) {
            fprintf(stderr, "[AUTOTUNE] GPU %d: hipEventElapsedTime failed: %s\n",
                    device_id_, hipGetErrorString(time_err));
            fflush(stderr);
            hipEventDestroy(start_ev);
            hipEventDestroy(stop_ev);
            return result;
        }

        if (ms < 0.001f) {
            TNN_LOG_DEBUG("[TUNE DEBUG] GPU %d: Suspicious 0ms timing - batch=%u, block=%d, stream=%p\n",
                    device_id_, test_batch, test_block_size, (void*)stream);
        }

        result.valid = true;
        result.timed_out = false;
        result.time_ms = ms;
        result.hashrate = (test_batch * 1000.0) / ms;

        hipEventDestroy(start_ev);
        hipEventDestroy(stop_ev);
        return result;
    }

    bool run_autotune() {
        // CRITICAL: Ensure we're on the correct device for all tuning operations
        // In multi-GPU setups, device context can switch between threads
        hipError_t err = hipSetDevice(device_id_);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR] run_autotune: hipSetDevice(%d) failed: %s\n",
                    device_id_, hipGetErrorString(err));
            return calculate_static_batch();
        }

        // Generate tune key for this config
        std::string tune_key = TuneCoordinator::make_tune_key(device_props_, config_.name);

        // Try to claim tuning rights
        auto status = TuneCoordinator::instance().begin_tune(tune_key, device_id_);
        
        switch (status) {
            case TuneCoordinator::TuneStatus::ALREADY_DONE: {
                // Another identical GPU already tuned this session, just load cache
                TuneOutputBuffer out(device_id_);
                out.printf("[AUTOTUNE] GPU %d: Tune already done this session for %s, loading cache\n",
                           device_id_, tune_key.c_str());
                out.flush();
                
                if (load_cached_tune()) {
                    TuneOutputBuffer out2(device_id_);
                    out2.printf("[AUTOTUNE] GPU %d: Loaded cached tune: %s\n", 
                               device_id_, tuning_result_.describe().c_str());
                    return true;
                }
                // Cache load failed somehow, fall through to static
                return calculate_static_batch();
            }
            
            case TuneCoordinator::TuneStatus::WAIT_FOR_OTHER: {
                // Another GPU is currently tuning this config
                TuneOutputBuffer out(device_id_);
                out.printf("[AUTOTUNE] GPU %d: Waiting for another GPU to finish tuning %s...\n",
                           device_id_, tune_key.c_str());
                out.flush();
                
                bool success = TuneCoordinator::instance().wait_for_tune(tune_key);
                
                if (success && load_cached_tune()) {
                    TuneOutputBuffer out2(device_id_);
                    out2.printf("[AUTOTUNE] GPU %d: Loaded tune from other GPU: %s\n", 
                               device_id_, tuning_result_.describe().c_str());
                    return true;
                }
                
                // Other GPU's tune failed or cache load failed
                TuneOutputBuffer out3(device_id_);
                out3.printf("[AUTOTUNE] GPU %d: Other GPU's tune failed, using static fallback\n",
                           device_id_);
                return calculate_static_batch();
            }
            
            case TuneCoordinator::TuneStatus::SHOULD_TUNE:
                // We're responsible for tuning, continue below
                break;
        }
        
        // Check disk cache first (before doing expensive tuning)
        if (!g_tuning_overrides.force_retune && load_cached_tune()) {
            TuneOutputBuffer out(device_id_);
            out.printf("[AUTOTUNE] GPU %d: Loaded cached tune for %s\n", 
                       device_id_, config_.name.c_str());
            out.printf("[AUTOTUNE] GPU %d: %s\n", 
                       device_id_, tuning_result_.describe().c_str());
            
            // Mark as complete so other identical GPUs can use cache
            TuneCoordinator::instance().end_tune(tune_key, true);
            return true;
        }
        
        // === Actual tuning starts here ===
        
        // Header block
        {
            TuneOutputBuffer out(device_id_);
            out.newline();
            out.separator('=');
            out.printf("[AUTOTUNE] GPU %d: Auto-tuning %s\n", device_id_, config_.name.c_str());
            out.printf("[AUTOTUNE] GPU %d: Device: %s (%d CUs)\n", 
                       device_id_, device_props_.name, compute_units_);
            out.printf("[AUTOTUNE] GPU %d: Tune key: %s\n", device_id_, tune_key.c_str());
            out.separator('=');
        }
        
#ifdef __HIP_PLATFORM_AMD__
        const auto& limits = config_.amd_blocks;
        const int occupancy_factor = 4;
#else
        const auto& limits = config_.nvidia_blocks;
        const int occupancy_factor = 2;
#endif
        
        size_t free_mem, total_mem;
        hipMemGetInfo(&free_mem, &total_mem);
        size_t reserved = (size_t)(config_.memory_reserve_mb * 1024 * 1024);
        size_t max_usable = (size_t)((free_mem - reserved) * config_.memory_usage_factor);
        
        // Config info block
        {
            TuneOutputBuffer out(device_id_);
            out.printf("[AUTOTUNE] GPU %d: Memory: %.1f MB free, %.1f MB max usable\n",
                       device_id_, free_mem / (1024.0 * 1024.0), max_usable / (1024.0 * 1024.0));
            out.printf("[AUTOTUNE] GPU %d: Scratch per hash: %.2f KB\n", 
                       device_id_, config_.scratch_per_hash / 1024.0);
            out.printf("[AUTOTUNE] GPU %d: Block sizes: %d-%d (step %d)\n", 
                       device_id_, limits.block_min, limits.block_max, limits.step);
            out.printf("[AUTOTUNE] GPU %d: Target time: %.0fms, Max: %.0fms\n",
                       device_id_, config_.target_batch_time_ms, config_.max_batch_time_ms);
            
            // Show if other GPUs are waiting
            auto waiters = TuneCoordinator::instance().get_waiters(tune_key);
            if (!waiters.empty()) {
                out.printf("[AUTOTUNE] GPU %d: Other GPUs waiting: ", device_id_);
                for (size_t i = 0; i < waiters.size(); i++) {
                    out.printf("%d%s", waiters[i], (i < waiters.size() - 1) ? ", " : "");
                }
                out.printf("\n");
            }
            out.newline();
        }
        
        hipStream_t tune_stream;
        hipStreamCreate(&tune_stream);

        TuningResult best;
        best.hashrate = 0;
        best.valid = false;

        double timeout_ms = config_.max_batch_time_ms * 1.5;

        // Build strategy list: if algo defines variants, sweep them; otherwise just strategy=0
        std::vector<uint8_t> strategies_to_test;
        if (!config_.strategy_variants.empty()) {
            strategies_to_test = config_.strategy_variants;
        } else {
            strategies_to_test = {0};
        }

        // Occupancy-based tuning: For each strategy × block size, probe for max batch
        {
            TuneOutputBuffer out(device_id_);
            out.printf("[AUTOTUNE] GPU %d: Using occupancy-based tuning (faster than memory percentage sweep)\n", device_id_);
            if (strategies_to_test.size() > 1) {
                out.printf("[AUTOTUNE] GPU %d: Sweeping %zu strategies", device_id_, strategies_to_test.size());
                for (size_t si = 0; si < strategies_to_test.size(); si++) {
                    const char* sname = (si < config_.strategy_names.size())
                        ? config_.strategy_names[si].c_str()
                        : "?";
                    out.printf("%s %s(%u)", si == 0 ? ":" : ",", sname, strategies_to_test[si]);
                }
                out.printf("\n");
            }
            out.newline();
        }

        for (uint8_t test_strategy : strategies_to_test) {

        const char* strategy_label = "";
        if (strategies_to_test.size() > 1) {
            size_t si = 0;
            for (size_t k = 0; k < config_.strategy_variants.size(); k++) {
                if (config_.strategy_variants[k] == test_strategy) { si = k; break; }
            }
            strategy_label = (si < config_.strategy_names.size())
                ? config_.strategy_names[si].c_str()
                : "?";
            TuneOutputBuffer out(device_id_);
            out.printf("[AUTOTUNE] GPU %d: === Strategy: %s (%u) ===\n", device_id_, strategy_label, test_strategy);
            out.flush();
        }

        // Batch probing uses a fixed base unit independent of block_size so that
        // larger block dims still explore the same batch-size range as smaller ones.
        // Half-step multipliers (1x, 1.5x, 2x, 2.5x, ..., 8x) give finer granularity.
        const uint32_t base_batch = compute_units_ * occupancy_factor * limits.block_min;
        const uint32_t max_half_mult = 16;  // 8x in half-steps (2=1x, 3=1.5x, ..., 16=8x)

        // Track the best batch size for each half-multiplier step
        std::vector<uint32_t> best_batch_per_step(max_half_mult + 1, 0);

        for (int test_block_size = limits.block_min;
             test_block_size <= limits.block_max;
             test_block_size += limits.step)
        {
            TuneOutputBuffer block_out(device_id_);
            if (strategies_to_test.size() > 1) {
                block_out.printf("[AUTOTUNE] GPU %d: --- Block size %d [%s] ---\n", device_id_, test_block_size, strategy_label);
            } else {
                block_out.printf("[AUTOTUNE] GPU %d: --- Block size %d ---\n", device_id_, test_block_size);
            }

            // Probe increasing batch sizes using half-step multipliers
            uint32_t successful_batch = 0;
            uint32_t half_step = 2;  // starts at 1x (2/2)

            while (half_step <= max_half_mult) {
                // Compute raw probe batch from fixed base, then align to block_size
                uint64_t raw_batch = (uint64_t)base_batch * half_step / 2;
                uint32_t probe_batch = (uint32_t)((raw_batch / test_block_size) * test_block_size);
                if (probe_batch < (uint32_t)test_block_size) probe_batch = test_block_size;

                // Skip if we've already tested this exact batch (alignment can cause duplicates)
                if (probe_batch <= successful_batch) {
                    half_step++;
                    continue;
                }

                size_t probe_mem = (size_t)probe_batch * config_.scratch_per_hash;
                double mult_display = half_step / 2.0;

                // Check if this would exceed available memory
                bool clamped_to_max = false;

                if (probe_mem > max_usable) {
                    // Clamp to max available memory, aligned to block_size
                    probe_batch = (uint32_t)(max_usable / config_.scratch_per_hash);
                    probe_batch = (probe_batch / test_block_size) * test_block_size;
                    probe_mem = (size_t)probe_batch * config_.scratch_per_hash;
                    clamped_to_max = true;

                    if (probe_batch <= successful_batch || probe_batch < (uint32_t)test_block_size) {
                        break;
                    }
                }

                // Try allocation
                uint64_t* test_scratch = nullptr;
                uint8_t* test_input = nullptr;
                uint8_t* test_outputs = nullptr;
                uint64_t* test_target = nullptr;
                uint64_t* test_solutions = nullptr;

                hipError_t err = hipMalloc(&test_scratch, probe_mem);
                if (err != hipSuccess) {
                    block_out.printf("[AUTOTUNE] GPU %d:   %.1fx: Alloc failed (%.1f MB), stopping probe\n",
                                     device_id_, mult_display, probe_mem / (1024.0 * 1024.0));
                    break;
                }

                // Allocate auxiliary buffers
                hipMalloc(&test_input, config_.template_size);
                hipMalloc(&test_outputs, probe_batch * config_.hash_size);
                hipMalloc(&test_target, 32);
                hipMalloc(&test_solutions, 8 + 1024 * 40 + 16);

                hipMemset(test_input, 0, config_.template_size);
                hipMemset(test_target, 0xFF, 32);

                int test_num_blocks = probe_batch / test_block_size;

                // Warmup run
                if (tnn_log_enabled(TnnLogLevel::Trace)) {
                    fprintf(stderr, "[TRACE] GPU %d: Warmup: strategy=%u, batch=%u, block=%d, blocks=%d\n",
                            device_id_, test_strategy, probe_batch, test_block_size, test_num_blocks);
                    fflush(stderr);
                }

                auto warmup = run_timed_kernel_test(
                    probe_batch, test_block_size, test_num_blocks,
                    test_input, test_outputs, test_scratch, test_target, test_solutions,
                    tune_stream, timeout_ms * 2, test_strategy
                );

                if (tnn_log_enabled(TnnLogLevel::Trace)) {
                    fprintf(stderr, "[TRACE] GPU %d: Warmup: valid=%d, time=%.2fms\n",
                            device_id_, warmup.valid, warmup.time_ms);
                    fflush(stderr);
                }

                if (!warmup.valid) {
                    block_out.printf("[AUTOTUNE] GPU %d:   %.1fx (batch=%6u): WARMUP FAILED\n",
                                     device_id_, mult_display, probe_batch);
                    hipFree(test_scratch);
                    hipFree(test_input);
                    hipFree(test_outputs);
                    hipFree(test_target);
                    hipFree(test_solutions);
                    break;
                }

                // Benchmark runs
                double total_time = 0;
                int valid_runs = 0;
                bool any_timeout = false;

                for (int iter = 0; iter < config_.autotune_iterations; iter++) {
                    if (tnn_log_enabled(TnnLogLevel::Trace)) {
                        fprintf(stderr, "[TRACE] GPU %d: Bench %d/%d: strategy=%u, batch=%u, block=%d\n",
                                device_id_, iter + 1, config_.autotune_iterations, test_strategy, probe_batch, test_block_size);
                        fflush(stderr);
                    }

                    auto result = run_timed_kernel_test(
                        probe_batch, test_block_size, test_num_blocks,
                        test_input, test_outputs, test_scratch, test_target, test_solutions,
                        tune_stream, timeout_ms, test_strategy
                    );

                    if (tnn_log_enabled(TnnLogLevel::Trace)) {
                        fprintf(stderr, "[TRACE] GPU %d: Bench %d: valid=%d, time=%.2fms\n",
                                device_id_, iter + 1, result.valid, result.time_ms);
                        fflush(stderr);
                    }

                    if (!result.valid) {
                        fprintf(stderr, "[AUTOTUNE] GPU %d: Benchmark iter %d FAILED (strategy=%u, batch=%u, block=%d)\n",
                                device_id_, iter + 1, test_strategy, probe_batch, test_block_size);
                        fflush(stderr);
                        break;
                    }

                    total_time += result.time_ms;
                    valid_runs++;

                    if (result.timed_out) {
                        any_timeout = true;
                    }

                    if (result.time_ms > config_.max_batch_time_ms * 1.2) {
                        break;
                    }
                }

                // Cleanup
                hipFree(test_scratch);
                hipFree(test_input);
                hipFree(test_outputs);
                hipFree(test_target);
                hipFree(test_solutions);

                if (valid_runs == 0) {
                    block_out.printf("[AUTOTUNE] GPU %d:   %.1fx (batch=%6u): NO VALID RUNS\n",
                                     device_id_, mult_display, probe_batch);
                    break;
                }

                double avg_time = total_time / valid_runs;
                double hashrate = (probe_batch * 1000.0) / avg_time;

                const char* status;
                bool is_acceptable = false;

                if (any_timeout || avg_time > config_.max_batch_time_ms) {
                    status = "SLOW";
                } else if (avg_time > config_.target_batch_time_ms) {
                    status = "OK+";
                    is_acceptable = true;
                } else if (avg_time < config_.min_batch_time_ms) {
                    status = "FAST";
                    is_acceptable = true;
                } else {
                    status = "OK";
                    is_acceptable = true;
                }

                const char* clamp_marker = clamped_to_max ? " (max mem)" : "";

                block_out.printf("[AUTOTUNE] GPU %d:   %.1fx (batch=%6u): %7.1fms %10.1f H/s [%s]%s\n",
                       device_id_, mult_display, probe_batch, avg_time, hashrate, status, clamp_marker);

                if (is_acceptable && hashrate > best.hashrate) {
                    best.block_size = test_block_size;
                    best.num_blocks = test_num_blocks;
                    best.batch_size = probe_batch;
                    best.hashrate = hashrate;
                    best.batch_time_ms = avg_time;
                    best.strategy = test_strategy;
                    best.valid = true;
                }

                successful_batch = probe_batch;

                // Stop if too slow - don't test higher multipliers
                if (any_timeout || avg_time > config_.max_batch_time_ms) {
                    block_out.printf("[AUTOTUNE] GPU %d:   Batch time too slow, stopping probe for this block size\n", device_id_);
                    break;
                }

                // If clamped to max memory, we've tested the max for this block size
                if (clamped_to_max) {
                    block_out.printf("[AUTOTUNE] GPU %d:   Reached memory limit for this block size\n", device_id_);
                    break;
                }

                half_step++;
            }

            block_out.newline();
        }

        } // end strategy loop

        hipStreamDestroy(tune_stream);
        
        // Results block
        bool tune_success = false;
        {
            TuneOutputBuffer out(device_id_);
            
            if (!best.valid) {
                out.printf("[AUTOTUNE] GPU %d: No valid configuration found, using static fallback\n",
                           device_id_);
                out.flush();
                
                TuneCoordinator::instance().end_tune(tune_key, false);
                return calculate_static_batch();
            }
            
            block_size_ = best.block_size;
            num_blocks_ = best.num_blocks;
            batch_size_ = best.batch_size;
            tuning_result_ = best;
            tune_success = true;
            
            out.separator('=');
            out.printf("[AUTOTUNE] GPU %d: BEST: %s\n", device_id_, best.describe().c_str());
            out.separator('=');
            out.newline();
        }
        
        // Save to disk
        save_tune_cache();
        
        // Notify coordinator that we're done
        TuneCoordinator::instance().end_tune(tune_key, tune_success);
        
        return true;
    }

    // ========================================================================
    // Member Variables
    // ========================================================================
    
    AlgoConfig config_;
    bool initialized_ = false;
    int device_id_ = 0;
    hipDeviceProp_t device_props_{};

    KernelMap kernels_;
    hipModule_t module_ = nullptr;

    uint8_t *d_input_ = nullptr;
    uint8_t *d_outputs_ = nullptr;
    uint64_t *d_scratch_ = nullptr;
    uint64_t *d_difficulty_target_ = nullptr;
    uint64_t *d_solutions_ = nullptr;

    // Host-side copy of work template (for solution verification)
    std::vector<uint8_t> h_work_template_;

    hipEvent_t start_event_ = nullptr;
    hipEvent_t stop_event_ = nullptr;

    int compute_units_ = 0;
    int block_size_ = 64;
    int num_blocks_ = 0;
    uint32_t batch_size_ = 0;
    double last_hashrate_ = 0;
    
    TuningResult tuning_result_;
};