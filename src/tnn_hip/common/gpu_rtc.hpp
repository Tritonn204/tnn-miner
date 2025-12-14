#pragma once
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

class RTCCompiler {
public:
    struct CompiledKernel {
        hipModule_t module = nullptr;
        hipFunction_t function = nullptr;
        std::string kernel_name;
    };

    static RTCCompiler& instance() {
        static RTCCompiler inst;
        return inst;
    }

    // ---------------------------------------------------------------------
    // Header registration helpers
    // ---------------------------------------------------------------------

    void add_header_file(const std::string& header_path) {
        add_header_file_internal("", header_path);
    }

    void add_header_file(const std::string& include_name,
                         const std::string& header_path) {
        add_header_file_internal(include_name, header_path);
    }

    void add_header_source(const std::string& include_name,
                           const std::string& header_source) {
        if (include_name.empty()) {
            throw std::runtime_error("include_name cannot be empty for add_header_source");
        }

        // Idempotent: if already registered, do nothing.
        for (const auto& h : headers_) {
            if (h.name == include_name) {
                return;
            }
        }

        Header h;
        h.name   = include_name;
        h.source = header_source;
        headers_.push_back(std::move(h));
    }

    // ---------------------------------------------------------------------
    // Compile
    // ---------------------------------------------------------------------

    // Compile from embedded source (preferred for HIPRTC)
    CompiledKernel compile_from_source(
        const std::string& source,
        const std::string& source_name,
        const std::string& kernel_name,
        const std::vector<std::string>& extra_options = {}
    ) {
        printf("[TRACE] RTCCompiler::compile_from_source: Entry\n");
        printf("[TRACE]   source_name: %s\n", source_name.c_str());
        printf("[TRACE]   kernel_name: %s\n", kernel_name.c_str());
        printf("[TRACE]   source size: %zu\n", source.size());
        fflush(stdout);

        // Build normalized options exactly as compile_internal will.
        const auto defaults = build_default_options();
        const auto norm = normalize_options(defaults, extra_options);

        const std::string cache_key =
            make_cache_key(source_name, kernel_name, norm.sorted);

        printf("[TRACE] RTCCompiler::compile_from_source: Checking cache, key='%s'\n",
               cache_key.c_str());
        fflush(stdout);

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = cache_.find(cache_key);
            if (it != cache_.end()) {
                printf("[TRACE] RTCCompiler::compile_from_source: Found in cache\n");
                fflush(stdout);
                return it->second;
            }
        }

        printf("[TRACE] RTCCompiler::compile_from_source: Not in cache, calling compile_internal\n");
        fflush(stdout);

        return compile_internal(source, source_name, kernel_name, extra_options, cache_key);
    }

    // Compile from file path (fallback)
    CompiledKernel compile(
        const std::string& source_path,
        const std::string& kernel_name,
        const std::vector<std::string>& extra_options = {}
    ) {
        // Build normalized options exactly as compile_internal will.
        const auto defaults = build_default_options();
        const auto norm = normalize_options(defaults, extra_options);

        const std::string cache_key =
            make_cache_key(source_path, kernel_name, norm.sorted);

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = cache_.find(cache_key);
            if (it != cache_.end()) {
                return it->second;
            }
        }

        std::string source = load_text_file(source_path);
        return compile_internal(source, source_path, kernel_name, extra_options, cache_key);
    }

    // Manual override; whatever you set here is passed directly as
    //   --gpu-architecture=<arch>
    void set_gpu_arch(const std::string& arch) {
        gpu_arch_ = arch;
    }

    void clear_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto& kv : cache_) {
            if (kv.second.module) {
                hipModuleUnload(kv.second.module);
            }
        }
        cache_.clear();
    }

private:
    enum class Backend {
        AMD,
        NVIDIA,
        UNKNOWN
    };

    struct Header {
        std::string name;   // include name, e.g. "xelis_common.h"
        std::string source; // full header text
    };

    // ----------------------------
    // Option normalization helpers
    // ----------------------------

    static inline bool starts_with(const std::string& s, const char* pfx) {
        return s.rfind(pfx, 0) == 0;
    }

    // Return a "dedupe key" for keyed options, else empty for plain options.
    // Keyed options are last-one-wins.
    static std::string option_key(const std::string& opt) {
        if (starts_with(opt, "--gpu-architecture=")) return "--gpu-architecture";
        if (starts_with(opt, "--maxrregcount="))     return "--maxrregcount";
        if (starts_with(opt, "-std="))               return "-std";
        if (starts_with(opt, "--std="))              return "--std";
        if (starts_with(opt, "-O"))                  return "-O";

        // Macro defines: last-one-wins per macro name
        // -DFOO / -DFOO=1 / -DFOO=bar  -> key "-D:FOO"
        if (starts_with(opt, "-D")) {
            std::string rest = opt.substr(2);
            size_t eq = rest.find('=');
            std::string name = (eq == std::string::npos) ? rest : rest.substr(0, eq);
            if (!name.empty()) return std::string("-D:") + name;
        }

        return {};
    }

    struct NormalizedOptions {
        std::vector<std::string> ordered; // deterministic compile order
        std::vector<std::string> sorted;  // stable sorted representation for cache key
    };

    static NormalizedOptions normalize_options(
        const std::vector<std::string>& defaults,
        const std::vector<std::string>& extras
    ) {
        std::unordered_map<std::string, std::string> keyed; // last-one-wins
        std::unordered_set<std::string> plain_set;          // exact dedupe

        auto ingest = [&](const std::vector<std::string>& src) {
            for (const auto& o : src) {
                if (o.empty()) continue;
                std::string k = option_key(o);
                if (!k.empty()) {
                    keyed[k] = o;
                } else {
                    plain_set.insert(o);
                }
            }
        };

        // extras override defaults
        ingest(defaults);
        ingest(extras);

        std::vector<std::string> out;
        out.reserve(keyed.size() + plain_set.size());

        auto emit_key = [&](const char* k) {
            auto it = keyed.find(k);
            if (it != keyed.end()) {
                out.push_back(it->second);
                keyed.erase(it);
            }
        };

        // Preferred deterministic ordering for common knobs
        emit_key("-std");
        emit_key("--std");
        emit_key("-O");
        emit_key("--gpu-architecture");
        emit_key("--maxrregcount");

        // Remaining keyed options sorted by key name
        {
            std::vector<std::pair<std::string, std::string>> rest;
            rest.reserve(keyed.size());
            for (auto& kv : keyed) rest.push_back(kv);

            std::sort(rest.begin(), rest.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            for (auto& kv : rest) out.push_back(kv.second);
        }

        // Plain options sorted by option string
        {
            std::vector<std::string> plain(plain_set.begin(), plain_set.end());
            std::sort(plain.begin(), plain.end());
            for (auto& p : plain) out.push_back(p);
        }

        // Cache-key form: fully sorted by string so cache never depends on caller ordering
        std::vector<std::string> sorted_key = out;
        std::sort(sorted_key.begin(), sorted_key.end());

        return { out, sorted_key };
    }

    // ----------------------------
    // Construction / helpers
    // ----------------------------

    RTCCompiler() {
        hipDeviceProp_t props{};
        hipError_t err = hipGetDeviceProperties(&props, 0);
        if (err != hipSuccess) {
            backend_ = Backend::UNKNOWN;
            gpu_arch_.clear();
            printf("[TRACE] RTCCompiler: Failed to get device properties\n");
            fflush(stdout);
            return;
        }

#ifdef __HIP_PLATFORM_NVIDIA__
        backend_ = Backend::NVIDIA;
        if (props.major != 0 || props.minor != 0) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "sm_%d%d", props.major, props.minor);
            gpu_arch_ = buf; // e.g. "sm_86"
            printf("[TRACE] RTCCompiler: Detected NVIDIA GPU, arch=%s\n", gpu_arch_.c_str());
            fflush(stdout);
        }
#else
        backend_  = Backend::AMD;
        gpu_arch_ = props.gcnArchName; // e.g. "gfx1100"
        printf("[TRACE] RTCCompiler: Detected AMD GPU, arch=%s\n", gpu_arch_.c_str());
        fflush(stdout);
#endif
    }

    ~RTCCompiler() {
        clear_cache();
    }

    static std::string load_text_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }
        std::ostringstream oss;
        oss << file.rdbuf();
        return oss.str();
    }

    void add_header_file_internal(const std::string& explicit_name,
                                  const std::string& header_path) {
        Header h;
        if (!explicit_name.empty()) {
            h.name = explicit_name;
        } else {
            h.name = std::filesystem::path(header_path).filename().string();
        }
        h.source = load_text_file(header_path);
        headers_.push_back(std::move(h));
    }

    std::vector<std::string> build_default_options() const {
        std::vector<std::string> d;

        // Common
#if defined(__HIP_PLATFORM_AMD__)
        d.emplace_back("-O3");
#endif
        d.emplace_back("-std=c++20");

        // Arch
        if (!gpu_arch_.empty()) {
            d.emplace_back("--gpu-architecture=" + gpu_arch_);
        }

        // Your platform macro (not HIP built-ins)
        if (backend_ == Backend::AMD) {
            d.emplace_back("-DHIP_PLATFORM_AMD");
        } else if (backend_ == Backend::NVIDIA) {
            d.emplace_back("-DHIP_PLATFORM_NVIDIA");
        }

        return d;
    }

    static std::string make_cache_key(
        const std::string& source_id,
        const std::string& kernel_name,
        const std::vector<std::string>& normalized_sorted_options
    ) {
        std::string key = source_id + ":" + kernel_name;
        for (const auto& opt : normalized_sorted_options) {
            key.push_back('+');
            key.append(opt);
        }
        return key;
    }

    // ----------------------------
    // Actual compile
    // ----------------------------

    CompiledKernel compile_internal(
        const std::string& source,
        const std::string& source_name,
        const std::string& kernel_name,
        const std::vector<std::string>& extra_options,
        const std::string& cache_key
    ) {
        printf("[TRACE] RTCCompiler::compile_internal: Entry\n");
        printf("[TRACE]   source_name: %s\n", source_name.c_str());
        printf("[TRACE]   kernel_name: %s\n", kernel_name.c_str());
        printf("[TRACE]   source size: %zu\n", source.size());
        printf("[TRACE]   headers: %zu\n", headers_.size());
        printf("[TRACE]   cache_key: %s\n", cache_key.c_str());
        fflush(stdout);

        // Build header arrays for hiprtcCreateProgram
        std::vector<const char*> header_sources;
        std::vector<const char*> header_names;
        header_sources.reserve(headers_.size());
        header_names.reserve(headers_.size());

        for (const auto& h : headers_) {
            if (h.name == source_name) {
                printf("[WARNING] Skipping header '%s' - same as main source\n", h.name.c_str());
                continue;
            }
            printf("[TRACE]   Header: %s (size=%zu)\n", h.name.c_str(), h.source.size());
            header_sources.push_back(h.source.c_str());
            header_names.push_back(h.name.c_str());
        }

        printf("[TRACE] RTCCompiler::compile_internal: Calling hiprtcCreateProgram\n");
        printf("[TRACE]   Passing %d headers to HIPRTC\n", (int)header_sources.size());
        fflush(stdout);

        hiprtcProgram prog{};
        hiprtcResult rc{};

#ifdef _WIN32
        __try {
            printf("[TRACE] About to call hiprtcCreateProgram...\n");
            fflush(stdout);

            rc = hiprtcCreateProgram(
                &prog,
                source.c_str(),
                source_name.c_str(),
                static_cast<int>(header_sources.size()),
                header_sources.empty() ? nullptr : header_sources.data(),
                header_names.empty() ? nullptr : header_names.data()
            );

            printf("[TRACE] hiprtcCreateProgram call returned successfully\n");
            fflush(stdout);
        }
        __except(EXCEPTION_EXECUTE_HANDLER) {
            printf("[ERROR] RTCCompiler: hiprtcCreateProgram crashed! Exception code: 0x%08X\n", GetExceptionCode());
            fflush(stdout);
            throw std::runtime_error("hiprtcCreateProgram crashed with SEH exception");
        }
#else
        rc = hiprtcCreateProgram(
            &prog,
            source.c_str(),
            source_name.c_str(),
            static_cast<int>(header_sources.size()),
            header_sources.empty() ? nullptr : header_sources.data(),
            header_names.empty() ? nullptr : header_names.data()
        );
#endif

        printf("[TRACE] RTCCompiler::compile_internal: hiprtcCreateProgram returned %d\n", rc);
        fflush(stdout);

        if (rc != HIPRTC_SUCCESS) {
            throw std::runtime_error("Failed to create HIPRTC program");
        }

        printf("[TRACE] RTCCompiler::compile_internal: HIPRTC program created, preparing compilation\n");
        fflush(stdout);

        // Normalize options (dedupe + deterministic ordering)
        const auto defaults = build_default_options();
        const auto norm = normalize_options(defaults, extra_options);

        // Convert to const char* (lifetime from std::string storage in norm.ordered)
        std::vector<const char*> options;
        options.reserve(norm.ordered.size());
        for (auto& s : norm.ordered) {
            options.push_back(s.c_str());
        }

        printf("[TRACE] RTCCompiler::compile_internal: Calling hiprtcCompileProgram with %d options\n",
               (int)options.size());
        for (size_t i = 0; i < options.size(); i++) {
            printf("[TRACE]   Option %zu: %s\n", i, options[i]);
        }
        fflush(stdout);

#ifdef _WIN32
        __try {
            rc = hiprtcCompileProgram(
                prog,
                static_cast<int>(options.size()),
                options.empty() ? nullptr : options.data()
            );
        }
        __except(EXCEPTION_EXECUTE_HANDLER) {
            printf("[ERROR] RTCCompiler: hiprtcCompileProgram crashed! Exception code: 0x%08X\n", GetExceptionCode());
            fflush(stdout);
            hiprtcDestroyProgram(&prog);
            throw std::runtime_error("hiprtcCompileProgram crashed with SEH exception");
        }
#else
        rc = hiprtcCompileProgram(
            prog,
            static_cast<int>(options.size()),
            options.empty() ? nullptr : options.data()
        );
#endif

        printf("[TRACE] RTCCompiler::compile_internal: hiprtcCompileProgram returned %d\n", rc);
        fflush(stdout);

        if (rc != HIPRTC_SUCCESS) {
            size_t log_size = 0;
            hiprtcGetProgramLogSize(prog, &log_size);
            std::vector<char> log(log_size ? log_size : 1, '\0');
            if (log_size > 0) {
                hiprtcGetProgramLog(prog, log.data());
            }
            hiprtcDestroyProgram(&prog);
            throw std::runtime_error(
                "HIPRTC compilation failed for " + source_name + ":\n" +
                std::string(log.data())
            );
        }

        size_t code_size = 0;
        hiprtcGetCodeSize(prog, &code_size);
        std::vector<char> code(code_size);
        hiprtcGetCode(prog, code.data());

        hiprtcDestroyProgram(&prog);

        CompiledKernel kernel;
        kernel.kernel_name = kernel_name;

        printf("[TRACE] About to load module (%zu bytes code)\n", code_size);

        size_t free_mem = 0, total_mem = 0;
        hipMemGetInfo(&free_mem, &total_mem);
        printf("[TRACE] GPU memory: %zu MB free / %zu MB total\n",
               free_mem / (1024 * 1024), total_mem / (1024 * 1024));
        fflush(stdout);

        hipError_t err = hipModuleLoadData(&kernel.module, code.data());
        if (err != hipSuccess) {
            printf("[ERROR] hipModuleLoadData failed: %d (%s)\n", err, hipGetErrorString(err));
            hipMemGetInfo(&free_mem, &total_mem);
            printf("[ERROR] GPU memory after failure: %zu MB free / %zu MB total\n",
                   free_mem / (1024 * 1024), total_mem / (1024 * 1024));
            fflush(stdout);
            throw std::runtime_error("hipModuleLoadData failed");
        }

        err = hipModuleGetFunction(&kernel.function, kernel.module, kernel.kernel_name.c_str());
        if (err != hipSuccess) {
            hipModuleUnload(kernel.module);
            throw std::runtime_error("Failed to get kernel function: " + kernel_name);
        }

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache_[cache_key] = kernel;
        }

        return kernel;
    }

private:
    Backend backend_ = Backend::UNKNOWN;
    std::string gpu_arch_;

    mutable std::mutex cache_mutex_;
    std::unordered_map<std::string, CompiledKernel> cache_;

    std::vector<Header> headers_;
};

extern "C" bool precompile_all_kernels();
