#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <thread>
#include <atomic>
#include <map>
#include <functional>
#include <memory>
#include <signal.h>
#include <mutex>

#include "terminal.hpp"

#ifdef __x86_64__
#include <cpuid.h>

#ifdef _WIN32
#include <Windows.h>
#include <winioctl.h>

// Direct-to-driver WinRing0 access (no DLL needed, only WinRing0x64.sys).
// Based on the XMRig approach: register .sys as a kernel service, talk via DeviceIoControl.

#define WR0_SERVICE_NAME    L"WinRing0_1_2_0"
#define WR0_IOCTL_READ_MSR  CTL_CODE(40000, 0x821, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define WR0_IOCTL_WRITE_MSR CTL_CODE(40000, 0x822, METHOD_BUFFERED, FILE_ANY_ACCESS)

class WinRing0Direct {
private:
    HANDLE driver       = INVALID_HANDLE_VALUE;
    SC_HANDLE manager   = nullptr;
    SC_HANDLE service   = nullptr;
    bool reuse          = false;

public:
    WinRing0Direct() = default;

    bool init() {
        manager = OpenSCManager(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
        if (!manager) {
            DWORD err = GetLastError();
            if (err == ERROR_ACCESS_DENIED) {
                fprintf(stderr, "MSR: administrator privileges required\n");
            } else {
                fprintf(stderr, "MSR: failed to open service control manager, error %lu\n", err);
            }
            return false;
        }

        // Resolve path to WinRing0x64.sys next to our exe
        std::vector<wchar_t> dir;
        DWORD err;
        do {
            dir.resize(dir.empty() ? MAX_PATH : dir.size() * 2);
            GetModuleFileNameW(nullptr, dir.data(), (DWORD)dir.size());
            err = GetLastError();
        } while (err == ERROR_INSUFFICIENT_BUFFER);

        for (auto it = dir.end() - 1; it != dir.begin(); --it) {
            if (*it == L'\\' || *it == L'/') {
                *(it + 1) = L'\0';
                break;
            }
        }
        std::wstring sysPath = std::wstring(dir.data()) + L"WinRing0x64.sys";

        // Check if service already exists
        service = OpenServiceW(manager, WR0_SERVICE_NAME, SERVICE_ALL_ACCESS);
        if (service) {
            SERVICE_STATUS status;
            if (QueryServiceStatus(service, &status) && status.dwCurrentState == SERVICE_RUNNING) {
                reuse = true;
            } else {
                // Service exists but not running — remove and recreate
                uninstall();
            }
        }

        // Try opening driver directly (may already be running under different service name)
        driver = CreateFileW(L"\\\\.\\WinRing0_1_2_0", GENERIC_READ | GENERIC_WRITE,
                            0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (driver != INVALID_HANDLE_VALUE) {
            reuse = true;
            return true;
        }

        // Install and start the service
        if (!reuse) {
            service = CreateServiceW(manager, WR0_SERVICE_NAME, WR0_SERVICE_NAME,
                                    SERVICE_ALL_ACCESS, SERVICE_KERNEL_DRIVER,
                                    SERVICE_DEMAND_START, SERVICE_ERROR_NORMAL,
                                    sysPath.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
            if (!service) {
                fprintf(stderr, "MSR: failed to install WinRing0 driver, error %lu\n", GetLastError());
                return false;
            }

            if (!StartService(service, 0, nullptr)) {
                err = GetLastError();
                if (err != ERROR_SERVICE_ALREADY_RUNNING) {
                    if (err == ERROR_FILE_NOT_FOUND) {
                        fprintf(stderr, "MSR: WinRing0x64.sys not found next to executable\n");
                    } else {
                        fprintf(stderr, "MSR: failed to start WinRing0 driver, error %lu\n", err);
                    }
                    uninstall();
                    return false;
                }
            }
        }

        driver = CreateFileW(L"\\\\.\\WinRing0_1_2_0", GENERIC_READ | GENERIC_WRITE,
                            0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (driver == INVALID_HANDLE_VALUE) {
            fprintf(stderr, "MSR: failed to connect to WinRing0 driver, error %lu\n", GetLastError());
            uninstall();
            return false;
        }
        return true;
    }

    void uninstall() {
        if (driver != INVALID_HANDLE_VALUE) {
            CloseHandle(driver);
            driver = INVALID_HANDLE_VALUE;
        }
        if (service && !reuse) {
            SERVICE_STATUS ss;
            ControlService(service, SERVICE_CONTROL_STOP, &ss);
            DeleteService(service);
        }
        if (service) {
            CloseServiceHandle(service);
            service = nullptr;
        }
        if (manager) {
            CloseServiceHandle(manager);
            manager = nullptr;
        }
    }

    ~WinRing0Direct() {
        uninstall();
    }

    bool isAvailable() const { return driver != INVALID_HANDLE_VALUE; }

    bool rdmsr(uint32_t reg, uint64_t &value) const {
        DWORD size = 0;
        return DeviceIoControl(driver, WR0_IOCTL_READ_MSR,
                              &reg, sizeof(reg), &value, sizeof(value), &size, nullptr) != 0;
    }

    bool wrmsr(uint32_t reg, uint64_t value) {
        struct { uint32_t reg; uint32_t val[2]; } input;
        static_assert(sizeof(input) == 12, "WinRing0 IOCTL struct size");
        input.reg = reg;
        *reinterpret_cast<uint64_t*>(input.val) = value;
        DWORD output, k;
        return DeviceIoControl(driver, WR0_IOCTL_WRITE_MSR,
                              &input, sizeof(input), &output, sizeof(output), &k, nullptr) != 0;
    }
};

static WinRing0Direct g_winRing0;

inline bool initMSRAccess() {
    return g_winRing0.init();
}

inline bool readMSR(uint32_t reg, uint64_t& value, int core = 0) {
    (void)core;  // direct IOCTL operates on current core
    if (!g_winRing0.isAvailable()) return false;
    return g_winRing0.rdmsr(reg, value);
}

inline bool writeMSR(uint32_t reg, uint64_t value, int core = 0) {
    (void)core;
    if (!g_winRing0.isAvailable()) return false;
    return g_winRing0.wrmsr(reg, value);
}

#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#endif
#endif

// CPU detection (architecture-protected)
enum class CPUType {
    UNSUPPORTED,
    INTEL,
    AMD_ZEN1_ZEN2,
    AMD_ZEN3,
    AMD_ZEN4_ZEN5
};

// Base MSR optimization profile class
class MSROptimizationProfile {
public:
    virtual ~MSROptimizationProfile() = default;
    
    virtual std::string getName() const = 0;
    virtual std::vector<std::pair<uint32_t, uint64_t>> getMSRValues(CPUType cpuType) const = 0;
};

// RandomX-specific profile implementation
class RandomXOptimizationProfile : public MSROptimizationProfile {
public:
    std::string getName() const override {
        return "RandomX";
    }
    
    std::vector<std::pair<uint32_t, uint64_t>> getMSRValues(CPUType cpuType) const override {
        std::vector<std::pair<uint32_t, uint64_t>> msrValues;
        
        switch (cpuType) {
            case CPUType::INTEL:
                msrValues.emplace_back(0x1A4, 0xF);
                break;
                
            case CPUType::AMD_ZEN1_ZEN2:
                msrValues.emplace_back(0xC0011020, 0x0);
                msrValues.emplace_back(0xC0011021, 0x40);
                msrValues.emplace_back(0xC0011022, 0x1510000);
                msrValues.emplace_back(0xC001102B, 0x2000CC16);
                break;
                
            case CPUType::AMD_ZEN3:
                msrValues.emplace_back(0xC0011020, 0x4480000000000);
                msrValues.emplace_back(0xC0011021, 0x1C000200000040);
                msrValues.emplace_back(0xC0011022, 0xC000000401570000);
                msrValues.emplace_back(0xC001102B, 0x2000CC10);
                break;
                
            case CPUType::AMD_ZEN4_ZEN5:
                msrValues.emplace_back(0xC0011020, 0x4400000000000);
                msrValues.emplace_back(0xC0011021, 0x4000000000040);
                msrValues.emplace_back(0xC0011022, 0x8680000401570000);
                msrValues.emplace_back(0xC001102B, 0x2040CC10);
                break;
                
            default:
                break;
        }
        
        return msrValues;
    }
};

// Example: Ethash optimization profile (placeholder values - replace with actual research)
class EthashOptimizationProfile : public MSROptimizationProfile {
public:
    std::string getName() const override {
        return "Ethash";
    }
    
    std::vector<std::pair<uint32_t, uint64_t>> getMSRValues(CPUType cpuType) const override {
        std::vector<std::pair<uint32_t, uint64_t>> msrValues;
        
        // Example values - would need real research for actual Ethash optimizations
        if (cpuType == CPUType::INTEL) {
            msrValues.emplace_back(0x1A4, 0x3); // Different from RandomX
        }
        
        return msrValues;
    }
};

// XelisV3-specific profile: use the Xelis-tuned MSR bundle.
class XelisV3OptimizationProfile : public MSROptimizationProfile {
public:
    std::string getName() const override {
        return "XelisV3";
    }

    std::vector<std::pair<uint32_t, uint64_t>> getMSRValues(CPUType cpuType) const override {
        std::vector<std::pair<uint32_t, uint64_t>> msrValues;

        switch (cpuType) {
            case CPUType::INTEL:
                msrValues.emplace_back(0x1A4, 0xF);
                break;
                
            case CPUType::AMD_ZEN1_ZEN2:
                msrValues.emplace_back(0xC0011022, 0x1510000);
                break;
                
            case CPUType::AMD_ZEN3:
                msrValues.emplace_back(0xC0011022, 0x01570000);
                break;
                
            case CPUType::AMD_ZEN4_ZEN5:
                msrValues.emplace_back(0xC0011022, 0x01570000);
                break;

            default:
                break;
        }

        return msrValues;
    }
};

// Main MSR Manager class
class MSRManager {
private:
    struct MSRBackup {
        uint32_t reg;
        std::vector<uint64_t> originalValues;
    };
    
    CPUType cpuType = CPUType::UNSUPPORTED;
    int numCores = 0;
    bool msrAvailable = false;
    std::atomic<bool> optimizationsApplied{false};
    std::string activeProfileName;
    std::vector<MSRBackup> backups;
    std::map<std::string, std::shared_ptr<MSROptimizationProfile>> profiles;
    std::mutex mutex;

public:
    MSRManager() {
        #ifdef __x86_64__
        detectCPU();
        numCores = std::thread::hardware_concurrency();
        msrAvailable = initMSRAccess();
        
        if (msrAvailable) {
            setcolor(BRIGHT_YELLOW);
            std::cout << "\nMSR access available. Detected: ";
            switch (cpuType) {
                case CPUType::INTEL:
                    std::cout << "Intel CPU";
                    break;
                case CPUType::AMD_ZEN1_ZEN2:
                    std::cout << "AMD Zen1/Zen2 CPU";
                    break;
                case CPUType::AMD_ZEN3:
                    std::cout << "AMD Zen3 CPU";
                    break;
                case CPUType::AMD_ZEN4_ZEN5:
                    std::cout << "AMD Zen4/Zen5 CPU";
                    break;
                default:
                    std::cout << "Unsupported CPU";
            }
            std::cout << " with " << numCores << " logical cores" << std::endl;
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        } else {
            setcolor(RED);
            std::cerr << "MSR access not available. Run as administrator/root." << std::endl;
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
        }
        
        // Register default profiles
        registerProfile(std::make_shared<RandomXOptimizationProfile>());
        registerProfile(std::make_shared<EthashOptimizationProfile>());
        registerProfile(std::make_shared<XelisV3OptimizationProfile>());
        #else
        setcolor(RED);
        std::cout << "MSR optimization not available on non-x86_64 architecture" << std::endl;
        fflush(stdout);
        setcolor(BRIGHT_WHITE);
        #endif
    }
    
    ~MSRManager() {
        restoreOriginalValues();
    }
    
    void registerProfile(std::shared_ptr<MSROptimizationProfile> profile) {
        std::lock_guard<std::mutex> lock(mutex);
        profiles[profile->getName()] = profile;
    }
    
    bool applyOptimizationProfile(const std::string& profileName) {
        #ifdef __x86_64__
        if (!msrAvailable) {
            std::cerr << "MSR access not available" << std::endl;
            return false;
        }
        
        if (optimizationsApplied.load()) {
            std::cout << "Another optimization profile is already active. Restoring first..." << std::endl;
            restoreOriginalValues();
        }
        
        std::shared_ptr<MSROptimizationProfile> profile;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = profiles.find(profileName);
            if (it == profiles.end()) {
                std::cerr << "Optimization profile '" << profileName << "' not found" << std::endl;
                return false;
            }
            profile = it->second;
        }
        
        std::vector<std::pair<uint32_t, uint64_t>> msrValues = profile->getMSRValues(cpuType);
        
        if (msrValues.empty()) {
            std::cerr << "No MSR values defined for this CPU in profile: " << profileName << std::endl;
            return false;
        }
        
        // Backup current MSR values before modifying
        backupMSRValues(msrValues);
        
        // Apply new MSR values to all cores
        bool success = true;
        for (const auto& [reg, value] : msrValues) {
            for (int core = 0; core < numCores; core++) {
                if (!writeMSR(reg, value, core)) {
                    std::cerr << "Failed to write MSR 0x" << std::hex << reg 
                              << " on core " << std::dec << core << std::endl;
                    success = false;
                }
            }
        }
        
        if (success) {
            setcolor(BRIGHT_YELLOW);
            std::cout << profileName << " MSR optimizations applied successfully" << std::endl;
            fflush(stdout);
            setcolor(BRIGHT_WHITE);
            activeProfileName = profileName;
            optimizationsApplied.store(true);
        } else {
            std::cerr << "MSR optimization partially failed - system may be in inconsistent state" << std::endl;
            restoreOriginalValues();
        }
        
        return success;
        #else
        return false;
        #endif
    }
    
    bool restoreOriginalValues() {
        #ifdef __x86_64__
        std::lock_guard<std::mutex> lock(mutex);
        
        if (!msrAvailable || !optimizationsApplied.load() || backups.empty()) 
            return false;
        
        bool success = true;
        for (const auto& backup : backups) {
            for (int core = 0; core < backup.originalValues.size(); core++) {
                if (!writeMSR(backup.reg, backup.originalValues[core], core)) {
                    std::cerr << "Failed to restore MSR 0x" << std::hex << backup.reg 
                              << " on core " << std::dec << core << std::endl;
                    success = false;
                }
            }
        }
        
        if (success) {
            std::cout << "Original MSR values restored successfully" << std::endl;
            optimizationsApplied.store(false);
            activeProfileName = "";
            backups.clear();
        }
        
        return success;
        #else
        return false;
        #endif
    }
    
    std::string getActiveProfileName() const {
        return activeProfileName;
    }
    
    bool isOptimizationActive() const {
        return optimizationsApplied.load();
    }
    
    CPUType getCPUType() const {
        return cpuType;
    }

private:
    #ifdef __x86_64__
    void detectCPU() {
        uint32_t eax, ebx, ecx, edx;
        
        // Get vendor
        if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
            char vendor[13] = {0};
            memcpy(vendor, &ebx, 4);
            memcpy(vendor + 4, &edx, 4);
            memcpy(vendor + 8, &ecx, 4);
            vendor[12] = '\0';
            
            std::string vendorStr(vendor);
            
            if (vendorStr.find("Intel") != std::string::npos) {
                cpuType = CPUType::INTEL;
            } else if (vendorStr.find("AMD") != std::string::npos) {
                // Get detailed AMD info
                if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                    int family = ((eax >> 8) & 0xF) + ((eax >> 20) & 0xFF);
                    int model = ((eax >> 4) & 0xF) | ((eax >> 12) & 0xF0);
                    
                    if (family == 25) {
                        if (model == 97) {
                            cpuType = CPUType::AMD_ZEN4_ZEN5;
                        } else {
                            cpuType = CPUType::AMD_ZEN3;
                        }
                    } else if (family == 26) {
                        cpuType = CPUType::AMD_ZEN4_ZEN5;
                    } else if (family == 23) {
                        cpuType = CPUType::AMD_ZEN1_ZEN2;
                    }
                }
            }
        }
    }
    
    bool initMSRAccess() {
        #ifdef _WIN32
        return ::initMSRAccess();
        #else
        std::ifstream msrCheck("/dev/cpu/0/msr");
        if (!msrCheck.good()) {
            // Try loading msr module if not already loaded
            system("modprobe msr allow_writes=on 2>/dev/null || true");
            system("echo on > /sys/module/msr/parameters/allow_writes 2>/dev/null || true");
            msrCheck.close();
            msrCheck.open("/dev/cpu/0/msr");
        }
        return msrCheck.good();
        #endif
    }
    
    bool readMSR(uint32_t reg, uint64_t& value, int core = 0) {
        #ifdef _WIN32
        return ::readMSR(reg, value, core);
        return false;
        #else
        char path[64];
        snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
        int fd = open(path, O_RDONLY);
        if (fd < 0) return false;
        
        bool result = pread(fd, &value, sizeof(value), reg) == sizeof(value);
        close(fd);
        return result;
        #endif
    }
    
    bool writeMSR(uint32_t reg, uint64_t value, int core = 0) {
        #ifdef _WIN32
        return ::writeMSR(reg, value, core);
        #else
        char path[64];
        snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
        int fd = open(path, O_WRONLY);
        if (fd < 0) return false;
        
        bool result = pwrite(fd, &value, sizeof(value), reg) == sizeof(value);
        close(fd);
        return result;
        #endif
    }
    
    void backupMSRValues(const std::vector<std::pair<uint32_t, uint64_t>>& msrList) {
        backups.clear();
        
        for (const auto& [reg, _] : msrList) {
            MSRBackup backup;
            backup.reg = reg;
            backup.originalValues.resize(numCores);
            
            for (int core = 0; core < numCores; core++) {
                uint64_t value = 0;
                if (readMSR(reg, value, core)) {
                    backup.originalValues[core] = value;
                } else {
                    std::cerr << "Failed to read MSR 0x" << std::hex << reg 
                              << " on core " << std::dec << core << std::endl;
                    // Use default value of 0 on read failure
                }
            }
            
            backups.push_back(backup);
        }
    }
    #endif
};

// Global singleton
class MSRManagerGlobal {
private:
    static std::unique_ptr<MSRManager> instance;
    static std::mutex instanceMutex;
    static std::atomic<bool> signalHandlersRegistered;
    
    static void signalHandler(int sig) {
        std::cout << "Caught signal " << sig << ", restoring MSR values and exiting..." << std::endl;
        if (instance) {
            if (instance->isOptimizationActive()) {
                std::string profile = instance->getActiveProfileName();
                instance->restoreOriginalValues();
                std::cout << "MSR values for profile '" << profile << "' restored" << std::endl;
            }
        }
        exit(sig);
    }
    
    static void registerSignalHandlers() {
        if (!signalHandlersRegistered.load()) {
            signal(SIGINT, signalHandler);
            signal(SIGTERM, signalHandler);
            signalHandlersRegistered.store(true);
        }
    }

public:
    static MSRManager* getInstance() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (!instance) {
            instance = std::make_unique<MSRManager>();
            registerSignalHandlers();
        }
        return instance.get();
    }
    
    static void cleanup() {
        std::lock_guard<std::mutex> lock(instanceMutex);
        if (instance && instance->isOptimizationActive()) {
            instance->restoreOriginalValues();
        }
        instance.reset();
    }
};

inline std::unique_ptr<MSRManager> MSRManagerGlobal::instance = nullptr;
inline std::mutex MSRManagerGlobal::instanceMutex;
inline std::atomic<bool> MSRManagerGlobal::signalHandlersRegistered{false};

// Simple interface for algorithm-specific optimizations.
inline bool applyMSROptimization(const std::string& algorithm) {
    MSRManager* manager = MSRManagerGlobal::getInstance();
    return manager->applyOptimizationProfile(algorithm);
}

inline bool resetMSROptimizations() {
    MSRManager* manager = MSRManagerGlobal::getInstance();
    return manager->restoreOriginalValues();
}

inline void cleanupMSROnExit() {
    MSRManagerGlobal::cleanup();
}
