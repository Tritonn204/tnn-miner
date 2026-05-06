#pragma once
#include "cpu_algo.hpp"
#include <map>
#include <functional>
#include <memory>
#include <string>

// Factory function type for creating algorithm instances
using CPUAlgoFactory = std::function<std::unique_ptr<ICPUAlgorithm>()>;

// Singleton registry for CPU mining algorithms
class CPUAlgoRegistry {
public:
    static CPUAlgoRegistry& instance() {
        static CPUAlgoRegistry registry;
        return registry;
    }

    // Register an algorithm with a factory function
    void register_algorithm(const std::string& name, CPUAlgoFactory factory) {
        factories_[name] = factory;
    }

    // Create an algorithm instance by name
    std::unique_ptr<ICPUAlgorithm> create(const std::string& name) {
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }

    // Check if an algorithm is registered
    bool has_algorithm(const std::string& name) const {
        return factories_.find(name) != factories_.end();
    }

private:
    CPUAlgoRegistry() = default;
    std::map<std::string, CPUAlgoFactory> factories_;
};

// Helper macro for registering algorithms at startup
#define REGISTER_CPU_ALGORITHM(name, class_name) \
    namespace { \
        struct class_name##_Registrar { \
            class_name##_Registrar() { \
                CPUAlgoRegistry::instance().register_algorithm(name, []() { \
                    return std::make_unique<class_name>(); \
                }); \
            } \
        }; \
        static class_name##_Registrar class_name##_registrar_instance; \
    }
