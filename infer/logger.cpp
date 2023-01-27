#include "logger.h"

void Logger::log(Severity severity, const char* msg) noexcept {
    if ((severity == Severity::kWARNING) || (severity == Severity::kINTERNAL_ERROR)) {
        std::cout << msg << std::endl;
    }
}