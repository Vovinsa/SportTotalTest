#pragma once

#include "iostream"

#include "NvInfer.h"

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};
