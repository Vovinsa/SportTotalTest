#pragma once

#include "vector"
#include "iostream"
#include "fstream"
#include "numeric"

#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core.hpp"

#include "logger.h"

class TRTInference {
public:
    TRTInference(std::string engine_path, std::string onnx_model_path);

    void buildEngine(std::string &onnx_model_path, std::string &engine_path);
    nvinfer1::ICudaEngine* loadEngine(std::string &engine_path);
    std::vector<std::pair<std::string, float>> run(cv::Mat &input);
    size_t getSizeByDim(const nvinfer1::Dims& dims);
    void preprocessImage(cv::Mat &image, float* gpu_input, const nvinfer1::Dims& dims);
    std::vector<std::string> getClassNames(const std::string& imagenet_classes);
    std::vector<std::pair<std::string, float>> postprocessResults(float *gpu_output, const nvinfer1::Dims &dims);

private:
    nvinfer1::ICudaEngine *_engine;
    nvinfer1::IExecutionContext *_context;
    Logger _logger;
};