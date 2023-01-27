#include "TRTInference.h"

TRTInference::TRTInference(std::string engine_path, std::string onnx_model_path) {
    if (!std::ifstream(engine_path)) {
        buildEngine(onnx_model_path, engine_path);
    }
    _engine = loadEngine(engine_path);
    _context = _engine->createExecutionContext();
}

void TRTInference::buildEngine(std::string &onnx_model_path, std::string &engine_path) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(_logger);
    builder->setMaxBatchSize(1);

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, _logger);
    if (!(parser->parseFromFile(onnx_model_path.c_str(), static_cast<uint32_t>(nvinfer1::ILogger::Severity::kWARNING)))) {
        for (int i = 0; i< parser->getNbErrors(); i++) {
            std::cout << parser->getError(i)->desc() << std::endl;
        }
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);

    std::ofstream plan_file(engine_path, std::ios::binary | std::ios::out);
    plan_file.write(reinterpret_cast<char*>(plan->data()), plan->size());
}

nvinfer1::ICudaEngine* TRTInference::loadEngine(std::string &engine_path) {
    std::ifstream engine_f(engine_path, std::ios::binary | std::ios::in);
    std::vector<char> engine_data(std::istreambuf_iterator<char>{engine_f}, {});
    engine_f.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(_logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    return engine;
}

std::vector<std::pair<std::string, float>> TRTInference::run(cv::Mat &input) {
    std::vector<nvinfer1::Dims> output_dims;
    std::vector<void*> buffers(_engine->getNbBindings());
    for (int i = 0; i < _engine->getNbBindings(); ++i) {
        auto binding_size = getSizeByDim(_engine->getBindingDimensions(i)) * 1 * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (!_engine->bindingIsInput(i)) {
            output_dims.emplace_back(_engine->getBindingDimensions(i));
        }
    }
    preprocessImage(input, (float*)buffers[0]);
    _context->enqueueV2(buffers.data(), 0, nullptr);
    std::vector<std::pair<std::string, float>> results = postprocessResults((float*)buffers[1], output_dims[0]);
    for (void* buf : buffers) {
        cudaFree(buf);
    }
    return results;
}

void TRTInference::preprocessImage(cv::Mat &image, float* gpu_input) {
    cv::Size input_size = cv::Size(_input_height, _input_width);

    cv::cuda::GpuMat gpu_image;
    gpu_image.upload(image);

    cv::cuda::resize(gpu_image, gpu_image, input_size, 0, 0, cv::INTER_NEAREST);
    gpu_image.convertTo(gpu_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(gpu_image, cv::Scalar(0.485f, 0.456f, 0.406f), gpu_image, cv::noArray(), -1);
    cv::cuda::divide(gpu_image, cv::Scalar(0.229f, 0.224f, 0.225f), gpu_image, 1, -1);
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < 3; ++i) {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * _input_width * _input_height));
    }
    cv::cuda::split(gpu_image, chw);
}

size_t TRTInference::getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

std::vector<std::string> TRTInference::getClassNames(const std::string& imagenet_classes) {
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    std::string class_name;
    while (std::getline(classes_file, class_name)) {
        classes.push_back(class_name);
    }
    return classes;
}

std::vector<std::pair<std::string, float>> TRTInference::postprocessResults(float *gpu_output, const nvinfer1::Dims &dims) {
    auto classes = getClassNames("./classes.txt");

    std::vector<float> cpu_output(getSizeByDim(dims) * 1);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);

    std::vector<int> indices(getSizeByDim(dims) * 1);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    std::vector<std::pair<std::string, float>> output;
    output.emplace_back(std::make_pair(classes[indices[0]], cpu_output[indices[0]] / sum));
    return output;
}