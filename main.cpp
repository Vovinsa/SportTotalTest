#include "src/TRTInference.h"

int main() {
    std::string onnx_path = "./resnet50.onnx";
    std::string e_path = "./trt.engine";
    TRTInference trt(e_path, onnx_path);

    cv::Mat img = cv::imread("./California Alligator Lizard.jpg");

    std::vector<std::pair<std::string, float>> res = trt.run(img);
    for (auto & re : res) {
        std::cout << "class: " << re.first << " | confidence: " << re.second << std::endl;
    }
}