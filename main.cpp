#include "opencv2/opencv.hpp"
#include "microhttpd.h"
#include "iostream"
#include "infer/TRTInference.h"

TRTInference trt("/SportTotalTest/trt.engine",
                 "/SportTotalTest/resnet34.onnx");
std::string image_data;

int postRequestCallback(void *cls, MHD_Connection *connection,
                          const char *url, const char *method,
                          const char *version, const char *upload_data,
                          size_t *upload_data_size, void **ptr) {
    if (*ptr == nullptr) {
        image_data.clear();
        *ptr = (void*)1;
        return MHD_YES;
    }
    if (*upload_data_size != 0) {
        image_data.append(upload_data, *upload_data_size);
        *upload_data_size = 0;
        return MHD_YES;
    }
    try {
        cv::Mat image = cv::imdecode(cv::Mat(1, image_data.size(), CV_8UC1, (void*)image_data.data()), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cout << "Failed to read image file" << std::endl;
            return MHD_NO;
        }
        std::vector<std::pair<std::string, float>> res = trt.run(image);

        std::string response;

        for (auto & re : res) {
            response = re.first + " " + std::to_string(re.second);
        }
        MHD_Response *mhd_response = MHD_create_response_from_buffer(response.size(), (void*)response.c_str(), MHD_RESPMEM_MUST_COPY);
        MHD_queue_response(connection, MHD_HTTP_OK, mhd_response);
        MHD_destroy_response(mhd_response);
        return MHD_YES;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return MHD_NO;
    }
}

int main() {
    MHD_Daemon *daemon = MHD_start_daemon(MHD_USE_THREAD_PER_CONNECTION, 3000, nullptr, nullptr, postRequestCallback, nullptr, MHD_OPTION_END);
    if (daemon == nullptr) {
        std::cerr << "Failed to start the server" << std::endl;
        return 1;
    }
    std::cout << "Server started on port 3000, press enter to stop" << std::endl;
    std::cin.get();
    MHD_stop_daemon(daemon);
    return 0;
}

