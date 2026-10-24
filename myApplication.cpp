#include <opencv2/opencv.hpp>

#include <print>

void MyApplication() {
    std::println("OpenCV {}", CV_VERSION);

    cv::Mat image(240, 480, CV_8UC3, cv::Scalar(40, 120, 40));
    cv::circle(image, {120, 120}, 60, cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(image, "Hello, OpenCV", {200, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);

    cv::Mat grey;
    cv::Mat edges;
    cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    cv::Canny(grey, edges, 50, 150);

    cv::imshow("Image", image);
    cv::imshow("Edges", edges);
    cv::waitKey(0);
}
