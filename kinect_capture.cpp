// kinect_capture.cpp
// Capture one RGB and one depth frame from Kinect v2 using libfreenect2
// Save as PNG and .npy (NumPy) files
// Usage: ./kinect_capture <rgb_png> <depth_png> <rgb_npy> <depth_npy>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

// Helper to save a NumPy .npy file (float32 or uint8)
// Only supports 2D arrays
void save_npy(const std::string &filename, const void *data, int rows, int cols, int elem_size, const std::string &dtype) {
    std::ofstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open file for writing: " + filename);
    // Write header
    std::string header = "\x93NUMPY";
    header += char(1); // major version
    header += char(0); // minor version
    std::string dict = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " + std::to_string(cols) + "), }";
    size_t pad = 16 - ((10 + dict.size()) % 16);
    dict += std::string(pad, ' ');
    uint16_t hlen = uint16_t(dict.size());
    f.write(header.c_str(), 8);
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(dict.c_str(), dict.size());
    // Write data
    f.write(reinterpret_cast<const char*>(data), rows * cols * elem_size);
    f.close();
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <rgb_png> <depth_png> <rgb_npy> <depth_npy>\n";
        return 1;
    }
    std::string rgb_png = argv[1];
    std::string depth_png = argv[2];
    std::string rgb_npy = argv[3];
    std::string depth_npy = argv[4];

    libfreenect2::Freenect2 freenect2;
    if (freenect2.enumerateDevices() == 0) {
        std::cerr << "No Kinect v2 device connected!\n";
        return 2;
    }
    std::string serial = freenect2.getDefaultDeviceSerialNumber();
    libfreenect2::OpenGLPacketPipeline pipeline;
    libfreenect2::Freenect2Device *dev = freenect2.openDevice(serial, &pipeline);
    if (!dev) {
        std::cerr << "Failed to open Kinect v2 device!\n";
        return 3;
    }
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth);
    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    dev->start();

    // Warm up
    for (int i = 0; i < 30; ++i) {
        libfreenect2::FrameMap frames;
        listener.waitForNewFrame(frames);
        listener.release(frames);
    }

    // Capture
    libfreenect2::FrameMap frames;
    listener.waitForNewFrame(frames);
    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    // Convert to OpenCV
    cv::Mat rgb_mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    cv::Mat rgb_bgr;
    cv::cvtColor(rgb_mat, rgb_bgr, cv::COLOR_BGRA2BGR);
    cv::imwrite(rgb_png, rgb_bgr);
    // Save as .npy (uint8)
    save_npy(rgb_npy, rgb_bgr.data, rgb_bgr.rows, rgb_bgr.cols * 3, 1, "|u1");

    // Depth (float32, in mm)
    cv::Mat depth_mat(depth->height, depth->width, CV_32FC1, depth->data);
    cv::Mat depth_mm;
    depth_mat.convertTo(depth_mm, CV_16UC1, 1000.0); // meters to mm
    cv::imwrite(depth_png, depth_mm);
    // Save as .npy (uint16)
    save_npy(depth_npy, depth_mm.data, depth_mm.rows, depth_mm.cols, 2, "<u2");

    listener.release(frames);
    dev->stop();
    dev->close();
    std::cout << "Capture complete.\n";
    return 0;
}
