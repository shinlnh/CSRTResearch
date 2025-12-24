#include <opencv2/opencv.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>

#include "../inc/csrt_tracker.hpp"

namespace {

cv::Rect g_selection_bbox;
bool g_selecting = false;
cv::Point g_origin;

void OnMouse(int event, int x, int y, int, void *) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        g_selecting = true;
        g_origin = cv::Point(x, y);
        g_selection_bbox = cv::Rect(x, y, 0, 0);
    } else if (event == cv::EVENT_LBUTTONUP) {
        g_selecting = false;
    } else if (event == cv::EVENT_MOUSEMOVE && g_selecting) {
        g_selection_bbox.x = std::min(x, g_origin.x);
        g_selection_bbox.y = std::min(y, g_origin.y);
        g_selection_bbox.width = std::abs(x - g_origin.x);
        g_selection_bbox.height = std::abs(y - g_origin.y);
    }
}

bool ParseArgs(int argc, char **argv, std::string &video_path, cv::Rect &init_bbox, bool &use_webcam) {
    use_webcam = false;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> [--bbox x,y,w,h]\n";
        std::cerr << "       " << argv[0] << " --webcam [--bbox x,y,w,h]\n";
        return false;
    }

    if (std::string(argv[1]) == "--webcam") {
        use_webcam = true;
    } else {
        video_path = argv[1];
    }

    for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--bbox") {
            int x = 0;
            int y = 0;
            int w = 0;
            int h = 0;
            if (std::sscanf(argv[i + 1], "%d,%d,%d,%d", &x, &y, &w, &h) == 4) {
                init_bbox = cv::Rect(x, y, w, h);
                return true;
            }
        }
    }
    return true;
}

void VisualizeTracking(cv::Mat &frame, const cv::Rect &bbox, const csrt::CsrtTracker &tracker, int frame_num) {
    cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

    std::ostringstream ss;
    ss << "Frame: " << frame_num;
    cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(0, 255, 0), 2);

    ss.str("");
    ss << "Peak: " << std::fixed << std::setprecision(3) << tracker.GetLastPeak();
    cv::putText(frame, ss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(255, 255, 0), 2);

    ss.str("");
    ss << "PSR: " << std::fixed << std::setprecision(2) << tracker.GetLastPsr();
    cv::putText(frame, ss.str(), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6,
        cv::Scalar(255, 255, 0), 2);

    const cv::Mat &response = tracker.GetResponseMap();
    if (!response.empty() && response.rows > 0 && response.cols > 0) {
        cv::Mat response_vis;
        cv::normalize(response, response_vis, 0, 255, cv::NORM_MINMAX);
        response_vis.convertTo(response_vis, CV_8U);
        cv::applyColorMap(response_vis, response_vis, cv::COLORMAP_JET);
        cv::resize(response_vis, response_vis, cv::Size(200, 200));

        cv::Rect roi(frame.cols - 210, 10, 200, 200);
        if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= frame.cols &&
            roi.y + roi.height <= frame.rows) {
            response_vis.copyTo(frame(roi));
            cv::putText(frame, "Response Map", cv::Point(frame.cols - 200, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    std::cout << "================================================================================\n";
    std::cout << "CSRT From Scratch Demo\n";
    std::cout << "================================================================================\n";

    std::string video_path;
    cv::Rect init_bbox;
    bool use_webcam = false;

    if (!ParseArgs(argc, argv, video_path, init_bbox, use_webcam)) {
        return -1;
    }

    cv::VideoCapture cap;
    if (use_webcam) {
        cap.open(0);
        std::cout << "Opening webcam...\n";
    } else {
        cap.open(video_path);
        std::cout << "Opening video: " << video_path << "\n";
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open video source\n";
        return -1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to read first frame\n";
        return -1;
    }

    if (init_bbox.width == 0 || init_bbox.height == 0) {
        std::cout << "\n=== Manual Selection ===\n";
        std::cout << "Drag a rectangle, then press ENTER\n";

        cv::namedWindow("Select Object", cv::WINDOW_NORMAL);
        cv::setMouseCallback("Select Object", OnMouse, nullptr);

        while (true) {
            cv::Mat display = frame.clone();
            if (g_selecting || (g_selection_bbox.width > 0 && g_selection_bbox.height > 0)) {
                cv::rectangle(display, g_selection_bbox, cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("Select Object", display);
            int key = cv::waitKey(10);
            if (key == 13 || key == 10) {
                if (g_selection_bbox.width > 10 && g_selection_bbox.height > 10) {
                    init_bbox = g_selection_bbox;
                    break;
                }
            } else if (key == 27) {
                std::cout << "Selection cancelled\n";
                return 0;
            }
        }
        cv::destroyWindow("Select Object");
    }

    std::cout << "Initial bbox: " << init_bbox << "\n";

    csrt::CsrtTracker tracker;
    if (!tracker.Init(frame, init_bbox)) {
        std::cerr << "Failed to initialize tracker\n";
        return -1;
    }

    std::cout << "\n=== Tracking Started ===\n";
    std::cout << "Press 'q' to quit, 'p' to pause\n";

    cv::namedWindow("CSRT From Scratch", cv::WINDOW_NORMAL);
    int frame_num = 0;
    bool paused = false;

    while (true) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << "End of video\n";
                break;
            }
            frame_num++;

            cv::Rect bbox;
            bool success = tracker.Update(frame, bbox);
            if (success) {
                VisualizeTracking(frame, bbox, tracker, frame_num);
            } else {
                cv::putText(frame, "Tracking Lost", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }
        }

        cv::imshow("CSRT From Scratch", frame);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        } else if (key == 'p' || key == 'P') {
            paused = !paused;
            std::cout << (paused ? "Paused\n" : "Resumed\n");
        }
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "\n=== Tracking Finished ===\n";
    std::cout << "Total frames: " << frame_num << "\n";
    return 0;
}
