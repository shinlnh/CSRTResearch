/**
 * @file main.cpp
 * @brief Demo application for Updated CSRT Tracker
 * 
 * Usage:
 *   updated_csrt_demo <video_path> [--bbox x,y,w,h]
 *   updated_csrt_demo --webcam [--bbox x,y,w,h]
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "../inc/UpdatedCSRTTracker.hpp"

using namespace update_csrt;

// Global variables for mouse selection
cv::Rect g_selection_bbox;
bool g_selecting = false;
cv::Point g_origin;

/**
 * @brief Mouse callback for bbox selection
 */
void onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        g_selecting = true;
        g_origin = cv::Point(x, y);
        g_selection_bbox = cv::Rect(x, y, 0, 0);
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        g_selecting = false;
        if (g_selection_bbox.width > 10 && g_selection_bbox.height > 10) {
            std::cout << "Selected bbox: " << g_selection_bbox << std::endl;
        }
    }
    else if (event == cv::EVENT_MOUSEMOVE && g_selecting) {
        g_selection_bbox.x = std::min(x, g_origin.x);
        g_selection_bbox.y = std::min(y, g_origin.y);
        g_selection_bbox.width = std::abs(x - g_origin.x);
        g_selection_bbox.height = std::abs(y - g_origin.y);
    }
}

/**
 * @brief Parse command line arguments
 */
bool parseArgs(int argc, char** argv, std::string& video_path, cv::Rect& init_bbox, bool& use_webcam) {
    use_webcam = false;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> [--bbox x,y,w,h]" << std::endl;
        std::cerr << "       " << argv[0] << " --webcam [--bbox x,y,w,h]" << std::endl;
        return false;
    }
    
    // Check for webcam mode
    if (std::string(argv[1]) == "--webcam") {
        use_webcam = true;
    } else {
        video_path = argv[1];
    }
    
    // Parse optional bbox
    for (int i = 2; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--bbox") {
            std::string bbox_str = argv[i + 1];
            int x, y, w, h;
            if (sscanf(bbox_str.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) == 4) {
                init_bbox = cv::Rect(x, y, w, h);
                return true;
            }
        }
    }
    
    return true;
}

/**
 * @brief Visualize tracking results
 */
void visualizeTracking(cv::Mat& frame, const cv::Rect& bbox, 
                       const UpdatedCSRTTracker& tracker, int frame_num) {
    // Draw bounding box
    cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
    
    // Draw info text
    std::stringstream ss;
    ss << "Frame: " << frame_num;
    cv::putText(frame, ss.str(), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    // Display alpha value
    ss.str("");
    ss << "Alpha: " << std::fixed << std::setprecision(3) << tracker.getLastAlpha();
    cv::putText(frame, ss.str(), cv::Point(10, 60), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    
    // Display PSR
    ss.str("");
    ss << "PSR: " << std::fixed << std::setprecision(2) << tracker.getLastPSR();
    cv::putText(frame, ss.str(), cv::Point(10, 90), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    
    // Visualize response map (optional)
    cv::Mat response = tracker.getResponseMap();
    if (!response.empty() && response.rows > 0 && response.cols > 0) {
        cv::Mat response_vis;
        cv::normalize(response, response_vis, 0, 255, cv::NORM_MINMAX);
        response_vis.convertTo(response_vis, CV_8U);
        cv::applyColorMap(response_vis, response_vis, cv::COLORMAP_JET);
        
        // Resize for visualization
        cv::resize(response_vis, response_vis, cv::Size(200, 200));
        
        // Place in top-right corner
        cv::Rect roi(frame.cols - 210, 10, 200, 200);
        response_vis.copyTo(frame(roi));
        
        cv::putText(frame, "Response Map", cv::Point(frame.cols - 200, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

/**
 * @brief Main function
 */
int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << "Updated CSRT Tracker Demo\n";
    std::cout << "================================================================================\n";
    
    // Parse arguments
    std::string video_path;
    cv::Rect init_bbox;
    bool use_webcam;
    
    if (!parseArgs(argc, argv, video_path, init_bbox, use_webcam)) {
        return -1;
    }
    
    // Open video
    cv::VideoCapture cap;
    if (use_webcam) {
        cap.open(0);
        std::cout << "Opening webcam..." << std::endl;
    } else {
        cap.open(video_path);
        std::cout << "Opening video: " << video_path << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video source" << std::endl;
        return -1;
    }
    
    // Read first frame
    cv::Mat frame;
    cap >> frame;
    
    if (frame.empty()) {
        std::cerr << "Failed to read first frame" << std::endl;
        return -1;
    }
    
    std::cout << "Video resolution: " << frame.cols << "x" << frame.rows << std::endl;
    
    // Get initial bbox (manual selection or from args)
    if (init_bbox.width == 0 || init_bbox.height == 0) {
        std::cout << "\n=== Manual Selection ===" << std::endl;
        std::cout << "Select object by dragging a rectangle, then press ENTER" << std::endl;
        
        cv::namedWindow("Select Object", cv::WINDOW_NORMAL);
        cv::setMouseCallback("Select Object", onMouse, nullptr);
        
        while (true) {
            cv::Mat display = frame.clone();
            
            if (g_selecting || (g_selection_bbox.width > 0 && g_selection_bbox.height > 0)) {
                cv::rectangle(display, g_selection_bbox, cv::Scalar(0, 255, 0), 2);
            }
            
            cv::imshow("Select Object", display);
            
            int key = cv::waitKey(10);
            if (key == 13 || key == 10) {  // ENTER
                if (g_selection_bbox.width > 10 && g_selection_bbox.height > 10) {
                    init_bbox = g_selection_bbox;
                    break;
                }
            }
            else if (key == 27) {  // ESC
                std::cout << "Selection cancelled" << std::endl;
                return 0;
            }
        }
        
        cv::destroyWindow("Select Object");
    }
    
    std::cout << "Initial bbox: " << init_bbox << std::endl;
    
    // Create tracker
    Config config;
    config.visualize = true;
    config.verbose = true;
    
    UpdatedCSRTTracker tracker(config);
    
    // Initialize tracker
    if (!tracker.initialize(frame, init_bbox)) {
        std::cerr << "Failed to initialize tracker" << std::endl;
        return -1;
    }
    
    std::cout << "\n=== Tracking Started ===" << std::endl;
    std::cout << "Press 'q' to quit, 'p' to pause" << std::endl;
    
    // Tracking loop
    cv::namedWindow("Updated CSRT Tracker", cv::WINDOW_NORMAL);
    
    int frame_num = 0;
    bool paused = false;
    
    while (true) {
        if (!paused) {
            cap >> frame;
            
            if (frame.empty()) {
                std::cout << "End of video" << std::endl;
                break;
            }
            
            frame_num++;
            
            // Track
            cv::Rect bbox;
            bool success = tracker.track(frame, bbox);
            
            if (success) {
                visualizeTracking(frame, bbox, tracker, frame_num);
            } else {
                cv::putText(frame, "Tracking Lost", cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }
        }
        
        cv::imshow("Updated CSRT Tracker", frame);
        
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
        else if (key == 'p' || key == 'P') {
            paused = !paused;
            std::cout << (paused ? "Paused" : "Resumed") << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "\n=== Tracking Finished ===" << std::endl;
    std::cout << "Total frames: " << frame_num << std::endl;
    
    return 0;
}
