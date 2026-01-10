#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/tracking.hpp>

namespace {

struct BoundingBox {
    float x{0.0f};
    float y{0.0f};
    float width{0.0f};
    float height{0.0f};

    BoundingBox() = default;

    BoundingBox(float x_, float y_, float w_, float h_)
        : x(x_), y(y_), width(w_), height(h_) {}

    [[nodiscard]] float area() const {
        return std::max(width, 0.0f) * std::max(height, 0.0f);
    }

    [[nodiscard]] std::array<float, 2> center() const {
        return {x + width * 0.5f, y + height * 0.5f};
    }

    [[nodiscard]] BoundingBox intersect(const BoundingBox& other) const {
        const float x1 = std::max(x, other.x);
        const float y1 = std::max(y, other.y);
        const float x2 = std::min(x + width, other.x + other.width);
        const float y2 = std::min(y + height, other.y + other.height);
        return {x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1)};
    }

    [[nodiscard]] float iou(const BoundingBox& other) const {
        const float inter = intersect(other).area();
        const float uni = area() + other.area() - inter;
        return (uni > 0.0f) ? (inter / uni) : 0.0f;
    }
};

struct Args {
    std::string datasetRoot{"otb100"};
    std::optional<std::string> sequenceName;
    bool display{false};
    std::optional<int> maxFrames;
    std::optional<std::filesystem::path> saveVisRoot;
    std::optional<std::filesystem::path> outputCsv;
};

void printUsage() {
    std::cout << "Usage: csrtpure [options]\n"
                 "Options:\n"
                 "  --dataset-root <path>       Path to OTB100 root (default: otb100)\n"
                 "  --sequence <name>           Run a single sequence (default: run all)\n"
                 "  --display                   Show visualization with GT (green) and CSRT (red)\n"
                 "  --save-vis <path>           Save drawn frames to this directory\n"
                 "  --output-csv <path>          Write per-sequence metrics to CSV\n"
                 "  --max-frames <int>          Limit number of frames per sequence\n"
                 "  --help                      Show this help message\n";
}

bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string current = argv[i];
        if (current == "--sequence" && (i + 1) < argc) {
            args.sequenceName = std::string(argv[++i]);
        } else if (current == "--dataset-root" && (i + 1) < argc) {
            args.datasetRoot = argv[++i];
        } else if (current == "--display") {
            args.display = true;
        } else if (current == "--max-frames" && (i + 1) < argc) {
            args.maxFrames = std::stoi(argv[++i]);
        } else if (current == "--save-vis" && (i + 1) < argc) {
            args.saveVisRoot = std::filesystem::path(argv[++i]);
        } else if (current == "--output-csv" && (i + 1) < argc) {
            args.outputCsv = std::filesystem::path(argv[++i]);
        } else if (current == "--help") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown option: " << current << "\n";
            return false;
        }
    }
    return true;
}

std::vector<std::filesystem::path> collectFramePaths(const std::filesystem::path& imgDir) {
    std::vector<std::filesystem::path> frames;
    if (!std::filesystem::exists(imgDir)) {
        return frames;
    }
    for (const auto& entry : std::filesystem::directory_iterator(imgDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".JPG" ||
            ext == ".jpeg" || ext == ".JPEG" ||
            ext == ".png" || ext == ".PNG" ||
            ext == ".bmp" || ext == ".BMP") {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    return frames;
}

std::optional<std::vector<BoundingBox>> loadGroundTruth(const std::filesystem::path& gtFile) {
    std::ifstream stream(gtFile);
    if (!stream.is_open()) {
        return std::nullopt;
    }
    std::vector<BoundingBox> boxes;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        for (char& ch : line) {
            if (ch == ',' || ch == '\t') {
                ch = ' ';
            }
        }
        std::stringstream ss(line);
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        if (ss >> x >> y >> w >> h) {
            // OTB annotations are 1-indexed; convert to 0-index for OpenCV.
            boxes.emplace_back(x - 1.0f, y - 1.0f, w, h);
        }
    }
    if (boxes.empty()) {
        return std::nullopt;
    }
    return boxes;
}

struct SequenceData {
    std::string name;
    std::vector<std::filesystem::path> frames;
    std::vector<BoundingBox> groundTruth;
};

struct SequenceMetrics {
    std::string name;
    std::size_t frames{0};
    double auc{0.0};
    double success50{0.0};
    double precision20{0.0};
    double fps{0.0};
};

struct SequenceResult {
    SequenceMetrics metrics;
    std::vector<double> ious;
    std::vector<double> centerErrors;
    double trackingSeconds{0.0};
};

std::vector<std::filesystem::path> findGroundTruthFiles(const std::filesystem::path& seqPath) {
    std::vector<std::filesystem::path> gtFiles;
    // Collect all groundtruth_rect*.txt files
    for (const auto& entry : std::filesystem::directory_iterator(seqPath)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto filename = entry.path().filename().string();
        if (filename.find("groundtruth_rect") == 0 && filename.rfind(".txt") == filename.length() - 4) {
            gtFiles.push_back(entry.path());
        }
    }
    std::sort(gtFiles.begin(), gtFiles.end());
    return gtFiles;
}

std::vector<std::filesystem::path> listSequences(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> seqPaths;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory()) {
            continue;
        }
        // Skip metadata/helper directories
        const auto dirName = entry.path().filename().string();
        if (dirName == "OTB-dataset" || dirName == ".git") {
            continue;
        }
        const auto imgDir = entry.path() / "img";
        const auto gtFiles = findGroundTruthFiles(entry.path());
        if (!gtFiles.empty() && std::filesystem::exists(imgDir)) {
            seqPaths.push_back(entry.path());
        }
    }
    std::sort(seqPaths.begin(), seqPaths.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.filename().string() < rhs.filename().string();
    });
    return seqPaths;
}

std::vector<SequenceData> loadAllSequenceVariants(const std::filesystem::path& seqPath) {
    std::vector<SequenceData> results;
    const auto gtFiles = findGroundTruthFiles(seqPath);
    const auto frames = collectFramePaths(seqPath / "img");
    
    if (frames.empty() || gtFiles.empty()) {
        return results;
    }
    
    const auto baseName = seqPath.filename().string();
    
    for (const auto& gtFile : gtFiles) {
        SequenceData data;
        auto gt = loadGroundTruth(gtFile);
        if (!gt || gt->empty()) {
            continue;
        }
        
        // Generate name based on ground truth file variant
        if (gtFiles.size() == 1) {
            data.name = baseName;
        } else {
            // Extract variant number from filename (e.g., groundtruth_rect.1.txt -> _1)
            const auto gtFilename = gtFile.filename().string();
            const auto dotPos = gtFilename.rfind(".txt");
            if (dotPos != std::string::npos && dotPos > 0) {
                const auto beforeTxt = gtFilename.substr(0, dotPos);
                const auto lastDot = beforeTxt.rfind('.');
                if (lastDot != std::string::npos) {
                    const auto variant = beforeTxt.substr(lastDot + 1);
                    data.name = baseName + "_" + variant;
                } else {
                    data.name = baseName;
                }
            } else {
                data.name = baseName;
            }
        }
        
        const std::size_t count = std::min<std::size_t>(frames.size(), gt->size());
        data.frames.assign(frames.begin(), frames.begin() + static_cast<std::ptrdiff_t>(count));
        data.groundTruth.assign(gt->begin(), gt->begin() + static_cast<std::ptrdiff_t>(count));
        results.push_back(std::move(data));
    }
    
    return results;
}

std::optional<SequenceData> loadSequence(const std::filesystem::path& seqPath) {
    auto variants = loadAllSequenceVariants(seqPath);
    if (variants.empty()) {
        return std::nullopt;
    }
    return variants[0];
}

double centerError(const BoundingBox& pred, const BoundingBox& gt) {
    const auto pc = pred.center();
    const auto gc = gt.center();
    const double dx = static_cast<double>(pc[0] - gc[0]);
    const double dy = static_cast<double>(pc[1] - gc[1]);
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<double> successCurve(const std::vector<double>& ious, const std::vector<double>& thresholds) {
    std::vector<double> curve;
    curve.reserve(thresholds.size());
    if (ious.empty()) {
        curve.assign(thresholds.size(), 0.0);
        return curve;
    }
    for (double t : thresholds) {
        const auto success = std::count_if(ious.begin(), ious.end(), [t](double v) { return v >= t; });
        curve.push_back(static_cast<double>(success) / static_cast<double>(ious.size()));
    }
    return curve;
}

double trapezoidAuc(const std::vector<double>& xs, const std::vector<double>& ys) {
    double area = 0.0;
    for (std::size_t i = 1; i < xs.size(); ++i) {
        const double dx = xs[i] - xs[i - 1];
        area += (ys[i] + ys[i - 1]) * 0.5 * dx;
    }
    return area;
}

double precisionAt(const std::vector<double>& errors, double threshold) {
    const auto ok = std::count_if(errors.begin(), errors.end(), [threshold](double d) {
        return d <= threshold;
    });
    return errors.empty() ? 0.0 : static_cast<double>(ok) / static_cast<double>(errors.size());
}

void drawOverlay(cv::Mat& frame,
                 const BoundingBox& gt,
                 const std::optional<BoundingBox>& pred,
                 double iou,
                 double error,
                 int frameIndex) {
    cv::rectangle(frame,
                  cv::Rect(static_cast<int>(std::round(gt.x)),
                           static_cast<int>(std::round(gt.y)),
                           static_cast<int>(std::round(gt.width)),
                           static_cast<int>(std::round(gt.height))),
                  cv::Scalar(0, 200, 0), 2);

    if (pred) {
        cv::rectangle(frame,
                      cv::Rect(static_cast<int>(std::round(pred->x)),
                               static_cast<int>(std::round(pred->y)),
                               static_cast<int>(std::round(pred->width)),
                               static_cast<int>(std::round(pred->height))),
                      cv::Scalar(0, 0, 255), 2);
    }

    std::ostringstream oss;
    oss << "F:" << frameIndex + 1
        << " IoU:" << std::fixed << std::setprecision(3) << iou
        << " | Err(px):" << std::setprecision(1) << error;
    cv::putText(frame, oss.str(), {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(240, 240, 240), 2, cv::LINE_AA);
    cv::putText(frame, "GT", {static_cast<int>(gt.x), static_cast<int>(gt.y) - 6},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 0), 1, cv::LINE_AA);
    if (pred) {
        cv::putText(frame, "CSRT", {static_cast<int>(pred->x), static_cast<int>(pred->y) - 6},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
}

struct RawSequenceResult {
    std::string name;
    std::vector<double> ious;
    std::vector<double> centerErrors;
    double fps{0.0};
    double totalSeconds{0.0};
};

std::optional<RawSequenceResult> runSequence(const SequenceData& data,
                                          const Args& args,
                                          const std::optional<std::filesystem::path>& visRoot) {
    if (data.frames.empty() || data.groundTruth.empty()) {
        return std::nullopt;
    }

    const auto first = cv::imread(data.frames.front().string(), cv::IMREAD_COLOR);
    if (first.empty()) {
        std::cerr << "Failed to read first frame for " << data.name << "\n";
        return std::nullopt;
    }

    const auto initBox = data.groundTruth.front();
    cv::Rect initRect(
        static_cast<int>(std::round(initBox.x)),
        static_cast<int>(std::round(initBox.y)),
        static_cast<int>(std::round(initBox.width)),
        static_cast<int>(std::round(initBox.height)));

    auto tracker = cv::TrackerCSRT::create();
    tracker->init(first, initRect);

    std::filesystem::path sequenceVisDir;
    if (visRoot) {
        sequenceVisDir = *visRoot / data.name;
        std::filesystem::create_directories(sequenceVisDir);
    }

    RawSequenceResult result;
    result.name = data.name;
    result.ious.reserve(data.frames.size());
    result.centerErrors.reserve(data.frames.size());

    const std::size_t frameLimit = args.maxFrames
        ? std::min<std::size_t>(*args.maxFrames, data.frames.size())
        : data.frames.size();

    double trackingSeconds = 0.0;
    for (std::size_t i = 0; i < frameLimit; ++i) {
        cv::Mat frame = (i == 0) ? first.clone() : cv::imread(data.frames[i].string(), cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "Failed to read frame " << data.frames[i] << "\n";
            break;
        }

        cv::Rect rect;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = tracker->update(frame, rect);
        const auto t1 = std::chrono::steady_clock::now();
        trackingSeconds += std::chrono::duration<double>(t1 - t0).count();

        std::optional<BoundingBox> pred;
        if (ok) {
            pred = BoundingBox{
                static_cast<float>(rect.x),
                static_cast<float>(rect.y),
                static_cast<float>(rect.width),
                static_cast<float>(rect.height)};
        }

        const auto& gt = data.groundTruth[i];
        const double iou = pred ? pred->iou(gt) : 0.0;
        const double error = pred ? centerError(*pred, gt) : std::numeric_limits<double>::infinity();

        result.ious.push_back(iou);
        result.centerErrors.push_back(error);

        if (args.display || visRoot) {
            cv::Mat vis = frame.clone();
            drawOverlay(vis, gt, pred, iou, error, static_cast<int>(i));
            if (args.display) {
                cv::imshow("CSRT (pure)", vis);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {
                    break;
                }
            }
            if (visRoot) {
                const auto outFile = sequenceVisDir / data.frames[i].filename();
                cv::imwrite(outFile.string(), vis);
            }
        }
    }

    result.totalSeconds = trackingSeconds;
    result.fps = trackingSeconds > 0.0
        ? static_cast<double>(result.ious.size()) / trackingSeconds
        : 0.0;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        return 1;
    }

    const std::filesystem::path datasetRoot(args.datasetRoot);
    if (!std::filesystem::exists(datasetRoot)) {
        std::cerr << "Dataset root not found: " << datasetRoot << "\n";
        return 1;
    }

    std::vector<std::filesystem::path> sequencePaths;
    if (args.sequenceName) {
        sequencePaths.push_back(datasetRoot / *args.sequenceName);
    } else {
        sequencePaths = listSequences(datasetRoot);
    }

    if (sequencePaths.empty()) {
        std::cerr << "No sequences found under " << datasetRoot << "\n";
        return 1;
    }

    std::optional<std::filesystem::path> visRoot = args.saveVisRoot;
    if (visRoot) {
        std::filesystem::create_directories(*visRoot);
    }

    // Collect all sequence data first
    std::vector<SequenceData> allSequences;
    for (const auto& seqPath : sequencePaths) {
        auto variants = loadAllSequenceVariants(seqPath);
        allSequences.insert(allSequences.end(), variants.begin(), variants.end());
    }

    std::vector<SequenceResult> results(allSequences.size());

    std::cout << "Running pure CSRT on " << allSequences.size() << " sequence(s)\n";
    
    // Multi-threading
    const unsigned int numThreads = std::thread::hardware_concurrency();
    std::cout << "Using " << numThreads << " threads for parallel processing\n\n";
    
    std::mutex coutMutex;
    std::atomic<std::size_t> sequenceCounter{0};
    
    auto processSequence = [&](std::size_t idx) {
        const auto& seqData = allSequences[idx];
        
        {
            std::lock_guard<std::mutex> lock(coutMutex);
            const auto current = ++sequenceCounter;
            std::cout << "[" << current << "/" << allSequences.size() << "] "
                      << seqData.name << " (" << seqData.frames.size() << " frames)\n";
        }

        const auto result = runSequence(seqData, args, visRoot);
        if (!result) {
            std::lock_guard<std::mutex> lock(coutMutex);
            std::cerr << "Failed on sequence " << seqData.name << "\n";
            return;
        }

        const std::vector<double> thresholds = [] {
            std::vector<double> ts;
            for (int i = 0; i <= 20; ++i) {
                ts.push_back(static_cast<double>(i) * 0.05);
            }
            return ts;
        }();
        const auto scurve = successCurve(result->ious, thresholds);
        const double auc = trapezoidAuc(thresholds, scurve);
        const double success50 = precisionAt(result->ious, 0.5);
        const double precision20 = precisionAt(result->centerErrors, 20.0);

        results[idx].metrics = {seqData.name, result->ious.size(), auc, success50, precision20, result->fps};
        results[idx].ious = result->ious;
        results[idx].centerErrors = result->centerErrors;
        results[idx].trackingSeconds = result->totalSeconds;
    };
    
    std::vector<std::thread> threads;
    const std::size_t batchSize = (allSequences.size() + numThreads - 1) / numThreads;
    
    for (unsigned int t = 0; t < numThreads; ++t) {
        std::size_t startIdx = t * batchSize;
        std::size_t endIdx = std::min(startIdx + batchSize, allSequences.size());
        if (startIdx < allSequences.size()) {
            threads.emplace_back([&, startIdx, endIdx]() {
                for (std::size_t i = startIdx; i < endIdx; ++i) {
                    processSequence(i);
                }
            });
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Aggregate results
    std::vector<double> allIous;
    std::vector<double> allErrors;
    double totalSeconds = 0.0;
    std::size_t totalFrames = 0;
    std::vector<SequenceMetrics> perSequence;
    
    for (const auto& result : results) {
        if (result.metrics.frames == 0) continue;
        allIous.insert(allIous.end(), result.ious.begin(), result.ious.end());
        allErrors.insert(allErrors.end(), result.centerErrors.begin(), result.centerErrors.end());
        totalFrames += result.metrics.frames;
        totalSeconds += result.trackingSeconds;
        perSequence.push_back(result.metrics);
    }

    if (args.display) {
        cv::destroyAllWindows();
    }

    if (allIous.empty()) {
        std::cerr << "No results produced.\n";
        return 1;
    }

    std::vector<double> thresholds;
    thresholds.reserve(21);
    for (int i = 0; i <= 20; ++i) {
        thresholds.push_back(static_cast<double>(i) * 0.05);
    }
    const auto overallCurve = successCurve(allIous, thresholds);
    const double overallAuc = trapezoidAuc(thresholds, overallCurve);
    const double overallSuccess50 = precisionAt(allIous, 0.5);
    const double overallPrecision20 = precisionAt(allErrors, 20.0);
    const double overallFps = (totalSeconds > 0.0 && totalFrames > 0)
        ? static_cast<double>(totalFrames) / totalSeconds
        : 0.0;

    std::cout << "\n====================================================\n";
    std::cout << "Pure CSRT on OTB100 (paper-style metrics)\n";
    std::cout << "Sequences: " << sequencePaths.size()
              << " | Frames: " << totalFrames << "\n";
    std::cout << "AUC (Success plot): " << std::fixed << std::setprecision(3) << overallAuc << "\n";
    std::cout << "Success@0.5: " << overallSuccess50
              << " | Precision@20px: " << overallPrecision20 << "\n";
    std::cout << "Tracking FPS (tracker only): " << std::setprecision(2) << overallFps << "\n";
    std::cout << "====================================================\n";

    if (args.outputCsv) {
        std::ofstream out(*args.outputCsv);
        if (!out.is_open()) {
            std::cerr << "Failed to open output CSV: " << *args.outputCsv << "\n";
            return 1;
        }
        out << "sequence,frames,auc,success50,precision20,fps\n";
        out << std::fixed << std::setprecision(6);
        for (const auto& entry : perSequence) {
            out << entry.name << "," << entry.frames << ","
                << entry.auc << "," << entry.success50 << ","
                << entry.precision20 << "," << entry.fps << "\n";
        }
        out << "OVERALL," << totalFrames << ","
            << overallAuc << "," << overallSuccess50 << ","
            << overallPrecision20 << "," << overallFps << "\n";
        std::cout << "Wrote CSV to: " << *args.outputCsv << "\n";
    }

    return 0;
}
