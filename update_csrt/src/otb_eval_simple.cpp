#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <algorithm>
#include <array>
#include <atomic>
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

#include "../inc/csrt_tracker.hpp"

namespace fs = std::filesystem;

namespace {

struct BoundingBox {
    double x{0.0};
    double y{0.0};
    double width{0.0};
    double height{0.0};

    [[nodiscard]] double area() const {
        return std::max(width, 0.0) * std::max(height, 0.0);
    }

    [[nodiscard]] std::array<double, 2> center() const {
        return {x + width * 0.5, y + height * 0.5};
    }

    [[nodiscard]] BoundingBox intersect(const BoundingBox &other) const {
        const double x1 = std::max(x, other.x);
        const double y1 = std::max(y, other.y);
        const double x2 = std::min(x + width, other.x + other.width);
        const double y2 = std::min(y + height, other.y + other.height);
        return {x1, y1, std::max(0.0, x2 - x1), std::max(0.0, y2 - y1)};
    }

    [[nodiscard]] double iou(const BoundingBox &other) const {
        const double inter = intersect(other).area();
        const double uni = area() + other.area() - inter;
        return (uni > 0.0) ? (inter / uni) : 0.0;
    }
};

struct SequenceData {
    std::string name;
    std::vector<fs::path> frames;
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

std::vector<fs::path> collectFramePaths(const fs::path &imgDir) {
    std::vector<fs::path> frames;
    if (!fs::exists(imgDir)) {
        return frames;
    }
    for (const auto &entry : fs::directory_iterator(imgDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".JPG" || ext == ".jpeg" || ext == ".JPEG" ||
            ext == ".png" || ext == ".PNG" || ext == ".bmp" || ext == ".BMP") {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    return frames;
}

std::optional<std::vector<BoundingBox>> loadGroundTruth(const fs::path &gtFile) {
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
        for (char &ch : line) {
            if (ch == ',' || ch == '\t') {
                ch = ' ';
            }
        }
        std::stringstream ss(line);
        double x = 0.0;
        double y = 0.0;
        double w = 0.0;
        double h = 0.0;
        if (ss >> x >> y >> w >> h) {
            boxes.push_back({x - 1.0, y - 1.0, w, h});
        }
    }
    if (boxes.empty()) {
        return std::nullopt;
    }
    return boxes;
}

std::vector<fs::path> findGroundTruthFiles(const fs::path &seqPath) {
    std::vector<fs::path> gtFiles;
    for (const auto &entry : fs::directory_iterator(seqPath)) {
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

std::vector<fs::path> listSequences(const fs::path &root) {
    std::vector<fs::path> seqPaths;
    for (const auto &entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) {
            continue;
        }
        const auto dirName = entry.path().filename().string();
        if (dirName == "OTB-dataset" || dirName == ".git") {
            continue;
        }
        const auto imgDir = entry.path() / "img";
        const auto gtFiles = findGroundTruthFiles(entry.path());
        if (!gtFiles.empty() && fs::exists(imgDir)) {
            seqPaths.push_back(entry.path());
        }
    }
    std::sort(seqPaths.begin(), seqPaths.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.filename().string() < rhs.filename().string();
    });
    return seqPaths;
}

std::vector<SequenceData> loadAllSequenceVariants(const fs::path &seqPath) {
    std::vector<SequenceData> results;
    const auto gtFiles = findGroundTruthFiles(seqPath);
    const auto frames = collectFramePaths(seqPath / "img");
    
    if (frames.empty() || gtFiles.empty()) {
        return results;
    }
    
    const auto baseName = seqPath.filename().string();
    
    for (const auto &gtFile : gtFiles) {
        SequenceData data;
        auto gt = loadGroundTruth(gtFile);
        if (!gt || gt->empty()) {
            continue;
        }
        
        if (gtFiles.size() == 1) {
            data.name = baseName;
        } else {
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

double centerError(const BoundingBox &pred, const BoundingBox &gt) {
    const auto pc = pred.center();
    const auto gc = gt.center();
    const double dx = pc[0] - gc[0];
    const double dy = pc[1] - gc[1];
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<double> successCurve(const std::vector<double> &ious, const std::vector<double> &thresholds) {
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

double trapezoidAuc(const std::vector<double> &xs, const std::vector<double> &ys) {
    double area = 0.0;
    for (std::size_t i = 1; i < xs.size(); ++i) {
        const double dx = xs[i] - xs[i - 1];
        area += (ys[i] + ys[i - 1]) * 0.5 * dx;
    }
    return area;
}

double precisionAt(const std::vector<double> &values, double threshold) {
    const auto ok = std::count_if(values.begin(), values.end(), [threshold](double d) {
        return d <= threshold;
    });
    return values.empty() ? 0.0 : static_cast<double>(ok) / static_cast<double>(values.size());
}

SequenceResult evaluateSequence(const SequenceData &data) {
    SequenceResult result;
    
    if (data.frames.empty() || data.groundTruth.empty()) {
        return result;
    }

    cv::Mat first = cv::imread(data.frames.front().string(), cv::IMREAD_COLOR);
    if (first.empty()) {
        return result;
    }

    const auto initBox = data.groundTruth.front();
    cv::Rect initRect(
        static_cast<int>(std::round(initBox.x)),
        static_cast<int>(std::round(initBox.y)),
        static_cast<int>(std::round(initBox.width)),
        static_cast<int>(std::round(initBox.height)));

    csrt::CsrtTracker tracker;
    if (!tracker.Init(first, initRect)) {
        return result;
    }

    std::vector<double> ious;
    std::vector<double> errors;
    ious.reserve(data.frames.size());
    errors.reserve(data.frames.size());

    double trackingSeconds = 0.0;

    for (std::size_t i = 0; i < data.frames.size(); ++i) {
        cv::Mat frame = (i == 0) ? first.clone() : cv::imread(data.frames[i].string(), cv::IMREAD_COLOR);
        if (frame.empty()) {
            break;
        }

        cv::Rect bbox;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = tracker.Update(frame, bbox);
        const auto t1 = std::chrono::steady_clock::now();
        trackingSeconds += std::chrono::duration<double>(t1 - t0).count();

        std::optional<BoundingBox> pred;
        if (ok) {
            pred = BoundingBox{static_cast<double>(bbox.x), static_cast<double>(bbox.y),
                static_cast<double>(bbox.width), static_cast<double>(bbox.height)};
        }

        const auto &gt = data.groundTruth[i];
        const double iou = pred ? pred->iou(gt) : 0.0;
        const double error = pred ? centerError(*pred, gt) : std::numeric_limits<double>::infinity();

        ious.push_back(iou);
        errors.push_back(error);
    }

    result.ious = ious;
    result.centerErrors = errors;
    result.trackingSeconds = trackingSeconds;

    std::vector<double> thresholds;
    thresholds.reserve(21);
    for (int i = 0; i <= 20; ++i) {
        thresholds.push_back(static_cast<double>(i) * 0.05);
    }

    const auto curve = successCurve(ious, thresholds);
    const double auc = trapezoidAuc(thresholds, curve);
    const double success50 = precisionAt(ious, 0.5);
    const double precision20 = precisionAt(errors, 20.0);
    const double fps = trackingSeconds > 0.0 ? static_cast<double>(ious.size()) / trackingSeconds : 0.0;

    result.metrics = {data.name, ious.size(), auc, success50, precision20, fps};
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    fs::path datasetRoot;
    fs::path outputPath = "auc_update.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset-root" && i + 1 < argc) {
            datasetRoot = argv[++i];
        } else if (arg == "--output-csv" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: otb_eval_update [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --dataset-root <path>    Path to OTB100 root\n";
            std::cout << "  --output-csv <path>      Output CSV file (default: auc_update.csv)\n";
            std::cout << "  --help                   Show this help\n";
            return 0;
        }
    }

    if (datasetRoot.empty() || !fs::exists(datasetRoot)) {
        std::cerr << "Dataset root not found: " << datasetRoot << "\n";
        return 1;
    }

    const auto sequencePaths = listSequences(datasetRoot);
    if (sequencePaths.empty()) {
        std::cerr << "No sequences found under: " << datasetRoot << "\n";
        return 1;
    }

    // Collect all sequence data
    std::vector<SequenceData> allSequences;
    for (const auto &seqPath : sequencePaths) {
        auto variants = loadAllSequenceVariants(seqPath);
        allSequences.insert(allSequences.end(), variants.begin(), variants.end());
    }

    std::vector<SequenceResult> results(allSequences.size());

    std::cout << "Running update_csrt on " << allSequences.size() << " sequence(s)\n";
    
    // Multi-threading
    const unsigned int numThreads = std::thread::hardware_concurrency();
    std::cout << "Using " << numThreads << " threads for parallel processing\n\n";
    
    std::mutex coutMutex;
    std::atomic<std::size_t> sequenceCounter{0};
    
    auto processSequence = [&](std::size_t idx) {
        const auto &seqData = allSequences[idx];
        
        {
            std::lock_guard<std::mutex> lock(coutMutex);
            const auto current = ++sequenceCounter;
            std::cout << "[" << current << "/" << allSequences.size() << "] "
                      << seqData.name << " (" << seqData.frames.size() << " frames)\n";
        }

        results[idx] = evaluateSequence(seqData);
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
    
    for (auto &thread : threads) {
        thread.join();
    }
    
    // Aggregate results
    std::vector<double> allIous;
    std::vector<double> allErrors;
    double totalSeconds = 0.0;
    std::size_t totalFrames = 0;
    std::vector<SequenceMetrics> perSequence;
    
    for (const auto &result : results) {
        if (result.metrics.frames == 0) continue;
        allIous.insert(allIous.end(), result.ious.begin(), result.ious.end());
        allErrors.insert(allErrors.end(), result.centerErrors.begin(), result.centerErrors.end());
        totalFrames += result.metrics.frames;
        totalSeconds += result.trackingSeconds;
        perSequence.push_back(result.metrics);
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
    std::cout << "Update CSRT on OTB100\n";
    std::cout << "====================================================\n";
    std::cout << "Sequences:   " << perSequence.size() << "\n";
    std::cout << "Total frames: " << totalFrames << "\n";
    std::cout << "AUC:         " << std::fixed << std::setprecision(4) << overallAuc << "\n";
    std::cout << "Success@0.5: " << overallSuccess50 << "\n";
    std::cout << "Precision@20: " << overallPrecision20 << "\n";
    std::cout << "FPS:         " << std::setprecision(2) << overallFps << "\n";
    std::cout << "====================================================\n";

    // Write CSV
    std::ofstream csvFile(outputPath);
    if (!csvFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputPath << "\n";
        return 1;
    }

    csvFile << "sequence,frames,auc,success50,precision20,fps\n";
    for (const auto &metric : perSequence) {
        csvFile << metric.name << "," << metric.frames << ","
                << std::fixed << std::setprecision(6)
                << metric.auc << "," << metric.success50 << ","
                << metric.precision20 << "," << metric.fps << "\n";
    }
    csvFile << "OVERALL," << totalFrames << ","
            << overallAuc << "," << overallSuccess50 << ","
            << overallPrecision20 << "," << overallFps << "\n";

    std::cout << "\nWrote results to: " << outputPath << "\n";

    return 0;
}
