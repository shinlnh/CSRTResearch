#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
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
    fs::path root;
    std::vector<fs::path> frames;
    std::vector<BoundingBox> ground_truth;
};

struct TrackerMetrics {
    std::vector<double> ious;
    std::vector<double> errors;
    double auc{0.0};
    double success50{0.0};
    double precision20{0.0};
    double fps{0.0};
    double tracking_seconds{0.0};
};

struct TrackerPairMetrics {
    TrackerMetrics update;
    TrackerMetrics pure;
};

void PrintUsage() {
    std::cout << "Usage: otb_compare --dataset-root <path> [--output <csv>] [--max-frames <n>]\n";
}

std::vector<fs::path> CollectFramePaths(const fs::path &img_dir) {
    std::vector<fs::path> frames;
    if (!fs::exists(img_dir)) {
        return frames;
    }

    for (const auto &entry : fs::directory_iterator(img_dir)) {
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

std::optional<std::vector<BoundingBox>> LoadGroundTruth(const fs::path &gt_file) {
    std::ifstream stream(gt_file);
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

bool HasGroundTruthFiles(const fs::path &seq_root) {
    if (!fs::exists(seq_root) || !fs::is_directory(seq_root)) {
        return false;
    }
    const fs::path gt = seq_root / "groundtruth_rect.txt";
    if (fs::exists(gt)) {
        return true;
    }
    for (const auto &entry : fs::directory_iterator(seq_root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name.rfind("groundtruth_rect.", 0) == 0 && entry.path().extension() == ".txt") {
            return true;
        }
    }
    return false;
}

std::vector<fs::path> GetGroundTruthFiles(const fs::path &seq_root) {
    std::vector<fs::path> files;
    const fs::path gt = seq_root / "groundtruth_rect.txt";
    if (fs::exists(gt)) {
        files.push_back(gt);
        return files;
    }
    for (const auto &entry : fs::directory_iterator(seq_root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name.rfind("groundtruth_rect.", 0) == 0 && entry.path().extension() == ".txt") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool HasGroundTruthAndImages(const fs::path &seq_root) {
    if (!HasGroundTruthFiles(seq_root)) {
        return false;
    }
    const fs::path img = seq_root / "img";
    if (!fs::exists(img) || !fs::is_directory(img)) {
        return false;
    }
    const auto frames = CollectFramePaths(img);
    return !frames.empty();
}

std::optional<fs::path> ResolveSequenceRoot(const fs::path &seq_path) {
    if (HasGroundTruthAndImages(seq_path)) {
        return seq_path;
    }

    const fs::path nested = seq_path / seq_path.filename();
    if (HasGroundTruthAndImages(nested)) {
        return nested;
    }

    for (const auto &entry : fs::directory_iterator(seq_path)) {
        if (!entry.is_directory()) {
            continue;
        }
        if (HasGroundTruthAndImages(entry.path())) {
            return entry.path();
        }
    }

    return std::nullopt;
}

std::vector<SequenceData> LoadSequences(const fs::path &dataset_root) {
    std::vector<SequenceData> sequences;

    if (HasGroundTruthAndImages(dataset_root)) {
        auto frames = CollectFramePaths(dataset_root / "img");
        if (!frames.empty()) {
            const auto gt_files = GetGroundTruthFiles(dataset_root);
            for (const auto &gt_file : gt_files) {
                auto gt = LoadGroundTruth(gt_file);
                if (!gt) {
                    continue;
                }
                const std::size_t count = std::min(frames.size(), gt->size());
                SequenceData data;
                const auto base_name = dataset_root.filename().string();
                const auto gt_name = gt_file.filename().string();
                if (gt_name != "groundtruth_rect.txt") {
                    const std::string prefix = "groundtruth_rect.";
                    const std::size_t start = prefix.size();
                    const std::size_t end = gt_name.rfind(".txt");
                    std::string suffix;
                    if (end != std::string::npos && end > start) {
                        suffix = gt_name.substr(start, end - start);
                    }
                    data.name = suffix.empty() ? base_name : base_name + "_" + suffix;
                } else {
                    data.name = base_name;
                }
                data.root = dataset_root;
                data.frames.assign(frames.begin(), frames.begin() + static_cast<std::ptrdiff_t>(count));
                data.ground_truth.assign(gt->begin(), gt->begin() + static_cast<std::ptrdiff_t>(count));
                sequences.push_back(std::move(data));
            }
        }
        return sequences;
    }

    for (const auto &entry : fs::directory_iterator(dataset_root)) {
        if (!entry.is_directory()) {
            continue;
        }
        const auto resolved = ResolveSequenceRoot(entry.path());
        if (!resolved) {
            continue;
        }
        auto frames = CollectFramePaths(*resolved / "img");
        if (frames.empty()) {
            continue;
        }
        const auto gt_files = GetGroundTruthFiles(*resolved);
        for (const auto &gt_file : gt_files) {
            auto gt = LoadGroundTruth(gt_file);
            if (!gt) {
                continue;
            }
            const std::size_t count = std::min(frames.size(), gt->size());
            SequenceData data;
            const auto base_name = entry.path().filename().string();
            const auto gt_name = gt_file.filename().string();
            if (gt_name != "groundtruth_rect.txt") {
                const std::string prefix = "groundtruth_rect.";
                const std::size_t start = prefix.size();
                const std::size_t end = gt_name.rfind(".txt");
                std::string suffix;
                if (end != std::string::npos && end > start) {
                    suffix = gt_name.substr(start, end - start);
                }
                data.name = suffix.empty() ? base_name : base_name + "_" + suffix;
            } else {
                data.name = base_name;
            }
            data.root = *resolved;
            data.frames.assign(frames.begin(), frames.begin() + static_cast<std::ptrdiff_t>(count));
            data.ground_truth.assign(gt->begin(), gt->begin() + static_cast<std::ptrdiff_t>(count));
            sequences.push_back(std::move(data));
        }
    }

    std::sort(sequences.begin(), sequences.end(), [](const SequenceData &a, const SequenceData &b) {
        return a.name < b.name;
    });
    return sequences;
}

double CenterError(const BoundingBox &pred, const BoundingBox &gt) {
    const auto pc = pred.center();
    const auto gc = gt.center();
    const double dx = pc[0] - gc[0];
    const double dy = pc[1] - gc[1];
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<double> SuccessCurve(const std::vector<double> &ious, const std::vector<double> &thresholds) {
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

double TrapezoidAuc(const std::vector<double> &xs, const std::vector<double> &ys) {
    double area = 0.0;
    for (std::size_t i = 1; i < xs.size(); ++i) {
        const double dx = xs[i] - xs[i - 1];
        area += (ys[i] + ys[i - 1]) * 0.5 * dx;
    }
    return area;
}

double PrecisionAt(const std::vector<double> &values, double threshold) {
    const auto ok = std::count_if(values.begin(), values.end(), [threshold](double d) {
        return d <= threshold;
    });
    return values.empty() ? 0.0 : static_cast<double>(ok) / static_cast<double>(values.size());
}

TrackerMetrics ComputeMetrics(const std::vector<double> &ious,
    const std::vector<double> &errors, double tracking_seconds) {
    TrackerMetrics metrics;
    metrics.ious = ious;
    metrics.errors = errors;
    metrics.tracking_seconds = tracking_seconds;

    std::vector<double> thresholds;
    thresholds.reserve(21);
    for (int i = 0; i <= 20; ++i) {
        thresholds.push_back(static_cast<double>(i) * 0.05);
    }

    const auto curve = SuccessCurve(ious, thresholds);
    metrics.auc = TrapezoidAuc(thresholds, curve);
    metrics.success50 = PrecisionAt(ious, 0.5);
    metrics.precision20 = PrecisionAt(errors, 20.0);
    metrics.fps = tracking_seconds > 0.0 ? static_cast<double>(ious.size()) / tracking_seconds : 0.0;
    return metrics;
}

TrackerPairMetrics EvaluateTrackers(const SequenceData &data, const std::optional<int> &max_frames) {
    TrackerPairMetrics metrics;
    if (data.frames.empty() || data.ground_truth.empty()) {
        return metrics;
    }

    cv::Mat first = cv::imread(data.frames.front().string(), cv::IMREAD_COLOR);
    if (first.empty()) {
        return metrics;
    }

    const auto init_box = data.ground_truth.front();
    cv::Rect init_rect(
        static_cast<int>(std::round(init_box.x)),
        static_cast<int>(std::round(init_box.y)),
        static_cast<int>(std::round(init_box.width)),
        static_cast<int>(std::round(init_box.height)));

    csrt::CsrtTracker update_tracker;
    if (!update_tracker.Init(first, init_rect)) {
        return metrics;
    }

    auto pure_tracker = cv::TrackerCSRT::create();
    pure_tracker->init(first, init_rect);

    const std::size_t limit = max_frames
        ? std::min<std::size_t>(*max_frames, data.frames.size())
        : data.frames.size();

    std::vector<double> ious_update;
    std::vector<double> errors_update;
    std::vector<double> ious_pure;
    std::vector<double> errors_pure;
    ious_update.reserve(limit);
    errors_update.reserve(limit);
    ious_pure.reserve(limit);
    errors_pure.reserve(limit);

    double update_seconds = 0.0;
    double pure_seconds = 0.0;

    for (std::size_t i = 0; i < limit; ++i) {
        cv::Mat frame = (i == 0) ? first.clone() : cv::imread(data.frames[i].string(), cv::IMREAD_COLOR);
        if (frame.empty()) {
            break;
        }

        cv::Rect bbox_update;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok_update = update_tracker.Update(frame, bbox_update);
        const auto t1 = std::chrono::steady_clock::now();
        update_seconds += std::chrono::duration<double>(t1 - t0).count();

        cv::Rect bbox_pure;
        const auto t2 = std::chrono::steady_clock::now();
        const bool ok_pure = pure_tracker->update(frame, bbox_pure);
        const auto t3 = std::chrono::steady_clock::now();
        pure_seconds += std::chrono::duration<double>(t3 - t2).count();

        std::optional<BoundingBox> pred_update;
        if (ok_update) {
            pred_update = BoundingBox{static_cast<double>(bbox_update.x), static_cast<double>(bbox_update.y),
                static_cast<double>(bbox_update.width), static_cast<double>(bbox_update.height)};
        }
        std::optional<BoundingBox> pred_pure;
        if (ok_pure) {
            pred_pure = BoundingBox{static_cast<double>(bbox_pure.x), static_cast<double>(bbox_pure.y),
                static_cast<double>(bbox_pure.width), static_cast<double>(bbox_pure.height)};
        }

        const auto &gt = data.ground_truth[i];
        const double iou_update = pred_update ? pred_update->iou(gt) : 0.0;
        const double iou_pure = pred_pure ? pred_pure->iou(gt) : 0.0;
        const double error_update = pred_update ? CenterError(*pred_update, gt) : std::numeric_limits<double>::infinity();
        const double error_pure = pred_pure ? CenterError(*pred_pure, gt) : std::numeric_limits<double>::infinity();

        ious_update.push_back(iou_update);
        ious_pure.push_back(iou_pure);
        errors_update.push_back(error_update);
        errors_pure.push_back(error_pure);
    }

    metrics.update = ComputeMetrics(ious_update, errors_update, update_seconds);
    metrics.pure = ComputeMetrics(ious_pure, errors_pure, pure_seconds);
    return metrics;
}

}  // namespace

int main(int argc, char **argv) {
    fs::path dataset_root;
    fs::path output_path = "auc_compare.csv";  // Write to current directory
    std::optional<int> max_frames;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset-root" && i + 1 < argc) {
            dataset_root = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            PrintUsage();
            return 0;
        }
    }

    if (dataset_root.empty() || !fs::exists(dataset_root)) {
        std::cerr << "Dataset root not found: " << dataset_root << "\n";
        PrintUsage();
        return 1;
    }

    const auto sequences = LoadSequences(dataset_root);
    if (sequences.empty()) {
        std::cerr << "No valid sequences found under: " << dataset_root << "\n";
        return 1;
    }

    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << "\n";
        return 1;
    }

    out << "sequence,frames,auc_update,auc_pure,success50_update,success50_pure,precision20_update,precision20_pure,fps_update,fps_pure,delta_auc\n";

    std::vector<double> all_ious_update;
    std::vector<double> all_errors_update;
    std::vector<double> all_ious_pure;
    std::vector<double> all_errors_pure;
    double total_seconds_update = 0.0;
    double total_seconds_pure = 0.0;
    std::size_t total_frames = 0;

    std::cout << "Comparing trackers on " << sequences.size() << " sequence(s)...\n";

    for (const auto &seq : sequences) {
        std::cout << "  " << seq.name << " (" << seq.frames.size() << " frames)" << std::endl;

        const auto metrics_pair = EvaluateTrackers(seq, max_frames);
        const auto &update_metrics = metrics_pair.update;
        const auto &pure_metrics = metrics_pair.pure;

        const double delta_auc = update_metrics.auc - pure_metrics.auc;

        out << seq.name << "," << seq.frames.size() << ","
            << std::fixed << std::setprecision(4)
            << update_metrics.auc << "," << pure_metrics.auc << ","
            << update_metrics.success50 << "," << pure_metrics.success50 << ","
            << update_metrics.precision20 << "," << pure_metrics.precision20 << ","
            << update_metrics.fps << "," << pure_metrics.fps << ","
            << delta_auc << "\n";

        all_ious_update.insert(all_ious_update.end(), update_metrics.ious.begin(), update_metrics.ious.end());
        all_errors_update.insert(all_errors_update.end(), update_metrics.errors.begin(), update_metrics.errors.end());
        all_ious_pure.insert(all_ious_pure.end(), pure_metrics.ious.begin(), pure_metrics.ious.end());
        all_errors_pure.insert(all_errors_pure.end(), pure_metrics.errors.begin(), pure_metrics.errors.end());
        total_seconds_update += update_metrics.tracking_seconds;
        total_seconds_pure += pure_metrics.tracking_seconds;
        total_frames += seq.frames.size();
    }

    const auto overall_update = ComputeMetrics(all_ious_update, all_errors_update, total_seconds_update);
    const auto overall_pure = ComputeMetrics(all_ious_pure, all_errors_pure, total_seconds_pure);
    const double overall_delta_auc = overall_update.auc - overall_pure.auc;

    out << "OVERALL," << total_frames << ","
        << std::fixed << std::setprecision(4)
        << overall_update.auc << "," << overall_pure.auc << ","
        << overall_update.success50 << "," << overall_pure.success50 << ","
        << overall_update.precision20 << "," << overall_pure.precision20 << ","
        << overall_update.fps << "," << overall_pure.fps << ","
        << overall_delta_auc << "\n";

    std::cout << "\nWrote comparison CSV to: " << output_path << "\n";
    std::cout << "Overall AUC (update) = " << std::fixed << std::setprecision(4) << overall_update.auc
              << ", pure = " << overall_pure.auc
              << ", delta = " << overall_delta_auc << "\n";

    return 0;
}
