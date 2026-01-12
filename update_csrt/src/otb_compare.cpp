#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>

#include <array>
#include <algorithm>
#include <chrono>
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
#include <unordered_map>
#include <vector>

#include "../inc/csrt_tracker.hpp"

namespace fs = std::filesystem;

namespace {

const std::vector<std::string> kDebugSequences = {"Car4", "BlurCar1", "Football1", "Deer"};

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
    std::cout << "Usage: otb_compare --dataset-root <path> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --output <csv>       Output CSV file (default: auc_compare.csv)\n";
    std::cout << "  --max-frames <n>     Limit frames per sequence\n";
    std::cout << "  --pure-csv <csv>     Load OpenCV CSRT baseline from CSV (optional, for fast mode)\n";
    std::cout << "  --kf-mode <n>        KF/CSRT fusion mode (1,2,3)\n";
    std::cout << "  --kf-prior-mode <n>  KF prior mode (0=always,1=prev-PSR low)\n";
    std::cout << "  --model-lr-mode <n>  Model update mode (0=always,1=psr,2=psr-soft,3=psr+innov)\n";
    std::cout << "  --psr-th <val>       PSR threshold for gating (default: 0.035)\n";
    std::cout << "  --kf-r-min <val>     KF measurement noise min\n";
    std::cout << "  --kf-r-max <val>     KF measurement noise max\n";
    std::cout << "  --kf-q-pos <val>     KF process noise (position)\n";
    std::cout << "  --kf-q-vel <val>     KF process noise (velocity)\n";
    std::cout << "  --kf-innov-hard <v>  KF innovation threshold scale\n";
    std::cout << "  --search-scale-max <val>  Max search scale factor\n";
    std::cout << "  --threads <n>        Override number of worker threads\n";
    std::cout << "  --help               Show this help\n";
    std::cout << "\nDefault: Run both update_csrt and OpenCV CSRT for comparison\n";
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

double SuccessRate(const std::vector<double> &ious, double threshold) {
    const auto ok = std::count_if(ious.begin(), ious.end(), [threshold](double iou) {
        return iou >= threshold;
    });
    return ious.empty() ? 0.0 : static_cast<double>(ok) / static_cast<double>(ious.size());
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
    metrics.success50 = SuccessRate(ious, 0.5);
    metrics.precision20 = PrecisionAt(errors, 20.0);
    metrics.fps = tracking_seconds > 0.0 ? static_cast<double>(ious.size()) / tracking_seconds : 0.0;
    return metrics;
}

struct BaselineMetrics {
    std::size_t frames{0};
    double auc{0.0};
    double success50{0.0};
    double precision20{0.0};
    double fps{0.0};
};

struct BaselineData {
    std::unordered_map<std::string, BaselineMetrics> per_sequence;
    std::optional<BaselineMetrics> overall;
};

std::vector<std::string> SplitCsvLine(const std::string &line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    return tokens;
}

std::optional<BaselineData> LoadBaselineCsv(const fs::path &csv_path) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        std::cerr << "Failed to open baseline CSV: " << csv_path << "\n";
        return std::nullopt;
    }

    BaselineData data;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (first) {
            first = false;
            if (line.find("sequence") != std::string::npos) {
                continue;
            }
        }
        const auto tokens = SplitCsvLine(line);
        if (tokens.size() < 6) {
            continue;
        }
        const std::string &name = tokens[0];
        BaselineMetrics m;
        m.frames = static_cast<std::size_t>(std::stoull(tokens[1]));
        m.auc = std::stod(tokens[2]);
        m.success50 = std::stod(tokens[3]);
        m.precision20 = std::stod(tokens[4]);
        m.fps = std::stod(tokens[5]);
        if (name == "OVERALL") {
            data.overall = m;
        } else {
            data.per_sequence[name] = m;
        }
    }
    if (data.per_sequence.empty()) {
        std::cerr << "Baseline CSV has no per-sequence entries: " << csv_path << "\n";
        return std::nullopt;
    }
    return data;
}

TrackerMetrics MetricsFromBaseline(const BaselineMetrics &baseline) {
    TrackerMetrics metrics;
    metrics.auc = baseline.auc;
    metrics.success50 = baseline.success50;
    metrics.precision20 = baseline.precision20;
    metrics.fps = baseline.fps;
    return metrics;
}

TrackerMetrics EvaluateUpdateOnly(const SequenceData &data, const csrt::CsrtParams &params,
    const std::optional<int> &max_frames) {
    TrackerMetrics metrics;
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

    csrt::CsrtTracker update_tracker(params);
    if (!update_tracker.Init(first, init_rect)) {
        return metrics;
    }

    const std::size_t limit = max_frames
        ? std::min<std::size_t>(*max_frames, data.frames.size())
        : data.frames.size();

    std::vector<double> ious_update;
    std::vector<double> errors_update;
    ious_update.reserve(limit);
    errors_update.reserve(limit);

    double update_seconds = 0.0;

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

        std::optional<BoundingBox> pred_update;
        if (ok_update) {
            pred_update = BoundingBox{static_cast<double>(bbox_update.x), static_cast<double>(bbox_update.y),
                static_cast<double>(bbox_update.width), static_cast<double>(bbox_update.height)};
        }

        const auto &gt = data.ground_truth[i];
        const double iou_update = pred_update ? pred_update->iou(gt) : 0.0;
        const double error_update = pred_update ? CenterError(*pred_update, gt) : std::numeric_limits<double>::infinity();

        ious_update.push_back(iou_update);
        errors_update.push_back(error_update);
    }

    return ComputeMetrics(ious_update, errors_update, update_seconds);
}

TrackerPairMetrics EvaluateTrackers(const SequenceData &data, const csrt::CsrtParams &params,
    const std::optional<int> &max_frames) {
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

    csrt::CsrtTracker update_tracker(params);
    if (!update_tracker.Init(first, init_rect)) {
        return metrics;
    }

    auto pure_tracker = cv::TrackerCSRT::create();
    pure_tracker->init(first, init_rect);

    const std::size_t limit = max_frames
        ? std::min<std::size_t>(*max_frames, data.frames.size())
        : data.frames.size();

    const bool debug_sequence = std::find(kDebugSequences.begin(), kDebugSequences.end(),
        data.name) != kDebugSequences.end();
    std::ofstream debug_out;
    if (debug_sequence) {
        fs::path debug_path = fs::path("update_csrt") / ("debug_" + data.name + ".csv");
        debug_out.open(debug_path);
        if (debug_out.is_open()) {
            debug_out << "frame,gt_x,gt_y,gt_w,gt_h,"
                      << "upd_x,upd_y,upd_w,upd_h,upd_iou,upd_err,"
                      << "pure_x,pure_y,pure_w,pure_h,pure_iou,pure_err,"
                      << "apce,apce_norm,kf_trace,kf_uncert,kf_r,accept,"
                      << "meas_x,meas_y,kf_pred_x,kf_pred_y,kf_corr_x,kf_corr_y,"
                      << "innov,innov_thresh,search_scale\n";
        }
    }

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

        if (debug_out.is_open()) {
            debug_out << i << ","
                      << gt.x << "," << gt.y << "," << gt.width << "," << gt.height << ","
                      << (pred_update ? pred_update->x : 0.0) << ","
                      << (pred_update ? pred_update->y : 0.0) << ","
                      << (pred_update ? pred_update->width : 0.0) << ","
                      << (pred_update ? pred_update->height : 0.0) << ","
                      << iou_update << "," << error_update << ","
                      << (pred_pure ? pred_pure->x : 0.0) << ","
                      << (pred_pure ? pred_pure->y : 0.0) << ","
                      << (pred_pure ? pred_pure->width : 0.0) << ","
                      << (pred_pure ? pred_pure->height : 0.0) << ","
                      << iou_pure << "," << error_pure << ","
                      << update_tracker.GetLastApce() << ","
                      << update_tracker.GetLastApceNorm() << ","
                      << update_tracker.GetLastKfTrace() << ","
                      << update_tracker.GetLastKfUncert() << ","
                      << update_tracker.GetLastKfR() << ","
                      << (update_tracker.GetLastMeasurementAccepted() ? 1 : 0) << ","
                      << update_tracker.GetLastMeasuredCenter().x << ","
                      << update_tracker.GetLastMeasuredCenter().y << ","
                      << update_tracker.GetLastKfPredCenter().x << ","
                      << update_tracker.GetLastKfPredCenter().y << ","
                      << update_tracker.GetLastKfCorrectedCenter().x << ","
                      << update_tracker.GetLastKfCorrectedCenter().y << ","
                      << update_tracker.GetLastKfInnov() << ","
                      << update_tracker.GetLastKfInnovThresh() << ","
                      << update_tracker.GetSearchScaleFactor()
                      << "\n";
        }
    }

    metrics.update = ComputeMetrics(ious_update, errors_update, update_seconds);
    metrics.pure = ComputeMetrics(ious_pure, errors_pure, pure_seconds);
    return metrics;
}

}  // namespace

int main(int argc, char **argv) {
    // Initialize GPU/CUDA support
    std::cout << "=== GPU/CUDA Configuration ===" << std::endl;
    
    // Check CUDA availability
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_devices > 0) {
        std::cout << "CUDA devices found: " << cuda_devices << std::endl;
        cv::cuda::DeviceInfo dev_info(0);
        std::cout << "  Device 0: " << dev_info.name() << std::endl;
        std::cout << "  Compute capability: " << dev_info.majorVersion() << "." << dev_info.minorVersion() << std::endl;
        std::cout << "  Total memory: " << (dev_info.totalMemory() / (1024*1024)) << " MB" << std::endl;
        cv::cuda::setDevice(0);
        std::cout << "  Using CUDA device 0" << std::endl;
    } else {
        std::cout << "No CUDA devices found" << std::endl;
    }
    
    // Check OpenCL availability
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        if (cv::ocl::useOpenCL()) {
            std::cout << "OpenCL enabled: " << cv::ocl::Device::getDefault().name() << std::endl;
        }
    } else {
        std::cout << "OpenCL not available" << std::endl;
    }
    
    std::cout << "==============================" << std::endl << std::endl;

    fs::path dataset_root;
    fs::path output_path = "auc_compare.csv";  // Write to current directory
    std::optional<int> max_frames;
    std::optional<fs::path> pure_csv_path = std::nullopt;  // Default: run both trackers
    int kf_mode = 1;
    int kf_prior_mode = 0;
    int model_lr_mode = 1;
    std::optional<unsigned int> thread_override;
    std::optional<float> psr_threshold_override;
    std::optional<float> kf_r_min_override;
    std::optional<float> kf_r_max_override;
    std::optional<float> kf_q_pos_override;
    std::optional<float> kf_q_vel_override;
    std::optional<float> kf_innov_hard_override;
    std::optional<float> search_scale_max_override;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset-root" && i + 1 < argc) {
            dataset_root = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--pure-csv" && i + 1 < argc) {
            pure_csv_path = argv[++i];
        } else if (arg == "--kf-mode" && i + 1 < argc) {
            kf_mode = std::stoi(argv[++i]);
        } else if (arg == "--kf-prior-mode" && i + 1 < argc) {
            kf_prior_mode = std::stoi(argv[++i]);
        } else if (arg == "--model-lr-mode" && i + 1 < argc) {
            model_lr_mode = std::stoi(argv[++i]);
        } else if (arg == "--psr-th" && i + 1 < argc) {
            psr_threshold_override = std::stof(argv[++i]);
        } else if (arg == "--kf-r-min" && i + 1 < argc) {
            kf_r_min_override = std::stof(argv[++i]);
        } else if (arg == "--kf-r-max" && i + 1 < argc) {
            kf_r_max_override = std::stof(argv[++i]);
        } else if (arg == "--kf-q-pos" && i + 1 < argc) {
            kf_q_pos_override = std::stof(argv[++i]);
        } else if (arg == "--kf-q-vel" && i + 1 < argc) {
            kf_q_vel_override = std::stof(argv[++i]);
        } else if (arg == "--kf-innov-hard" && i + 1 < argc) {
            kf_innov_hard_override = std::stof(argv[++i]);
        } else if (arg == "--search-scale-max" && i + 1 < argc) {
            search_scale_max_override = std::stof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            thread_override = static_cast<unsigned int>(std::stoul(argv[++i]));
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

    if (kf_mode < 1 || kf_mode > 3) {
        std::cerr << "Invalid --kf-mode: " << kf_mode << " (expected 1,2,3)\n";
        return 1;
    }

    if (kf_prior_mode < 0 || kf_prior_mode > 1) {
        std::cerr << "Invalid --kf-prior-mode: " << kf_prior_mode << " (expected 0 or 1)\n";
        return 1;
    }

    if (model_lr_mode < 0 || model_lr_mode > 3) {
        std::cerr << "Invalid --model-lr-mode: " << model_lr_mode << " (expected 0..3)\n";
        return 1;
    }

    csrt::CsrtParams params;
    params.kf_mode = kf_mode;
    params.kf_prior_mode = kf_prior_mode;
    params.model_lr_mode = model_lr_mode;
    if (psr_threshold_override) {
        params.psr_threshold = *psr_threshold_override;
    }
    if (kf_r_min_override) {
        params.kf_r_min = *kf_r_min_override;
    }
    if (kf_r_max_override) {
        params.kf_r_max = *kf_r_max_override;
    }
    if (kf_q_pos_override) {
        params.kf_q_pos = *kf_q_pos_override;
    }
    if (kf_q_vel_override) {
        params.kf_q_vel = *kf_q_vel_override;
    }
    if (kf_innov_hard_override) {
        params.kf_innov_hard_scale = *kf_innov_hard_override;
    }
    if (search_scale_max_override) {
        params.search_scale_max = *search_scale_max_override;
    }

    const auto sequences = LoadSequences(dataset_root);
    if (sequences.empty()) {
        std::cerr << "No valid sequences found under: " << dataset_root << "\n";
        return 1;
    }

    std::optional<BaselineData> baseline;
    if (pure_csv_path) {
        if (!fs::exists(*pure_csv_path)) {
            std::cout << "Warning: Baseline CSV not found: " << *pure_csv_path << "\n";
            std::cout << "Falling back to running both trackers...\n\n";
            pure_csv_path = std::nullopt;
        } else {
            baseline = LoadBaselineCsv(*pure_csv_path);
            if (!baseline) {
                std::cout << "Warning: Failed to load baseline CSV, running both trackers...\n\n";
                pure_csv_path = std::nullopt;
            } else {
                bool all_found = true;
                for (const auto &seq : sequences) {
                    if (baseline->per_sequence.find(seq.name) == baseline->per_sequence.end()) {
                        std::cerr << "Warning: Baseline CSV missing sequence: " << seq.name << "\n";
                        all_found = false;
                    }
                }
                if (!all_found || !baseline->overall) {
                    std::cout << "Warning: Baseline incomplete, running both trackers...\n\n";
                    baseline = std::nullopt;
                    pure_csv_path = std::nullopt;
                } else {
                    std::cout << "Using baseline from: " << *pure_csv_path << "\n";
                    std::cout << "Only running update_csrt (faster mode)...\n\n";
                    if (max_frames) {
                        std::cout << "Warning: --max-frames with --pure-csv may be inconsistent with baseline.\n";
                    }
                }
            }
        }
    }
    
    if (!pure_csv_path) {
        std::cout << "Running both update_csrt and OpenCV CSRT for comparison...\n\n";
    }
    
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << "\n";
        return 1;
    }

    out << "sequence,frames,auc_update,auc_pure,success50_update,success50_pure,precision20_update,precision20_pure,fps_update,fps_pure,delta_auc\n";

    std::vector<double> all_ious_update;
    std::vector<double> all_errors_update;
    double total_seconds_update = 0.0;
    std::size_t total_frames = 0;

    std::cout << "Comparing trackers on " << sequences.size() << " sequence(s)...\n";
    std::cout << "KF mode: " << kf_mode << "\n";
    std::cout << "KF prior mode: " << kf_prior_mode << "\n";
    std::cout << "Model LR mode: " << model_lr_mode << "\n";
    std::cout << "PSR th: " << params.psr_threshold
              << " | KF R: [" << params.kf_r_min << ", " << params.kf_r_max << "]"
              << " | KF Q: [" << params.kf_q_pos << ", " << params.kf_q_vel << "]"
              << " | KF innov hard: " << params.kf_innov_hard_scale
              << " | search_scale_max: " << params.search_scale_max << "\n";

    // Parallel processing
    unsigned int num_threads = thread_override.value_or(std::thread::hardware_concurrency());
    if (num_threads == 0) {
        num_threads = 1;
    }
    std::cout << "Using " << num_threads << " threads for parallel processing\n\n";
    
    std::vector<TrackerPairMetrics> all_metrics(sequences.size());
    std::mutex cout_mutex;
    
    auto process_batch = [&](std::size_t start_idx, std::size_t end_idx) {
        for (std::size_t i = start_idx; i < end_idx; ++i) {
            const auto &seq = sequences[i];
            
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "  [Thread " << std::this_thread::get_id() << "] " 
                          << seq.name << " (" << seq.frames.size() << " frames)" << std::endl;
            }
            
            if (baseline) {
                all_metrics[i].update = EvaluateUpdateOnly(seq, params, max_frames);
                const auto &base_metrics = baseline->per_sequence.at(seq.name);
                all_metrics[i].pure = MetricsFromBaseline(base_metrics);
            } else {
                all_metrics[i] = EvaluateTrackers(seq, params, max_frames);
            }
        }
    };
    
    std::vector<std::thread> threads;
    const std::size_t batch_size = (sequences.size() + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        std::size_t start_idx = t * batch_size;
        std::size_t end_idx = std::min(start_idx + batch_size, sequences.size());
        if (start_idx < sequences.size()) {
            threads.emplace_back(process_batch, start_idx, end_idx);
        }
    }
    
    for (auto &thread : threads) {
        thread.join();
    }
    
    std::cout << "\nWriting results...\n";
    
    for (std::size_t i = 0; i < sequences.size(); ++i) {
        const auto &seq = sequences[i];
        const auto &update_metrics = all_metrics[i].update;
        const auto &pure_metrics = all_metrics[i].pure;

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
        total_seconds_update += update_metrics.tracking_seconds;
        total_frames += seq.frames.size();
    }

    const auto overall_update = ComputeMetrics(all_ious_update, all_errors_update, total_seconds_update);
    TrackerMetrics overall_pure;
    if (baseline) {
        overall_pure = MetricsFromBaseline(*baseline->overall);
    } else {
        std::vector<double> all_ious_pure;
        std::vector<double> all_errors_pure;
        double total_seconds_pure = 0.0;
        for (const auto &entry : all_metrics) {
            all_ious_pure.insert(all_ious_pure.end(), entry.pure.ious.begin(), entry.pure.ious.end());
            all_errors_pure.insert(all_errors_pure.end(), entry.pure.errors.begin(), entry.pure.errors.end());
            total_seconds_pure += entry.pure.tracking_seconds;
        }
        overall_pure = ComputeMetrics(all_ious_pure, all_errors_pure, total_seconds_pure);
    }
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
