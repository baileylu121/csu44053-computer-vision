#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <print>
#include <ranges>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace views = std::views;

namespace {

template <typename T>
using Result = std::expected<T, std::string>;

template <typename Function>
class ForEach : public std::ranges::range_adaptor_closure<ForEach<Function>> {
public:
    explicit ForEach(Function function) :
        function_(std::move(function)) {}

    void operator()(std::ranges::input_range auto&& range) const {
        std::ranges::for_each(range, function_);
    }

private:
    Function function_;
};

template <typename Function>
ForEach<Function> forEach(Function function) {
    return ForEach<Function>(std::move(function));
}

struct BallLabel {
    int number = 0;
    cv::Point2f position;
};

using TableCorners = std::array<cv::Point2f, 4>;
constexpr std::array<const char*, 4> corner_names = {
    "top_left", "bottom_left", "top_right", "bottom_right"
};

struct AnnotatedImage {
    cv::Mat image;
    cv::Size table_size;
    TableCorners corners;
    std::vector<BallLabel> balls;

    [[nodiscard]] std::vector<cv::Point2f> mapToImage(std::vector<cv::Point2f> points) const {
        const auto cols = static_cast<float>(table_size.width - 1);
        const auto rows = static_cast<float>(table_size.height - 1);
        const TableCorners table_corners = {
            cv::Point2f{0, 0}, cv::Point2f{0, rows}, cv::Point2f{cols, 0}, cv::Point2f{cols, rows}
        };
        const cv::Mat transform = cv::getPerspectiveTransform(table_corners.data(), corners.data());
        cv::perspectiveTransform(points, points, transform);
        return points;
    }
};

struct ImageLabels {
    std::string name;
    std::vector<BallLabel> balls;
};

struct Annotations {
    cv::Size table_size;
    TableCorners corners;
    std::vector<ImageLabels> images;
};

struct SourceImage {
    std::string name;
    AnnotatedImage annotated;
};

using DistortFunction = void (*)(AnnotatedImage&, double);

struct Distortion {
    std::string name;
    std::vector<double> strengths;
    DistortFunction distort = nullptr;
};

struct DistortionLevel {
    const Distortion* distortion = nullptr;
    int number = 0;
    double strength = 0.0;
};

struct DistortedSample {
    const SourceImage* source = nullptr;
    const DistortionLevel* level = nullptr;
    AnnotatedImage annotated;
};

constexpr const char* argument_spec =
    "{@annotations | data/ground-truth.yml    | annotation file}"
    "{@images      | OpenCVExample/Media/Pool | directory of table images}"
    "{@output      | data/augmented           | output directory}"
    "{help h       |                          | show this message}";

struct Arguments {
    fs::path annotation_file;
    fs::path image_dir;
    fs::path output_dir;
};

Result<Arguments> parseArguments(const cv::CommandLineParser& parser);
Result<Annotations> loadAnnotations(const fs::path& path);
Result<std::vector<SourceImage>>
loadTableImages(const Annotations& annotations, const fs::path& image_dir);
Result<void> distortAndSave(const std::vector<SourceImage>& sources, const fs::path& output_dir);

Result<void> generateDataset(const Arguments& arguments) {
    return loadAnnotations(arguments.annotation_file)
        .and_then([&](const Annotations& annotations) {
            return loadTableImages(annotations, arguments.image_dir);
        })
        .and_then([&](const std::vector<SourceImage>& sources) {
            return distortAndSave(sources, arguments.output_dir);
        });
}

} // namespace

int main(int argc, char** argv) {
    try {
        cv::CommandLineParser parser(argc, argv, argument_spec);
        parser.about("Generates distorted variants of the table images for robustness testing.");
        if (parser.has("help")) {
            parser.printMessage();
            return 0;
        }
        const auto result = parseArguments(parser).and_then(generateDataset);
        if (!result) {
            std::println(stderr, "{}", result.error());
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::fputs(error.what(), stderr);
        std::fputs("\n", stderr);
        return 1;
    }
}

namespace {

Result<Arguments> parseArguments(const cv::CommandLineParser& parser) {
    Arguments arguments{
        .annotation_file = parser.get<std::string>("@annotations"),
        .image_dir = parser.get<std::string>("@images"),
        .output_dir = parser.get<std::string>("@output")
    };
    if (!parser.check()) {
        return std::unexpected("invalid arguments; run with --help for usage");
    }
    return arguments;
}

auto intoYamlListRange(const cv::FileNode& node) {
    return views::iota(0, static_cast<int>(node.size()))
        | views::transform([node](int i) { return node[i]; });
}

cv::Point2f readPosition(const cv::FileNode& node) {
    return {static_cast<float>(node["col"]), static_cast<float>(node["row"])};
}

BallLabel readBallLabel(const cv::FileNode& node) {
    return {.number = static_cast<int>(node["number"]), .position = readPosition(node)};
}

ImageLabels readImageLabels(const cv::FileNode& node) {
    return {
        .name = node["name"],
        .balls = intoYamlListRange(node["balls"])
            | views::transform(readBallLabel)
            | std::ranges::to<std::vector>()
    };
}

Result<Annotations> loadAnnotations(const fs::path& path) {
    const cv::FileStorage file(path.string(), cv::FileStorage::READ);
    if (!file.isOpened()) {
        return std::unexpected("could not read " + path.string());
    }
    Annotations annotations{
        .table_size = {static_cast<int>(file["table_cols"]), static_cast<int>(file["table_rows"])},
        .corners = {},
        .images = intoYamlListRange(file["images"])
            | views::transform(readImageLabels)
            | std::ranges::to<std::vector>()
    };
    std::ranges::transform(corner_names, annotations.corners.begin(), [&](const char* name) {
        return readPosition(file["corners"][name]);
    });
    return annotations;
}

SourceImage
loadSource(const ImageLabels& labels, const Annotations& annotations, const fs::path& image_dir) {
    return {
        .name = labels.name,
        .annotated = {
            .image = cv::imread((image_dir / labels.name).string()),
            .table_size = annotations.table_size,
            .corners = annotations.corners,
            .balls = labels.balls
        }
    };
}

Result<std::vector<SourceImage>>
loadTableImages(const Annotations& annotations, const fs::path& image_dir) {
    auto sources = annotations.images
        | views::filter([](const ImageLabels& labels) { return labels.name.starts_with("Table"); })
        | views::transform([&](const ImageLabels& labels) {
                       return loadSource(labels, annotations, image_dir);
                   })
        | std::ranges::to<std::vector>();
    const auto unreadable = std::ranges::find_if(sources, [](const SourceImage& source) {
        return source.annotated.image.empty();
    });
    if (unreadable != sources.end()) {
        return std::unexpected("could not read " + (image_dir / unreadable->name).string());
    }
    return sources;
}

void scaleBrightness(cv::Mat& image, const cv::Mat& factor) {
    cv::Mat image_float;
    cv::Mat factor_bgr;
    image.convertTo(image_float, CV_32FC3);
    cv::cvtColor(factor, factor_bgr, cv::COLOR_GRAY2BGR);
    cv::Mat(image_float.mul(factor_bgr)).convertTo(image, CV_8UC3);
}

void blendTowardsWhite(cv::Mat& image, const cv::Mat& amount) {
    cv::Mat image_float;
    cv::Mat amount_bgr;
    image.convertTo(image_float, CV_32FC3);
    cv::cvtColor(amount, amount_bgr, cv::COLOR_GRAY2BGR);
    const cv::Mat white(image.size(), CV_32FC3, cv::Scalar::all(255));
    cv::Mat(image_float + amount_bgr.mul(white - image_float)).convertTo(image, CV_8UC3);
}

void applyGamma(AnnotatedImage& annotated, double strength) {
    const auto lut = views::iota(0, 256)
        | views::transform([strength](int value) {
                         return cv::saturate_cast<uchar>(255.0 * std::pow(value / 255.0, strength));
                     })
        | std::ranges::to<std::vector>();
    cv::LUT(annotated.image, lut, annotated.image);
}

void castColour(AnnotatedImage& annotated, double strength) {
    cv::multiply(annotated.image, cv::Scalar(1.0 - strength, 1.0, 1.0 + strength), annotated.image);
}

void reduceSaturation(AnnotatedImage& annotated, double strength) {
    cv::Mat grey;
    cv::cvtColor(annotated.image, grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grey, grey, cv::COLOR_GRAY2BGR);
    cv::addWeighted(annotated.image, strength, grey, 1.0 - strength, 0.0, annotated.image);
}

void addLightingGradient(AnnotatedImage& annotated, double strength) {
    const int cols = annotated.image.cols;
    const auto ramp = views::iota(0, cols)
        | views::transform([&](int col) {
                          return static_cast<float>(1.0 - ((strength * col) / (cols - 1)));
                      })
        | std::ranges::to<std::vector>();
    scaleBrightness(annotated.image, cv::repeat(cv::Mat(ramp).t(), annotated.image.rows, 1));
}

void addSensorNoise(AnnotatedImage& annotated, double strength) {
    constexpr int noise_seed = 44053;
    cv::RNG rng(noise_seed);
    cv::Mat image_float;
    annotated.image.convertTo(image_float, CV_32FC3);
    cv::Mat noise(annotated.image.size(), CV_32FC3);
    rng.fill(noise, cv::RNG::NORMAL, 0.0, strength);
    cv::Mat(image_float + noise).convertTo(annotated.image, CV_8UC3);
}

void blurDefocus(AnnotatedImage& annotated, double strength) {
    cv::GaussianBlur(annotated.image, annotated.image, {}, strength);
}

void reduceResolution(AnnotatedImage& annotated, double strength) {
    const cv::Size size = annotated.image.size();
    cv::Mat small;
    cv::resize(annotated.image, small, {}, strength, strength, cv::INTER_AREA);
    cv::resize(small, annotated.image, size, 0, 0, cv::INTER_LINEAR);
}

void offsetCorners(AnnotatedImage& annotated, double strength) {
    const auto offset = static_cast<float>(strength);
    const TableCorners outwards = {
        cv::Point2f{-offset, -offset},
        cv::Point2f{-offset, offset},
        cv::Point2f{offset, -offset},
        cv::Point2f{offset, offset}
    };
    std::ranges::transform(annotated.corners, outwards, annotated.corners.begin(), std::plus{});
}

const std::vector<cv::Point2f> shadow_outline = {{900, 0}, {1250, 0}, {1100, 682}, {750, 682}};

void castShadow(AnnotatedImage& annotated, double strength) {
    const auto outline = annotated.mapToImage(shadow_outline)
        | views::transform([](cv::Point2f point) { return cv::Point(point); })
        | std::ranges::to<std::vector>();
    cv::Mat mask = cv::Mat::zeros(annotated.image.size(), CV_32F);
    cv::fillConvexPoly(mask, outline, cv::Scalar(1.0), cv::LINE_AA);
    cv::GaussianBlur(mask, mask, {}, 20.0);
    scaleBrightness(annotated.image, 1.0 - (mask * strength));
}

const std::vector<cv::Point2f> glare_centres = {
    {200, 150}, {550, 500}, {900, 250}, {1200, 450}, {1350, 120}
};

void addGlare(AnnotatedImage& annotated, double strength) {
    cv::Mat mask = cv::Mat::zeros(annotated.image.size(), CV_32F);
    annotated.mapToImage(glare_centres) | forEach([&](cv::Point2f centre) {
        cv::ellipse(mask, centre, {30, 12}, 20.0, 0.0, 360.0, cv::Scalar(strength), cv::FILLED);
    });
    cv::GaussianBlur(mask, mask, {}, 8.0);
    blendTowardsWhite(annotated.image, mask);
}

const std::vector<Distortion>& listDistortions() {
    static const std::vector<Distortion> all = {
        {.name = "gamma", .strengths = {0.5, 0.7, 1.5, 2.0}, .distort = applyGamma},
        {.name = "colour-cast", .strengths = {-0.35, -0.2, 0.2, 0.35}, .distort = castColour},
        {.name = "reduced-saturation", .strengths = {0.7, 0.4, 0.15}, .distort = reduceSaturation},
        {.name = "lighting-gradient", .strengths = {0.3, 0.5, 0.7}, .distort = addLightingGradient},
        {.name = "sensor-noise", .strengths = {5, 15, 30}, .distort = addSensorNoise},
        {.name = "defocus-blur", .strengths = {1.5, 3, 5}, .distort = blurDefocus},
        {.name = "resolution-loss", .strengths = {0.5, 0.33, 0.2}, .distort = reduceResolution},
        {.name = "corner-offset", .strengths = {3, 6, 12}, .distort = offsetCorners},
        {.name = "cast-shadow", .strengths = {0.35, 0.5, 0.65}, .distort = castShadow},
        {.name = "glare", .strengths = {0.4, 0.6, 0.8}, .distort = addGlare},
    };
    return all;
}

std::vector<DistortionLevel> listDistortionLevels() {
    return listDistortions()
        | views::transform([](const Distortion& distortion) {
               return views::enumerate(distortion.strengths)
                   | views::transform([pointer = &distortion](const auto& item) {
                          const auto [index, strength] = item;
                          return DistortionLevel{
                              .distortion = pointer,
                              .number = static_cast<int>(index) + 1,
                              .strength = strength
                          };
                      });
           })
        | views::join
        | std::ranges::to<std::vector>();
}

DistortedSample distortSource(const SourceImage& source, const DistortionLevel& level) {
    AnnotatedImage annotated = source.annotated;
    annotated.image = source.annotated.image.clone();
    level.distortion->distort(annotated, level.strength);
    return {.source = &source, .level = &level, .annotated = std::move(annotated)};
}

constexpr std::string_view yaml_header = R"(%YAML:1.0
---
source: "{}"
distortion: {}
level: {}
strength: {}
corners:)";

auto formatYaml(const DistortedSample& sample) {
    const DistortionLevel& level = *sample.level;
    const std::string header = std::format(
        yaml_header, sample.source->name, level.distortion->name, level.number, level.strength
    );
    return views::concat(
        views::single(header),
        views::zip(corner_names, sample.annotated.corners) | views::transform([](const auto& pair) {
            const auto& [name, corner] = pair;
            return std::format("  {}: {{ row: {}, col: {} }}", name, corner.y, corner.x);
        }),
        views::single(std::string("balls:")),
        sample.annotated.balls | views::transform([](const BallLabel& ball) {
            return std::format(
                "  - {{ number: {}, row: {}, col: {} }}",
                ball.number,
                ball.position.y,
                ball.position.x
            );
        })
    );
}

std::string formatName(const DistortedSample& sample) {
    return std::format(
        "{}-{}-{}",
        fs::path(sample.source->name).stem().string(),
        sample.level->distortion->name,
        sample.level->number
    );
}

void saveSample(const fs::path& output_dir, const DistortedSample& sample) {
    constexpr int jpeg_quality = 95;
    const std::string stem = formatName(sample);

    cv::imwrite(
        (output_dir / (stem + ".jpeg")).string(),
        sample.annotated.image,
        {cv::IMWRITE_JPEG_QUALITY, jpeg_quality}
    );

    std::ofstream out(output_dir / (stem + ".yml"));
    formatYaml(sample) | forEach([&](const std::string& line) { std::println(out, "{}", line); });
}

Result<void> distortAndSave(const std::vector<SourceImage>& sources, const fs::path& output_dir) {
    std::error_code error;
    fs::create_directories(output_dir, error);
    if (error) {
        return std::unexpected("could not create " + output_dir.string() + ": " + error.message());
    }

    const auto levels = listDistortionLevels();
    views::cartesian_product(sources, levels)
        | views::transform([](const auto& pair) { return std::apply(distortSource, pair); })
        | forEach([&](const DistortedSample& sample) { saveSample(output_dir, sample); });

    std::println("wrote {} samples to {}", sources.size() * levels.size(), output_dir.string());

    return {};
}

} // namespace
