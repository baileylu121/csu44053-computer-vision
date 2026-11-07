#include "../myApplication.cpp" // NOLINT(bugprone-suspicious-include)

#include <exception>
#include <string_view>

namespace {
    namespace augmented {
        const fs::path SAMPLE_DIRECTORY = "data/augmented";

        struct AugmentedSample {
            table::Image table;
            std::string distortion;
            int level = 0;
        };

        constexpr std::array<const char*, 4> CORNER_NAMES =
            {"top_left", "bottom_left", "top_right", "bottom_right"};

        auto intoYamlListRange(const cv::FileNode& node) {
            return views::iota(0, static_cast<int>(node.size()))
                | views::transform([node](int i) { return node[i]; });
        }

        cv::Point2f readPosition(const cv::FileNode& node) {
            return {static_cast<float>(node["col"]), static_cast<float>(node["row"])};
        }

        table::Ball readBallLabel(const cv::FileNode& node) {
            return {.number = static_cast<int>(node["number"]), .position = readPosition(node)};
        }

        table::Corners readCorners(const cv::FileNode& node) {
            return std::apply(
                [&](auto... names) { return table::Corners{readPosition(node[names])...}; },
                CORNER_NAMES
            );
        }

        std::vector<table::Ball> readBallLabels(const cv::FileNode& node) {
            return intoYamlListRange(node)
                | views::transform(readBallLabel)
                | std::ranges::to<std::vector>();
        }

        Result<table::Image> readTable(const cv::FileStorage& file, const fs::path& yaml_path) {
            return table::readPhoto(fs::path(yaml_path).replace_extension(".jpeg"))
                .transform([&](const cv::Mat& photo) {
                    return table::Image{
                        .name = yaml_path.stem().string(),
                        .image = table::correctPerspective(photo, readCorners(file["corners"])),
                        .truth = readBallLabels(file["balls"])
                    };
                });
        }

        Result<AugmentedSample> readSample(const fs::path& yaml_path) {
            const cv::FileStorage file(yaml_path.string(), cv::FileStorage::READ);
            if (!file.isOpened()) {
                return std::unexpected("could not read " + yaml_path.string());
            }

            return readTable(file, yaml_path).transform([&](table::Image table) {
                return AugmentedSample{
                    .table = std::move(table),
                    .distortion = file["distortion"],
                    .level = static_cast<int>(file["level"])
                };
            });
        }

        Result<std::vector<AugmentedSample>> loadSamples(const fs::path& directory) {
            return fs::directory_iterator(directory)
                | views::transform([](const fs::directory_entry& entry) { return entry.path(); })
                | views::filter([](const fs::path& path) { return path.extension() == ".yml"; })
                | views::transform(readSample)
                | collectResults();
        }
    }

    std::string formatSummary(const std::vector<evaluation::Score>& scores) {
        return views::zip(evaluation::MATCH_DISTANCES, scores)
            | views::transform([](const auto& pair) {
                   const auto& [distance, score] = pair;
                   return std::format(
                       "{:>4} px: F1 {}, identified {}",
                       distance,
                       evaluation::formatF1(score),
                       evaluation::formatRatio(score.identified, score.found)
                   );
               })
            | views::join_with(std::string_view("   "))
            | std::ranges::to<std::string>();
    }

    std::vector<evaluation::Score> scoreSample(
        const augmented::AugmentedSample& sample,
        const detection::ReferenceColours& references
    ) {
        const auto detected = detection::detectBalls(sample.table, references);
        return evaluation::scoreDistances(sample.table, detected);
    }

    void printLevelSummary(
        const auto& level_samples,
        const detection::ReferenceColours& references
    ) {
        const auto scores = level_samples
            | views::transform([&](const auto& sample) { return scoreSample(sample, references); })
            | std::ranges::to<std::vector>();

        const augmented::AugmentedSample& first = *std::ranges::begin(level_samples);
        std::println(
            "{:<20} {}   {}",
            first.distortion,
            first.level,
            formatSummary(evaluation::sumScores(scores))
        );
    }

    auto identifyLevel(const augmented::AugmentedSample& sample) {
        return std::tie(sample.distortion, sample.level);
    }

    bool isSameLevel(
        const augmented::AugmentedSample& left,
        const augmented::AugmentedSample& right
    ) {
        return identifyLevel(left) == identifyLevel(right);
    }

    void evaluateSamples(
        std::vector<augmented::AugmentedSample> samples,
        const detection::ReferenceColours& references
    ) {
        std::println("loaded {} samples", samples.size());
        std::ranges::sort(samples, {}, identifyLevel);
        samples
            | views::chunk_by(isSameLevel)
            | forEach([&](const auto& group) { printLevelSummary(group, references); });
    }

    const fs::path POOL_DIRECTORY = "OpenCVExample" / table::PHOTO_DIRECTORY;

    Result<void> evaluateAugmented() {
        return table::loadImage(POOL_DIRECTORY, detection::REFERENCE_PHOTO)
            .transform(detection::measureReferences)
            .and_then([](const detection::ReferenceColours& references) {
                return augmented::loadSamples(augmented::SAMPLE_DIRECTORY)
                    .transform(std::bind_back(evaluateSamples, std::cref(references)));
            });
    }
}

int main() {
    try {
        const auto result = evaluateAugmented();
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
