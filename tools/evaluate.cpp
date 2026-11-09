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
            std::vector<AugmentedSample> samples;
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.path().extension() != ".yml") {
                    continue;
                }

                auto sample = readSample(entry.path());
                if (!sample) {
                    return std::unexpected(sample.error());
                }

                samples.push_back(std::move(*sample));
            }
            return samples;
        }
    }

    namespace labelled {
        constexpr int BALL_COUNT = 16;

        cv::Mat_<int> buildIdentificationConfusion(
            const table::Image& table,
            const measurement::AllBallsMeasurements& measurements
        ) {
            const auto positions = table.truth
                | views::transform(&table::Ball::position)
                | std::ranges::to<std::vector>();

            cv::Mat_<int> confusion = cv::Mat_<int>::zeros(BALL_COUNT, BALL_COUNT);
            std::ranges::for_each(
                views::zip(table.truth, detection::identifyBalls(table, positions, measurements)),
                [&](const auto& pair) {
                    const auto& [truth, identified] = pair;
                    ++confusion(truth.number, identified.number);
                }
            );

            return confusion;
        }

        cv::Mat_<int> sumConfusions(const auto& confusions) {
            return std::ranges::fold_left(
                confusions,
                cv::Mat_<int>(cv::Mat_<int>::zeros(BALL_COUNT, BALL_COUNT)),
                [](const cv::Mat_<int>& total, const cv::Mat_<int>& confusion) {
                    return cv::Mat_<int>(total + confusion);
                }
            );
        }

        std::string formatAccuracy(const cv::Mat_<int>& confusion) {
            const auto correct = static_cast<std::ptrdiff_t>(cv::trace(confusion)[0]);
            const auto total = static_cast<std::ptrdiff_t>(cv::sum(confusion)[0]);

            return std::format(
                "{}/{} ({})", correct, total, reporting::formatRatio(correct, total)
            );
        }

        std::string formatMatrixCell(int count) {
            return count == 0 ? std::string("  .") : std::format("{:>3}", count);
        }

        std::string formatMatrixRow(const cv::Mat_<int>& counts, int row) {
            return views::iota(0, BALL_COUNT)
                | views::transform([&](int col) { return formatMatrixCell(counts(row, col)); })
                | views::join
                | std::ranges::to<std::string>();
        }

        void printIdentificationMatrix(const cv::Mat_<int>& counts) {
            std::println("rows labelled, columns identified (0 is the cue ball):");
            std::println(
                "    {}",
                views::iota(0, BALL_COUNT)
                    | views::transform([](int number) { return std::format("{:>3}", number); })
                    | views::join
                    | std::ranges::to<std::string>()
            );
            std::ranges::for_each(views::iota(0, BALL_COUNT), [&](int row) {
                std::println("{:>3} {}", row, formatMatrixRow(counts, row));
            });
        }

        void printTables(
            const std::vector<table::Image>& tables,
            const measurement::AllBallsMeasurements& measurements
        ) {
            const cv::Mat_<int> confusion =
                sumConfusions(tables | views::transform([&](const table::Image& table) {
                                  return buildIdentificationConfusion(table, measurements);
                              }));

            std::println(
                "identification on labelled positions, Table1-6: {}", formatAccuracy(confusion)
            );
            printIdentificationMatrix(confusion);
        }
    }

    std::string formatSummary(const std::vector<scoring::Score>& scores) {
        return views::zip(scoring::MATCH_DISTANCES, scores)
            | views::transform([](const auto& pair) {
                   const auto& [distance, score] = pair;
                   return std::format(
                       "{:>4} px: F1 {}, identified {}",
                       distance,
                       reporting::formatF1(score),
                       reporting::formatRatio(score.identified, score.found)
                   );
               })
            | views::join_with(std::string_view("   "))
            | std::ranges::to<std::string>();
    }

    void printLevelSummary(
        const auto& level_samples,
        const measurement::AllBallsMeasurements& measurements
    ) {
        const auto scores = level_samples
            | views::transform([&](const auto& sample) {
                                const auto detected =
                                    detection::detectBalls(sample.table, measurements);
                                return scoring::scoreDistances(sample.table, detected);
                            })
            | std::ranges::to<std::vector>();

        const cv::Mat_<int> confusion = labelled::sumConfusions(
            level_samples | views::transform([&](const augmented::AugmentedSample& sample) {
                return labelled::buildIdentificationConfusion(sample.table, measurements);
            })
        );

        const augmented::AugmentedSample& first = *std::ranges::begin(level_samples);
        std::println(
            "{:<20} {}   {}   labelled {}",
            first.distortion,
            first.level,
            formatSummary(scoring::sumScores(scores)),
            labelled::formatAccuracy(confusion)
        );
    }

    auto identifyLevel(const augmented::AugmentedSample& sample) {
        return std::tie(sample.distortion, sample.level);
    }

    void evaluateSamples(
        std::vector<augmented::AugmentedSample> samples,
        const measurement::AllBallsMeasurements& measurements
    ) {
        std::println("loaded {} samples", samples.size());
        std::ranges::sort(samples, {}, identifyLevel);
        std::ranges::for_each(
            samples | views::chunk_by([](const auto& left, const auto& right) {
                return identifyLevel(left) == identifyLevel(right);
            }),
            [&](const auto& group) { printLevelSummary(group, measurements); }
        );
    }

    const fs::path POOL_DIRECTORY = "OpenCVExample" / table::PHOTO_DIRECTORY;

    Result<void> evaluateAugmented() {
        return table::loadImage(POOL_DIRECTORY, measurement::REFERENCE_PHOTO)
            .transform(measurement::measureAllBallsPhoto)
            .and_then([](const measurement::AllBallsMeasurements& measurements) {
                return table::loadImages(POOL_DIRECTORY, scoring::TABLE_PHOTOS)
                    .transform(std::bind_back(labelled::printTables, std::cref(measurements)))
                    .and_then([&] {
                        return augmented::loadSamples(augmented::SAMPLE_DIRECTORY)
                            .transform(std::bind_back(evaluateSamples, std::cref(measurements)));
                    });
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
