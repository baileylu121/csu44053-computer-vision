#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <print>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    namespace views = std::views;

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

    template <typename Range>
    using CollectedResults =
        Result<std::vector<typename std::ranges::range_value_t<Range>::value_type>>;

    class CollectResults : public std::ranges::range_adaptor_closure<CollectResults> {
    public:
        template <std::ranges::input_range Range>
        CollectedResults<Range> operator()(Range&& range) const {
            auto results = std::forward<Range>(range) | std::ranges::to<std::vector>();
            const auto failure = std::ranges::find_if(results, [](const auto& result) {
                return !result.has_value();
            });

            if (failure != results.end()) {
                return std::unexpected(failure->error());
            }

            return results
                | views::transform([](auto& result) { return std::move(*result); })
                | std::ranges::to<std::vector>();
        }
    };

    CollectResults collectResults() {
        return {};
    }

    namespace table {
        const fs::path PHOTO_DIRECTORY = "Media/Pool";

        struct Ball {
            int number = 0;
            cv::Point2f position;
        };

        struct Photo {
            std::string name;
            std::vector<Ball> balls;
        };

        const cv::Size TABLE_SIZE = {1434, 683};

        using Corners = std::array<cv::Point2f, 4>;

        const Corners PHOTO_CORNERS = {
            cv::Point2f{337, 35},
            cv::Point2f{38, 472},
            cv::Point2f{1570, 21},
            cv::Point2f{1888, 451}
        };

        struct Image {
            std::string name;
            cv::Mat image;
            std::vector<Ball> truth;
        };

        Result<cv::Mat> readPhoto(const fs::path& path) {
            cv::Mat photo = cv::imread(path.string());

            if (photo.empty()) {
                return std::unexpected("could not read " + path.string());
            }

            return photo;
        }

        cv::Mat correctPerspective(const cv::Mat& photo, const Corners& corners) {
            const auto cols = static_cast<float>(TABLE_SIZE.width - 1);
            const auto rows = static_cast<float>(TABLE_SIZE.height - 1);

            const Corners table_corners = {
                cv::Point2f{0, 0},
                cv::Point2f{0, rows},
                cv::Point2f{cols, 0},
                cv::Point2f{cols, rows}
            };
            const cv::Mat transform =
                cv::getPerspectiveTransform(corners.data(), table_corners.data());

            cv::Mat corrected;
            cv::warpPerspective(photo, corrected, transform, TABLE_SIZE, cv::INTER_CUBIC);

            return corrected;
        }

        Result<Image> loadImage(const fs::path& directory, const Photo& photo_labels) {
            return readPhoto(directory / photo_labels.name).transform([&](const cv::Mat& photo) {
                return Image{
                    .name = photo_labels.name,
                    .image = correctPerspective(photo, PHOTO_CORNERS),
                    .truth = photo_labels.balls
                };
            });
        }

        Result<std::vector<Image>> loadImages(
            const fs::path& directory,
            const std::vector<Photo>& photos
        ) {
            return photos
                | views::transform(std::bind_front(loadImage, directory))
                | collectResults();
        }
    }

    namespace detection {
        const table::Photo REFERENCE_PHOTO = {
            .name = "AllBalls.jpeg",
            .balls = {
                {1, {908, 253}},
                {2, {959, 250}},
                {3, {1013, 255}},
                {4, {1072, 256}},
                {5, {1132, 254}},
                {6, {1184, 262}},
                {7, {1244, 256}},
                {8, {1307, 253}},
                {9, {901, 329}},
                {10, {961, 330}},
                {11, {1013, 335}},
                {12, {1067, 336}},
                {13, {1130, 334}},
                {14, {1187, 345}},
                {15, {1245, 342}},
                {0, {797, 293}}
            }
        };

        struct ReferenceColours {};

        ReferenceColours measureReferences([[maybe_unused]] const table::Image& all_balls) {
            return {};
        }

        std::vector<cv::Point2f> locateBalls(
            [[maybe_unused]] const table::Image& table,
            [[maybe_unused]] const ReferenceColours& references
        ) {
            return {};
        }

        std::vector<table::Ball> identifyBalls(
            [[maybe_unused]] const table::Image& table,
            const std::vector<cv::Point2f>& positions,
            [[maybe_unused]] const ReferenceColours& references
        ) {
            return positions
                | views::transform([](cv::Point2f position) {
                       return table::Ball{.number = 0, .position = position};
                   })
                | std::ranges::to<std::vector>();
        }

        std::vector<table::Ball> detectBalls(
            const table::Image& table,
            const ReferenceColours& references
        ) {
            return identifyBalls(table, locateBalls(table, references), references);
        }
    }

    namespace evaluation {
        const std::vector<table::Photo> TABLE_PHOTOS = {
            {.name = "Table1.jpeg",
             .balls =
                 {{1, {1092, 309}},
                  {2, {1212, 429}},
                  {3, {1131, 383}},
                  {4, {1050, 335}},
                  {5, {1211, 285}},
                  {6, {1170, 261}},
                  {7, {1173, 352}},
                  {8, {1132, 331}},
                  {9, {1210, 240}},
                  {10, {1170, 310}},
                  {11, {1169, 407}},
                  {12, {1135, 281}},
                  {13, {1212, 374}},
                  {14, {1091, 360}},
                  {15, {1213, 331}},
                  {0, {29, 288}}}},
            {.name = "Table2.jpeg",
             .balls =
                 {{1, {1103, 282}},
                  {2, {1224, 620}},
                  {3, {1121, 391}},
                  {4, {1046, 315}},
                  {5, {1305, 310}},
                  {6, {621, 202}},
                  {7, {1173, 361}},
                  {8, {1132, 331}},
                  {9, {725, 177}},
                  {10, {1269, 185}},
                  {11, {798, 665}},
                  {12, {438, 296}},
                  {13, {1235, 53}},
                  {14, {1090, 362}},
                  {15, {1205, 399}},
                  {0, {1169, 507}}}},
            {.name = "Table3.jpeg",
             .balls =
                 {{1, {908, 23}},
                  {2, {1224, 616}},
                  {3, {1199, 447}},
                  {4, {29, 336}},
                  {5, {1406, 305}},
                  {6, {621, 210}},
                  {7, {1294, 260}},
                  {8, {1252, 226}},
                  {9, {725, 177}},
                  {10, {1270, 171}},
                  {11, {798, 665}},
                  {12, {351, 249}},
                  {13, {1233, 53}},
                  {14, {901, 169}},
                  {15, {1201, 399}},
                  {0, {29, 434}}}},
            {.name = "Table4.jpeg",
             .balls =
                 {{1, {908, 23}},
                  {2, {1407, 546}},
                  {3, {1204, 444}},
                  {5, {1406, 305}},
                  {6, {621, 210}},
                  {7, {1294, 260}},
                  {8, {1252, 226}},
                  {9, {724, 181}},
                  {10, {1270, 171}},
                  {11, {798, 665}},
                  {12, {351, 249}},
                  {13, {1231, 58}},
                  {14, {901, 169}},
                  {15, {1201, 399}},
                  {0, {1392, 88}}}},
            {.name = "Table5.jpeg",
             .balls =
                 {{1, {23, 535}},
                  {2, {1407, 546}},
                  {3, {1204, 444}},
                  {5, {1406, 305}},
                  {6, {25, 131}},
                  {7, {1294, 260}},
                  {8, {1252, 226}},
                  {9, {72, 538}},
                  {10, {1270, 171}},
                  {11, {798, 665}},
                  {12, {351, 249}},
                  {14, {901, 169}},
                  {15, {1201, 399}},
                  {0, {26, 608}}}},
            {.name = "Table6.jpeg",
             .balls = {
                 {1, {23, 535}},
                 {2, {1364, 477}},
                 {3, {1204, 444}},
                 {5, {1406, 305}},
                 {6, {25, 131}},
                 {7, {1294, 260}},
                 {8, {1252, 226}},
                 {9, {72, 538}},
                 {10, {1270, 171}},
                 {12, {351, 249}},
                 {14, {901, 169}},
                 {15, {1201, 399}},
                 {0, {947, 629}}
             }}
        };

        constexpr std::array<float, 3> MATCH_DISTANCES = {5.0F, 8.0F, 11.4F};

        struct Score {
            std::ptrdiff_t labelled = 0;
            std::ptrdiff_t found = 0;
            std::ptrdiff_t false_detections = 0;
            std::ptrdiff_t identified = 0;
        };

        struct Match {
            std::size_t detection = 0;
            std::size_t label = 0;
            double distance = 0.0;
        };

        std::vector<Match> listCandidates(
            const std::vector<table::Ball>& detected,
            const std::vector<table::Ball>& truth,
            float match_distance
        ) {
            auto candidates =
                views::cartesian_product(
                    views::iota(0UZ, detected.size()), views::iota(0UZ, truth.size())
                )
                | views::transform([&](const auto& pair) {
                      const auto [detection, label] = pair;
                      const cv::Point2f offset =
                          detected[detection].position - truth[label].position;
                      return Match{
                          .detection = detection, .label = label, .distance = cv::norm(offset)
                      };
                  })
                | views::filter([&](const Match& match) { return match.distance < match_distance; })
                | std::ranges::to<std::vector>();

            std::ranges::sort(candidates, {}, &Match::distance);
            return candidates;
        }

        bool sharesBall(const Match& left, const Match& right) {
            return left.detection == right.detection || left.label == right.label;
        }

        std::vector<Match> matchGreedily(const std::vector<Match>& candidates) {
            return std::ranges::fold_left(
                candidates,
                std::vector<Match>{},
                [](std::vector<Match> matches, const Match& candidate) {
                    if (std::ranges::none_of(matches, std::bind_front(sharesBall, candidate))) {
                        matches.push_back(candidate);
                    }

                    return matches;
                }
            );
        }

        Score tallyScore(
            const std::vector<Match>& matches,
            const std::vector<table::Ball>& detected,
            const std::vector<table::Ball>& truth
        ) {
            const auto found = std::ranges::ssize(matches);
            const auto identified = std::ranges::count_if(matches, [&](const Match& match) {
                return detected[match.detection].number == truth[match.label].number;
            });

            return {
                .labelled = std::ranges::ssize(truth),
                .found = found,
                .false_detections = std::ranges::ssize(detected) - found,
                .identified = identified
            };
        }

        Score scoreTable(
            const table::Image& table,
            const std::vector<table::Ball>& detected,
            float match_distance
        ) {
            const auto matches =
                matchGreedily(listCandidates(detected, table.truth, match_distance));

            return tallyScore(matches, detected, table.truth);
        }

        std::vector<Score> scoreDistances(
            const table::Image& table,
            const std::vector<table::Ball>& detected
        ) {
            return MATCH_DISTANCES
                | views::transform([&](float distance) {
                       return scoreTable(table, detected, distance);
                   })
                | std::ranges::to<std::vector>();
        }

        Score addScores(const Score& left, const Score& right) {
            return {
                .labelled = left.labelled + right.labelled,
                .found = left.found + right.found,
                .false_detections = left.false_detections + right.false_detections,
                .identified = left.identified + right.identified
            };
        }

        std::vector<Score> sumScores(const std::vector<std::vector<Score>>& scores) {
            return std::ranges::fold_left(
                scores,
                std::vector<Score>(MATCH_DISTANCES.size()),
                [](const std::vector<Score>& total, const std::vector<Score>& table_scores) {
                    return views::zip_transform(addScores, total, table_scores)
                        | std::ranges::to<std::vector>();
                }
            );
        }

        std::string formatRatio(std::ptrdiff_t numerator, std::ptrdiff_t denominator) {
            if (denominator == 0) {
                return "n/a";
            }

            return std::format(
                "{:.3f}", static_cast<double>(numerator) / static_cast<double>(denominator)
            );
        }

        std::string formatF1(const Score& score) {
            const auto missed = score.labelled - score.found;
            return formatRatio(
                2 * score.found, (2 * score.found) + score.false_detections + missed
            );
        }

        std::string formatScore(const Score& score) {
            return std::format(
                "found {}/{}, missed {}, false detections {}, precision {}, recall {}, F1 {}, "
                "identified {}",
                score.found,
                score.labelled,
                score.labelled - score.found,
                score.false_detections,
                formatRatio(score.found, score.found + score.false_detections),
                formatRatio(score.found, score.labelled),
                formatF1(score),
                formatRatio(score.identified, score.found)
            );
        }

        void printScores(const std::vector<Score>& scores) {
            views::zip(MATCH_DISTANCES, scores) | forEach([](const auto& pair) {
                const auto& [distance, score] = pair;
                std::println("  within {:>4} px: {}", distance, formatScore(score));
            });
        }

        void printResults(
            const table::Image& table,
            const std::vector<table::Ball>& detected,
            const std::vector<Score>& scores
        ) {
            std::println("{}", table.name);

            if (detected.empty()) {
                std::println("  no balls detected");
            }

            detected | forEach([](const table::Ball& ball) {
                std::println(
                    "  ball {:>2} at row {:.0f}, col {:.0f}",
                    ball.number,
                    ball.position.y,
                    ball.position.x
                );
            });

            printScores(scores);
        }

        std::vector<Score> evaluateTable(
            const table::Image& table,
            const detection::ReferenceColours& references
        ) {
            const auto detected = detection::detectBalls(table, references);
            auto scores = scoreDistances(table, detected);

            printResults(table, detected, scores);

            return scores;
        }

        void evaluateTables(
            const std::vector<table::Image>& tables,
            const detection::ReferenceColours& references
        ) {
            const auto scores = tables
                | views::transform(std::bind_back(evaluateTable, std::cref(references)))
                | std::ranges::to<std::vector>();

            std::println("Total");
            printScores(sumScores(scores));
        }
    }
}

void MyApplication() {
    const auto result =
        table::loadImage(table::PHOTO_DIRECTORY, detection::REFERENCE_PHOTO)
            .transform(detection::measureReferences)
            .and_then([](const detection::ReferenceColours& references) {
                return table::loadImages(table::PHOTO_DIRECTORY, evaluation::TABLE_PHOTOS)
                    .transform(std::bind_back(evaluation::evaluateTables, std::cref(references)));
            });

    if (!result) {
        std::println(stderr, "{}", result.error());
    }
}
