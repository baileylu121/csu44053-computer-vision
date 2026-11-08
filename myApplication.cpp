#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <numbers>
#include <print>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    namespace views = std::views;

    template <typename T>
    using Result = std::expected<T, std::string>;

    constexpr int MAX_PIXEL = 255;

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

    inline constexpr CollectResults collectResults;

    struct Match {
        std::size_t detection = 0;
        std::size_t label = 0;
        double distance = 0.0;
    };

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

    namespace table {
        const fs::path PHOTO_DIRECTORY = "Media/Pool";

        constexpr int BALL_DIAMETER = 43;

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

        const cv::Point2f BOTTOM_RIGHT = cv::Point(TABLE_SIZE - cv::Size(1, 1));

        const Corners TABLE_CORNERS = {
            cv::Point2f{0, 0},
            cv::Point2f{0, BOTTOM_RIGHT.y},
            cv::Point2f{BOTTOM_RIGHT.x, 0},
            BOTTOM_RIGHT
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
            const cv::Mat transform =
                cv::getPerspectiveTransform(corners.data(), TABLE_CORNERS.data());

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
                | collectResults;
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

        constexpr int SMOOTHING_SIZE = 5;
        constexpr std::array<int, 2> HUE_SATURATION_CHANNELS = {0, 2};
        constexpr std::array<int, 2> HUE_SATURATION_BINS = {16, 16};
        constexpr std::array<float, 2> HUE_RANGE = {0.0F, 180.0F};
        constexpr std::array<float, 2> BYTE_RANGE = {0.0F, MAX_PIXEL + 1.0F};
        constexpr int CLOTH_MARGIN = table::BALL_DIAMETER * 3 / 2;
        constexpr int OPENING_SIZE = 5;

        struct BallHistogram {
            int number = 0;
            cv::Mat hsv_histogram;
        };

        struct AllBallsMeasurements {
            cv::Mat cloth_histogram;
            cv::Mat cushion_mask;
            cv::Mat cushion_background;
            cv::Mat foreground;
            cv::Size ball_size;
            std::vector<BallHistogram> ball_histograms;
        };

        constexpr std::array<int, 3> HSV_CHANNELS = {0, 1, 2};
        constexpr std::array<int, 3> HSV_BINS = {16, 8, 8};

        cv::Mat calculateHsvHistogram(const cv::Mat& hsv, const cv::Mat& mask) {
            std::array<const float*, 3> ranges = {
                HUE_RANGE.data(), BYTE_RANGE.data(), BYTE_RANGE.data()
            };

            cv::Mat histogram;
            cv::calcHist(
                &hsv, 1, HSV_CHANNELS.data(), mask, histogram, 3, HSV_BINS.data(), ranges.data()
            );
            return histogram;
        }

        constexpr int HISTOGRAM_SMOOTHING = 3;

        cv::Mat smoothHistogram(const cv::Mat& histogram) {
            cv::Mat smoothed;
            cv::GaussianBlur(histogram, smoothed, {HISTOGRAM_SMOOTHING, HISTOGRAM_SMOOTHING}, 0.0);
            cv::normalize(smoothed, smoothed, 0.0, MAX_PIXEL, cv::NORM_MINMAX);
            return smoothed;
        }

        cv::Mat maskNonCloth(const cv::Mat& hls, const cv::Mat& cloth) {
            std::array<const float*, 2> ranges = {HUE_RANGE.data(), BYTE_RANGE.data()};
            cv::Mat likelihood;
            cv::calcBackProject(
                &hls, 1, HUE_SATURATION_CHANNELS.data(), cloth, likelihood, ranges.data()
            );

            cv::Mat foreground;
            cv::threshold(
                likelihood, foreground, 0.0, MAX_PIXEL, cv::THRESH_BINARY_INV | cv::THRESH_OTSU
            );
            cv::morphologyEx(
                foreground,
                foreground,
                cv::MORPH_OPEN,
                cv::getStructuringElement(cv::MORPH_ELLIPSE, {OPENING_SIZE, OPENING_SIZE})
            );
            return foreground;
        }

        cv::Mat keepBorderComponents(const cv::Mat& mask) {
            cv::Mat labels;
            cv::Mat stats;
            cv::Mat centroids;
            const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

            cv::Mat kept = cv::Mat::zeros(mask.size(), CV_8U);
            views::iota(1, count) | views::filter([&](int label) {
                const cv::Rect bounds(
                    stats.at<int>(label, cv::CC_STAT_LEFT),
                    stats.at<int>(label, cv::CC_STAT_TOP),
                    stats.at<int>(label, cv::CC_STAT_WIDTH),
                    stats.at<int>(label, cv::CC_STAT_HEIGHT)
                );
                const cv::Rect interior(cv::Point(1, 1), mask.size() - cv::Size(2, 2));
                return (bounds & interior) != bounds;
            }) | forEach([&](int label) { kept.setTo(MAX_PIXEL, labels == label); });
            return kept;
        }

        cv::Mat maskOutsideBalls(const table::Image& all_balls) {
            cv::Mat outside(all_balls.image.size(), CV_8U, cv::Scalar(MAX_PIXEL));
            all_balls.truth | forEach([&](const table::Ball& ball) {
                cv::circle(
                    outside, cv::Point(ball.position), CLOTH_MARGIN, cv::Scalar(0), cv::FILLED
                );
            });
            return outside;
        }

        int countForegroundRun(const cv::Mat& foreground, cv::Point start, cv::Point step) {
            const cv::Rect bounds({}, foreground.size());
            return static_cast<int>(std::ranges::distance(
                views::iota(1)
                | views::transform([&](int distance) { return start + (step * distance); })
                | views::take_while([&](cv::Point point) {
                      return bounds.contains(point) && foreground.at<uchar>(point) != 0;
                  })
            ));
        }

        int measureDiameterAlong(const cv::Mat& foreground, cv::Point centre, cv::Point step) {
            const int half = std::min(
                countForegroundRun(foreground, centre, step),
                countForegroundRun(foreground, centre, -step)
            );
            return (2 * half) + 1;
        }

        cv::Size measureBallSize(const cv::Mat& foreground, const std::vector<table::Ball>& balls) {
            const auto measureMedian = [&](cv::Point step) {
                auto extents =
                    balls
                    | views::transform([&](const table::Ball& ball) {
                          return measureDiameterAlong(foreground, cv::Point(ball.position), step);
                      })
                    | std::ranges::to<std::vector>();
                const auto middle = extents.begin() + std::ranges::ssize(extents) / 2;
                std::ranges::nth_element(extents, middle);
                return *middle;
            };

            return {measureMedian({1, 0}), measureMedian({0, 1})};
        }

        cv::Point2f findNearestCentre(
            cv::Point2f pixel,
            const std::vector<cv::Point2f>& positions,
            cv::Size2f ball_size
        ) {
            return *std::ranges::min_element(positions, {}, [&](cv::Point2f position) {
                const cv::Point2f offset = pixel - position;
                return std::hypot(offset.x, offset.y * ball_size.width / ball_size.height);
            });
        }

        auto enumeratePixels(cv::Rect bounds) {
            return views::cartesian_product(
                       views::iota(bounds.y, bounds.y + bounds.height),
                       views::iota(bounds.x, bounds.x + bounds.width)
                   )
                | views::transform([](const auto& pair) {
                       const auto [row, col] = pair;
                       return cv::Point(col, row);
                   });
        }

        cv::Mat maskBallFootprint(cv::Size size, cv::Point2f position, cv::Size ball_size) {
            cv::Mat region = cv::Mat::zeros(size, CV_8U);
            cv::ellipse(
                region,
                cv::Point(position),
                ball_size / 2,
                0.0,
                0.0,
                360.0,
                cv::Scalar(MAX_PIXEL),
                cv::FILLED
            );
            return region;
        }

        cv::Mat maskOwnBallPixels(
            const cv::Mat& foreground,
            cv::Point2f position,
            const std::vector<cv::Point2f>& positions,
            cv::Size ball_size
        ) {
            cv::Mat region = maskBallFootprint(foreground.size(), position, ball_size) & foreground;

            const cv::Rect bounds =
                cv::Rect(cv::Point(position) - cv::Point(ball_size / 2), ball_size)
                & cv::Rect({}, foreground.size());
            enumeratePixels(bounds) | views::filter([&](cv::Point pixel) {
                return findNearestCentre(cv::Point2f(pixel), positions, ball_size) != position;
            }) | forEach([&](cv::Point pixel) { region.at<uchar>(pixel) = 0; });

            return region;
        }

        cv::Mat measureClothHistogram(const cv::Mat& hls, const cv::Mat& outside) {
            std::array<const float*, 2> ranges = {HUE_RANGE.data(), BYTE_RANGE.data()};

            cv::Mat histogram;
            cv::calcHist(
                &hls,
                1,
                HUE_SATURATION_CHANNELS.data(),
                outside,
                histogram,
                2,
                HUE_SATURATION_BINS.data(),
                ranges.data()
            );
            return smoothHistogram(histogram);
        }

        std::vector<BallHistogram> measureBallHistograms(
            const table::Image& all_balls,
            const cv::Mat& foreground,
            cv::Size ball_size
        ) {
            cv::Mat smoothed;
            cv::medianBlur(all_balls.image, smoothed, SMOOTHING_SIZE);
            cv::Mat hsv;
            cv::cvtColor(smoothed, hsv, cv::COLOR_BGR2HSV);
            const auto positions = all_balls.truth
                | views::transform(&table::Ball::position)
                | std::ranges::to<std::vector>();

            return all_balls.truth
                | views::transform([&](const table::Ball& ball) {
                       const cv::Mat region =
                           maskOwnBallPixels(foreground, ball.position, positions, ball_size);
                       return BallHistogram{
                           .number = ball.number,
                           .hsv_histogram = calculateHsvHistogram(hsv, region)
                       };
                   })
                | std::ranges::to<std::vector>();
        }

        cv::Mat measureLabDistance(const cv::Mat& lab_difference) {
            std::array<cv::Mat, 3> channels;
            cv::split(lab_difference, channels.data());

            cv::Mat distance;
            cv::magnitude(channels[0], channels[1], distance);
            cv::magnitude(distance, channels[2], distance);
            return distance;
        }

        cv::Mat measureClothContrast(const cv::Mat& relative) {
            cv::Mat contrast;
            measureLabDistance(relative).convertTo(contrast, CV_8U);
            return contrast;
        }

        const cv::Mat PHOTO_TRANSFORM =
            cv::getPerspectiveTransform(table::TABLE_CORNERS.data(), table::PHOTO_CORNERS.data());
        const cv::Mat TABLE_TRANSFORM = PHOTO_TRANSFORM.inv();
        const cv::Rect PHOTO_BOUNDS = cv::boundingRect(table::PHOTO_CORNERS);

        cv::Mat warpPerspectiveToPhoto(const cv::Mat& image) {
            cv::Mat photo;
            cv::warpPerspective(
                image,
                photo,
                PHOTO_TRANSFORM,
                PHOTO_BOUNDS.br(),
                cv::INTER_LINEAR,
                cv::BORDER_REPLICATE
            );
            return photo;
        }

        cv::Mat maskPhotoTable() {
            cv::Mat inside;
            cv::warpPerspective(
                cv::Mat(table::TABLE_SIZE, CV_8U, cv::Scalar(MAX_PIXEL)),
                inside,
                PHOTO_TRANSFORM,
                PHOTO_BOUNDS.br(),
                cv::INTER_NEAREST
            );
            return inside;
        }

        const cv::Mat PHOTO_TABLE = maskPhotoTable();

        constexpr int GRADIENT_BINS = (4 * MAX_PIXEL) + 1;
        constexpr std::array<float, 2> GRADIENT_RANGE = {0.0F, float{GRADIENT_BINS}};

        cv::Mat countSobelGradients(const cv::Mat& photo, cv::Point order) {
            cv::Mat gradient;
            cv::Sobel(photo, gradient, CV_32F, order.x, order.y);
            gradient = cv::abs(gradient);

            const float* range = GRADIENT_RANGE.data();
            const int channel = 0;

            cv::Mat counts;
            cv::calcHist(&gradient, 1, &channel, PHOTO_TABLE, counts, 1, &GRADIENT_BINS, &range);
            return counts;
        }

        constexpr double NORMAL_MAD = 0.6745;

        double estimateMadNoise(const cv::Mat& photo) {
            const cv::Mat counts =
                countSobelGradients(photo, {1, 0}) + countSobelGradients(photo, {0, 1});
            const double half = cv::sum(counts)[0] / 2.0;
            const int median =
                *std::ranges::partition_point(views::iota(0, counts.rows), [&](int bin) {
                    return cv::sum(counts.rowRange(0, bin + 1))[0] < half;
                });
            return median / NORMAL_MAD;
        }

        constexpr double NOISE_SIGMAS = 3.0;
        constexpr double CANNY_RATIO = 2.0;

        double chooseCannyThreshold(const cv::Mat& photo) {
            cv::Mat binary;
            const double otsu =
                cv::threshold(photo, binary, 0.0, MAX_PIXEL, cv::THRESH_BINARY | cv::THRESH_OTSU);
            return std::max(otsu, CANNY_RATIO * NOISE_SIGMAS * estimateMadNoise(photo));
        }

        constexpr int CANNY_APERTURE = 3;

        struct EdgeImage {
            cv::Mat contrast;
            double canny_threshold = 0.0;
            cv::Mat edges;
        };

        EdgeImage detectCannyEdges(const cv::Mat& contrast) {
            const double canny_threshold = chooseCannyThreshold(contrast);

            cv::Mat edges;
            cv::Canny(
                contrast, edges, canny_threshold / CANNY_RATIO, canny_threshold, CANNY_APERTURE
            );
            return {.contrast = contrast, .canny_threshold = canny_threshold, .edges = edges};
        }

        cv::Point2d transformPoint(cv::Point2d point, const cv::Mat& transform) {
            std::vector<cv::Point2d> transformed;
            cv::perspectiveTransform(std::vector{point}, transformed, transform);
            return transformed.front();
        }

        cv::Point2f mapToPhoto(cv::Point2f position) {
            return transformPoint(position, PHOTO_TRANSFORM);
        }

        cv::Point2f mapToTable(cv::Point2d centre) {
            return transformPoint(centre, TABLE_TRANSFORM);
        }

        double measurePhotoRadius(cv::Point2f position) {
            const cv::Point2f step =
                mapToPhoto(position + cv::Point2f{1, 0}) - mapToPhoto(position);
            return table::BALL_DIAMETER / 2.0 * cv::norm(step);
        }

        table::Corners listCorners(const cv::Rect& band) {
            return {
                band.tl(),
                band.tl() + cv::Point(0, band.height),
                band.tl() + cv::Point(band.width, 0),
                band.br()
            };
        }

        constexpr double RADIUS_TOLERANCE = 0.2;

        std::pair<int, int> measureRadiusRange(const cv::Rect& band) {
            const auto radii = listCorners(band) | views::transform(measurePhotoRadius);
            return {
                cvFloor(std::ranges::min(radii) * (1.0 - RADIUS_TOLERANCE)),
                cvCeil(std::ranges::max(radii) * (1.0 + RADIUS_TOLERANCE))
            };
        }

        cv::Rect measurePhotoRegion(const cv::Rect& band, int margin) {
            const cv::Rect bounds = cv::boundingRect(
                listCorners(band) | views::transform(mapToPhoto) | std::ranges::to<std::vector>()
            );
            const cv::Point padding = {margin, margin};
            return cv::Rect(bounds.tl() - padding, bounds.br() + padding)
                & cv::Rect({}, PHOTO_BOUNDS.br());
        }

        struct Circle {
            cv::Point2d centre;
            double radius = 0.0;
        };

        Circle projectBallOutline(cv::Point2d centre) {
            return {.centre = centre, .radius = measurePhotoRadius(mapToTable(centre))};
        }

        double measureCircumference(double radius) {
            return 2.0 * std::numbers::pi * radius;
        }

        constexpr double ACCUMULATOR_SPACING = 2.0;
        constexpr double PEAK_SPACING = 1.0;
        constexpr double HOUGH_VOTE_FRACTION = 1.0 / 3.0;

        std::vector<Circle> findHoughCircles(const EdgeImage& photo, const cv::Rect& band) {
            const auto [minimum_radius, maximum_radius] = measureRadiusRange(band);
            const cv::Rect region = measurePhotoRegion(band, maximum_radius);
            const cv::Rect window = {
                band.tl() - cv::Point(0, band.height / 2), band.size() + cv::Size(0, band.height)
            };

            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(
                photo.contrast(region),
                circles,
                cv::HOUGH_GRADIENT,
                ACCUMULATOR_SPACING,
                PEAK_SPACING,
                photo.canny_threshold,
                HOUGH_VOTE_FRACTION * measureCircumference(minimum_radius),
                minimum_radius,
                maximum_radius
            );

            return circles
                | views::transform([&](const cv::Vec3f& circle) {
                       return cv::Point2d(circle[0], circle[1]) + cv::Point2d(region.tl());
                   })
                | views::filter([&](cv::Point2d centre) {
                       return window.contains(cv::Point(mapToTable(centre)));
                   })
                | views::transform(projectBallOutline)
                | std::ranges::to<std::vector>();
        }

        constexpr int BAND_HEIGHT = table::BALL_DIAMETER;

        std::vector<Circle> scanHoughBands(const EdgeImage& photo) {
            return views::iota(0, table::TABLE_SIZE.height)
                | views::stride(BAND_HEIGHT)
                | views::transform([&](int top) {
                       const cv::Rect band = {0, top, table::TABLE_SIZE.width, BAND_HEIGHT};
                       return findHoughCircles(photo, band);
                   })
                | views::join
                | std::ranges::to<std::vector>();
        }

        constexpr int CUSHION_MARGIN = table::BALL_DIAMETER / 4;

        bool clearsCushions(const Circle& circle) {
            const cv::Point margin = {CUSHION_MARGIN, CUSHION_MARGIN};
            return cv::Rect(margin, cv::Point(table::TABLE_SIZE) - margin)
                .contains(cv::Point(mapToTable(circle.centre)));
        }

        constexpr double CLOTH_SPREAD = 10.3;
        constexpr double CHANGE_DISTANCE = NOISE_SIGMAS * CLOTH_SPREAD;

        cv::Mat subtractBackground(const cv::Mat& relative, const AllBallsMeasurements& measurements) {
            const cv::Mat distance = measureLabDistance(relative - measurements.cushion_background);
            return measurements.cushion_mask & (distance <= CHANGE_DISTANCE);
        }

        constexpr double BALL_FRACTION = 0.5;

        bool coversBall(const Circle& circle, const cv::Mat& balls, cv::Size ball_size) {
            const cv::Mat region =
                maskBallFootprint(balls.size(), mapToTable(circle.centre), ball_size);
            return cv::mean(balls, region)[0] >= BALL_FRACTION * MAX_PIXEL;
        }

        std::vector<Circle> verifyCircles(
            const std::vector<Circle>& circles,
            const cv::Mat& balls,
            cv::Size ball_size
        ) {
            return circles
                | views::filter(clearsCushions)
                | views::filter([&](const Circle& circle) {
                       return coversBall(circle, balls, ball_size);
                   })
                | std::ranges::to<std::vector>();
        }

        constexpr double CANDIDATE_TOLERANCE = ACCUMULATOR_SPACING;

        double measureOutlineCoverage(const Circle& circle, const cv::Mat& distances) {
            const cv::Rect bounds({}, distances.size());
            const int samples = cvCeil(measureCircumference(circle.radius));
            const auto covered = std::ranges::count_if(views::iota(0, samples), [&](int sample) {
                const double angle = 2.0 * std::numbers::pi * sample / samples;
                const cv::Point point =
                    circle.centre + cv::Point2d(std::cos(angle), std::sin(angle)) * circle.radius;
                return bounds.contains(point) && distances.at<float>(point) <= CANDIDATE_TOLERANCE;
            });
            return static_cast<double>(covered) / samples;
        }

        void eraseOutline(cv::Mat& edges, const Circle& circle) {
            const double reach = circle.radius + CANDIDATE_TOLERANCE;
            const cv::Point2d corner(reach, reach);
            const cv::Rect bounds = cv::Rect(circle.centre - corner, circle.centre + corner)
                & cv::Rect({}, edges.size());
            enumeratePixels(bounds) | views::filter([&](cv::Point pixel) {
                const double residual = cv::norm(cv::Point2d(pixel) - circle.centre) - circle.radius;
                return std::abs(residual) <= CANDIDATE_TOLERANCE;
            }) | forEach([&](cv::Point pixel) { edges.at<uchar>(pixel) = 0; });
        }

        constexpr float CENTRE_SPACING = 0.8F * table::BALL_DIAMETER;

        constexpr double MIN_OUTLINE_COVERAGE = 1.0 / 3.0;

        std::vector<double> measureCoverages(
            const cv::Mat& edges,
            const std::vector<Circle>& candidates
        ) {
            cv::Mat distances;
            cv::distanceTransform(edges == 0, distances, cv::DIST_L2, cv::DIST_MASK_PRECISE);
            return candidates
                | views::transform(std::bind_back(measureOutlineCoverage, distances))
                | std::ranges::to<std::vector>();
        }

        std::vector<Circle> selectExplainingAway(
            const cv::Mat& edges,
            std::vector<Circle> candidates
        ) {
            cv::Mat remaining = edges.clone();
            std::vector<Circle> accepted;

            while (!candidates.empty()) {
                const auto coverages = measureCoverages(remaining, candidates);
                const auto best = std::ranges::max_element(coverages);

                if (*best < MIN_OUTLINE_COVERAGE) {
                    break;
                }

                const Circle chosen = candidates[std::ranges::distance(coverages.begin(), best)];
                accepted.push_back(chosen);
                eraseOutline(remaining, chosen);

                std::erase_if(candidates, [&](const Circle& candidate) {
                    return cv::norm(mapToTable(candidate.centre) - mapToTable(chosen.centre))
                        < CENTRE_SPACING;
                });
            }

            return accepted;
        }

        EdgeImage detectPhotoEdges(const cv::Mat& relative) {
            cv::Mat blurred;
            cv::GaussianBlur(
                warpPerspectiveToPhoto(measureClothContrast(relative)),
                blurred,
                {SMOOTHING_SIZE, SMOOTHING_SIZE},
                0.0
            );
            return detectCannyEdges(blurred);
        }

        std::vector<cv::Point2f> locateBalls(
            const table::Image& table,
            const AllBallsMeasurements& measurements
        ) {
            cv::Mat smoothed;
            cv::medianBlur(table.image, smoothed, SMOOTHING_SIZE);

            cv::Mat hls;
            cv::cvtColor(smoothed, hls, cv::COLOR_BGR2HLS);
            const cv::Mat foreground = maskNonCloth(hls, measurements.cloth_histogram);

            cv::Mat lab;
            cv::cvtColor(smoothed, lab, cv::COLOR_BGR2Lab);
            lab.convertTo(lab, CV_32F);
            const cv::Mat relative = lab - cv::mean(lab, ~foreground);
            const cv::Mat balls = foreground & ~subtractBackground(relative, measurements);

            const EdgeImage photo = detectPhotoEdges(relative);
            return selectExplainingAway(
                    photo.edges,
                    verifyCircles(scanHoughBands(photo), balls, measurements.ball_size)
                )
                | views::transform([](const Circle& circle) { return mapToTable(circle.centre); })
                | std::ranges::to<std::vector>();
        }

        std::vector<Match> listDetectionMatches(
            std::size_t detection,
            const std::vector<double>& distances
        ) {
            return views::iota(0UZ, distances.size())
                | views::transform([&](std::size_t label) {
                       return Match{
                           .detection = detection, .label = label, .distance = distances[label]
                       };
                   })
                | std::ranges::to<std::vector>();
        }

        std::vector<Match> rankCandidates(const std::vector<std::vector<double>>& distances) {
            auto candidates = views::iota(0UZ, distances.size())
                | views::transform([&](std::size_t detection) {
                                  return listDetectionMatches(detection, distances[detection]);
                              })
                | views::join
                | std::ranges::to<std::vector>();

            std::ranges::sort(candidates, {}, &Match::distance);
            return candidates;
        }

        std::size_t findNearestHistogram(const std::vector<double>& distances) {
            return std::ranges::min(views::iota(0UZ, distances.size()), {}, [&](std::size_t label) {
                return distances[label];
            });
        }

        std::vector<std::size_t> assignGreedily(const std::vector<std::vector<double>>& distances) {
            auto labels =
                distances | views::transform(findNearestHistogram) | std::ranges::to<std::vector>();
            matchGreedily(rankCandidates(distances))
                | forEach([&](const Match& match) { labels[match.detection] = match.label; });
            return labels;
        }

        std::vector<double> measureBhattacharyyaDistances(
            const cv::Mat& histogram,
            const std::vector<BallHistogram>& ball_histograms
        ) {
            return ball_histograms
                | views::transform([&](const BallHistogram& ball) {
                       return cv::compareHist(
                           histogram, ball.hsv_histogram, cv::HISTCMP_BHATTACHARYYA
                       );
                   })
                | std::ranges::to<std::vector>();
        }

        std::vector<table::Ball> identifyBalls(
            const table::Image& table,
            const std::vector<cv::Point2f>& positions,
            const AllBallsMeasurements& measurements
        ) {
            cv::Mat smoothed;
            cv::medianBlur(table.image, smoothed, SMOOTHING_SIZE);

            cv::Mat hsv;
            cv::cvtColor(smoothed, hsv, cv::COLOR_BGR2HSV);
            cv::Mat hls;
            cv::cvtColor(smoothed, hls, cv::COLOR_BGR2HLS);
            const cv::Mat foreground = maskNonCloth(hls, measurements.cloth_histogram);

            const auto measureDistancesToBalls = [&](cv::Point2f position) {
                const cv::Mat region =
                    maskOwnBallPixels(foreground, position, positions, measurements.ball_size);
                return measureBhattacharyyaDistances(
                    calculateHsvHistogram(hsv, region), measurements.ball_histograms
                );
            };
            const auto distances = positions
                | views::transform(measureDistancesToBalls)
                | std::ranges::to<std::vector>();

            const auto labels = assignGreedily(distances);
            return views::zip(positions, labels)
                | views::transform([&](const auto& pair) {
                       const auto [position, label] = pair;
                       return table::Ball{
                           .number = measurements.ball_histograms[label].number,
                           .position = position
                       };
                   })
                | std::ranges::to<std::vector>();
        }

        AllBallsMeasurements measureAllBallsPhoto(const table::Image& all_balls) {
            cv::Mat smoothed;
            cv::medianBlur(all_balls.image, smoothed, SMOOTHING_SIZE);

            cv::Mat hls;
            cv::cvtColor(smoothed, hls, cv::COLOR_BGR2HLS);
            cv::Mat lab;
            cv::cvtColor(smoothed, lab, cv::COLOR_BGR2Lab);
            lab.convertTo(lab, CV_32F);

            const cv::Mat outside = maskOutsideBalls(all_balls);
            const cv::Mat cloth_histogram = measureClothHistogram(hls, outside);
            const cv::Mat foreground = maskNonCloth(hls, cloth_histogram);
            const cv::Size ball_size = measureBallSize(foreground, all_balls.truth);

            return {
                .cloth_histogram = cloth_histogram,
                .cushion_mask = keepBorderComponents(foreground & outside),
                .cushion_background = lab - cv::mean(lab, ~foreground),
                .foreground = foreground,
                .ball_size = ball_size,
                .ball_histograms = measureBallHistograms(all_balls, foreground, ball_size)
            };
        }

        std::vector<table::Ball> detectBalls(
            const table::Image& table,
            const AllBallsMeasurements& measurements
        ) {
            const auto positions = locateBalls(table, measurements);
            return identifyBalls(table, positions, measurements);
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

        Score operator+(const Score& left, const Score& right) {
            return {
                .labelled = left.labelled + right.labelled,
                .found = left.found + right.found,
                .false_detections = left.false_detections + right.false_detections,
                .identified = left.identified + right.identified
            };
        }

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
                      return Match{
                          .detection = detection,
                          .label = label,
                          .distance = cv::norm(detected[detection].position - truth[label].position)
                      };
                  })
                | views::filter([&](const Match& match) { return match.distance < match_distance; })
                | std::ranges::to<std::vector>();

            std::ranges::sort(candidates, {}, &Match::distance);
            return candidates;
        }

        Score tallyScore(
            const std::vector<Match>& matches,
            const std::vector<table::Ball>& detected,
            const std::vector<table::Ball>& truth
        ) {
            const auto found = std::ranges::ssize(matches);
            return {
                .labelled = std::ranges::ssize(truth),
                .found = found,
                .false_detections = std::ranges::ssize(detected) - found,
                .identified = std::ranges::count_if(matches, [&](const Match& match) {
                    return detected[match.detection].number == truth[match.label].number;
                })
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
                | views::transform(
                       std::bind_front(scoreTable, std::cref(table), std::cref(detected))
                )
                | std::ranges::to<std::vector>();
        }

        std::vector<Score> sumScores(const std::vector<std::vector<Score>>& scores) {
            return std::ranges::fold_left(
                scores,
                std::vector<Score>(MATCH_DISTANCES.size()),
                [](const std::vector<Score>& total, const std::vector<Score>& table_scores) {
                    return views::zip_transform(std::plus{}, total, table_scores)
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
            return formatRatio(
                2 * score.found, score.found + score.labelled + score.false_detections
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
            const detection::AllBallsMeasurements& measurements
        ) {
            const auto detected = detection::detectBalls(table, measurements);
            const auto scores = scoreDistances(table, detected);
            printResults(table, detected, scores);
            return scores;
        }

        void evaluateTables(
            const std::vector<table::Image>& tables,
            const detection::AllBallsMeasurements& measurements
        ) {
            const auto scores = tables
                | views::transform(std::bind_back(evaluateTable, std::cref(measurements)))
                | std::ranges::to<std::vector>();

            std::println("Total");
            printScores(sumScores(scores));
        }
    }
}

void MyApplication() {
    const auto result =
        table::loadImage(table::PHOTO_DIRECTORY, detection::REFERENCE_PHOTO)
            .transform(detection::measureAllBallsPhoto)
            .and_then([](const detection::AllBallsMeasurements& measurements) {
                return table::loadImages(table::PHOTO_DIRECTORY, evaluation::TABLE_PHOTOS)
                    .transform(std::bind_back(evaluation::evaluateTables, std::cref(measurements)));
            });

    if (!result) {
        std::println(stderr, "{}", result.error());
    }
}
