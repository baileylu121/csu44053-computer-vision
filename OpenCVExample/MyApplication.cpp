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
#include <optional>
#include <print>
#include <random>
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

    namespace table {
        struct Ball {
            int number = 0;
            cv::Point2f position;
        };

        struct Photo {
            std::string name;
            std::vector<Ball> balls;
        };

        using Corners = std::array<cv::Point2f, 4>;

        struct Image {
            std::string name;
            cv::Mat image;
            std::vector<Ball> truth;
        };

        const fs::path PHOTO_DIRECTORY = "Media/Pool";

        constexpr int BALL_DIAMETER = 43;

        const cv::Size TABLE_SIZE = {1434, 683};

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
            std::vector<Image> images;
            for (const auto& photo : photos) {
                auto image = loadImage(directory, photo);
                if (!image) {
                    return std::unexpected(image.error());
                }

                images.push_back(std::move(*image));
            }
            return images;
        }
    }

    namespace colour {
        constexpr int SMOOTHING_SIZE = 5;
        constexpr int HISTOGRAM_SMOOTHING = 3;

        // Hue and saturation
        constexpr std::array<int, 2> HUE_SATURATION_CHANNELS = {0, 2};
        constexpr std::array<int, 2> HUE_SATURATION_BINS = {16, 16};
        constexpr std::array<float, 2> HUE_RANGE = {0.0F, 180.0F};
        constexpr std::array<float, 2> BYTE_RANGE = {0.0F, MAX_PIXEL + 1.0F};

        constexpr std::array<int, 3> HSV_CHANNELS = {0, 1, 2};
        // Bins for per ball histograms
        constexpr std::array<int, 3> HSV_BINS = {16, 8, 8};

        constexpr int OPENING_SIZE = 5;

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

        cv::Mat smoothHistogram(const cv::Mat& histogram) {
            cv::Mat smoothed;
            cv::GaussianBlur(histogram, smoothed, {HISTOGRAM_SMOOTHING, HISTOGRAM_SMOOTHING}, 0.0);
            cv::normalize(smoothed, smoothed, 0.0, MAX_PIXEL, cv::NORM_MINMAX);
            return smoothed;
        }

        // Instead of finding the balls directly, we back project the cloth,
        // otsu threshold, and then open (removing smaller specks)
        cv::Mat segmentNonCloth(const cv::Mat& hls, const cv::Mat& cloth) {
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

        // We can filter the scoring holes/cushions by keeping circles that
        // not only touch the border but are connected with it
        cv::Mat keepBorderComponents(const cv::Mat& mask) {
            cv::Mat labels;
            cv::Mat stats;
            cv::Mat centroids;
            const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

            cv::Mat kept = cv::Mat::zeros(mask.size(), CV_8U);
            std::ranges::for_each(
                views::iota(1, count) | views::filter([&](int label) {
                    const cv::Rect bounds(
                        stats.at<int>(label, cv::CC_STAT_LEFT),
                        stats.at<int>(label, cv::CC_STAT_TOP),
                        stats.at<int>(label, cv::CC_STAT_WIDTH),
                        stats.at<int>(label, cv::CC_STAT_HEIGHT)
                    );
                    const cv::Rect interior(cv::Point(1, 1), mask.size() - cv::Size(2, 2));
                    return (bounds & interior) != bounds;
                }),
                [&](int label) { kept.setTo(MAX_PIXEL, labels == label); }
            );
            return kept;
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

        // Take the distance in LAB since it approximately matches percieved colour difference
        cv::Mat measureRelativeLab(const cv::Mat& smoothed, const cv::Mat& foreground) {
            cv::Mat lab_image;
            cv::cvtColor(smoothed, lab_image, cv::COLOR_BGR2Lab);
            lab_image.convertTo(lab_image, CV_32F);
            return lab_image - cv::mean(lab_image, ~foreground);
        }

        double measureClothSpread(const cv::Mat& relative, const cv::Mat& foreground) {
            const cv::Mat distance = measureLabDistance(relative);
            return std::sqrt(cv::mean(distance.mul(distance), ~foreground)[0]);
        }
    }

    namespace measurement {
        struct BallHistogram {
            int number = 0;
            cv::Mat hsv_histogram;
        };

        struct AllBallsMeasurements {
            cv::Mat cloth_histogram;
            cv::Mat cushion_mask;
            cv::Mat cushion_background;
            double cloth_spread = 0.0;
            cv::Mat foreground;
            cv::Size ball_size;
            std::vector<BallHistogram> ball_histograms;
        };

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

        // Rough extra margin around each ball so perspective transforms
        // don't mess with ball detection
        constexpr int CLOTH_MARGIN = table::BALL_DIAMETER * 3 / 2;

        cv::Mat maskOutsideBalls(const table::Image& all_balls) {
            cv::Mat outside(all_balls.image.size(), CV_8U, cv::Scalar(MAX_PIXEL));
            std::ranges::for_each(all_balls.truth, [&](const table::Ball& ball) {
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

        // Measures how big a ball appears in the perspective corrected foreground mask
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

        // Find distance with the transform stretch removed
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

        // Pick the pixels belonging to one ball
        cv::Mat maskOwnBallPixels(
            const cv::Mat& foreground,
            cv::Point2f position,
            const std::vector<cv::Point2f>& positions,
            cv::Size ball_size
        ) {
            cv::Mat region =
                colour::maskBallFootprint(foreground.size(), position, ball_size) & foreground;

            const cv::Rect bounds =
                cv::Rect(cv::Point(position) - cv::Point(ball_size / 2), ball_size)
                & cv::Rect({}, foreground.size());

            std::ranges::for_each(
                colour::enumeratePixels(bounds) | views::filter([&](cv::Point pixel) {
                    return findNearestCentre(cv::Point2f(pixel), positions, ball_size) != position;
                }),
                [&](cv::Point pixel) { region.at<uchar>(pixel) = 0; }
            );

            return region;
        }

        std::vector<BallHistogram> measureBallHistograms(
            const table::Image& all_balls,
            const cv::Mat& foreground,
            cv::Size ball_size
        ) {
            cv::Mat smoothed;
            cv::medianBlur(all_balls.image, smoothed, colour::SMOOTHING_SIZE);
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
                           .hsv_histogram = colour::calculateHsvHistogram(hsv, region)
                       };
                   })
                | std::ranges::to<std::vector>();
        }

        AllBallsMeasurements measureAllBallsPhoto(const table::Image& all_balls) {
            cv::Mat smoothed;
            cv::medianBlur(all_balls.image, smoothed, colour::SMOOTHING_SIZE);

            cv::Mat hls;
            cv::cvtColor(smoothed, hls, cv::COLOR_BGR2HLS);
            const cv::Mat outside = maskOutsideBalls(all_balls);
            const cv::Mat cloth_histogram = colour::measureClothHistogram(hls, outside);
            const cv::Mat foreground = colour::segmentNonCloth(hls, cloth_histogram);
            const cv::Size ball_size = measureBallSize(foreground, all_balls.truth);
            const cv::Mat relative = colour::measureRelativeLab(smoothed, foreground);

            return {
                .cloth_histogram = cloth_histogram,
                .cushion_mask = colour::keepBorderComponents(foreground & outside),
                .cushion_background = relative,
                .cloth_spread = colour::measureClothSpread(relative, foreground),
                .foreground = foreground,
                .ball_size = ball_size,
                .ball_histograms = measureBallHistograms(all_balls, foreground, ball_size)
            };
        }
    }

    namespace circles {
        struct EdgeImage {
            cv::Mat contrast;
            double canny_threshold = 0.0;
            cv::Mat edges;
        };

        struct Circle {
            cv::Point2d centre;
            double radius = 0.0;
        };

        const cv::Mat PHOTO_TRANSFORM =
            cv::getPerspectiveTransform(table::TABLE_CORNERS.data(), table::PHOTO_CORNERS.data());

        const cv::Mat TABLE_TRANSFORM = PHOTO_TRANSFORM.inv();
        const cv::Rect PHOTO_BOUNDS = cv::boundingRect(table::PHOTO_CORNERS);

        // Table area in the photos
        const cv::Mat PHOTO_TABLE = [] {
            cv::Mat inside;
            cv::warpPerspective(
                cv::Mat(table::TABLE_SIZE, CV_8U, cv::Scalar(MAX_PIXEL)),
                inside,
                PHOTO_TRANSFORM,
                PHOTO_BOUNDS.br(),
                cv::INTER_NEAREST
            );
            return inside;
        }();

        constexpr int GRADIENT_BINS = (4 * MAX_PIXEL) + 1;
        constexpr std::array<float, 2> GRADIENT_RANGE = {0.0F, float{GRADIENT_BINS}};

        // Median absolute deviation of Gaussian noise, in standard deviations
        constexpr double NORMAL_MAD = 0.6745;
        constexpr double NOISE_SIGMAS = 3.0;
        constexpr double CANNY_RATIO = 2.0;
        constexpr int CANNY_APERTURE = 3;

        constexpr double RADIUS_TOLERANCE = 0.2;
        constexpr double ACCUMULATOR_SPACING = 2.0;
        constexpr double PEAK_SPACING = 1.0;

        // Fraction of a circle's outline that must have enough edges for Hough to report
        constexpr double HOUGH_VOTE_FRACTION = 1.0 / 3.0;
        constexpr int BAND_HEIGHT = table::BALL_DIAMETER;

        constexpr int CUSHION_MARGIN = table::BALL_DIAMETER / 4;
        constexpr double BALL_FRACTION = 0.5;

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

        // Estimate how noisy the photo is
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

        // Pick the canny threshold either from a level above noise or the cloth split
        double chooseCannyThreshold(const cv::Mat& photo) {
            cv::Mat binary;

            const double otsu =
                cv::threshold(photo, binary, 0.0, MAX_PIXEL, cv::THRESH_BINARY | cv::THRESH_OTSU);

            return std::max(otsu, CANNY_RATIO * NOISE_SIGMAS * estimateMadNoise(photo));
        }

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

        // Predict how big a ball should look in the photo for a given point
        // on the table
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

        Circle projectBallOutline(cv::Point2d centre) {
            return {.centre = centre, .radius = measurePhotoRadius(mapToTable(centre))};
        }

        double measureCircumference(double radius) {
            return 2.0 * std::numbers::pi * radius;
        }

        // Look for ball sized circles in a horizontal strip of the table
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

        bool clearsCushions(const Circle& circle) {
            const cv::Point margin = {CUSHION_MARGIN, CUSHION_MARGIN};
            return cv::Rect(margin, cv::Point(table::TABLE_SIZE) - margin)
                .contains(cv::Point(mapToTable(circle.centre)));
        }

        // Checks if a circle is mostly a ball
        bool isMostlyBallFootprint(const Circle& circle, const cv::Mat& balls, cv::Size ball_size) {
            const cv::Mat region =
                colour::maskBallFootprint(balls.size(), mapToTable(circle.centre), ball_size);
            return cv::mean(balls, region)[0] >= BALL_FRACTION * MAX_PIXEL;
        }

        std::vector<Circle> keepCirclesOverBalls(
            const std::vector<Circle>& circles,
            const cv::Mat& balls,
            cv::Size ball_size
        ) {
            return circles
                | views::filter(clearsCushions)
                | views::filter([&](const Circle& circle) {
                       return isMostlyBallFootprint(circle, balls, ball_size);
                   })
                | std::ranges::to<std::vector>();
        }

        EdgeImage detectPhotoEdges(const cv::Mat& relative) {
            cv::Mat blurred;
            cv::GaussianBlur(
                warpPerspectiveToPhoto(colour::measureClothContrast(relative)),
                blurred,
                {colour::SMOOTHING_SIZE, colour::SMOOTHING_SIZE},
                0.0
            );
            return detectCannyEdges(blurred);
        }
    }

    namespace refinement {
        constexpr double CANDIDATE_TOLERANCE = circles::ACCUMULATOR_SPACING;
        constexpr float CENTRE_SPACING = 0.8F * table::BALL_DIAMETER;

        // How much of a circles outline must be backed by real edges to be
        // counted as a ball
        constexpr double MIN_OUTLINE_COVERAGE = 1.0 / 3.0;

        constexpr std::size_t MINIMAL_SAMPLE_SIZE = 3;
        constexpr double FIT_TOLERANCE = 1.5;
        constexpr double RANSAC_CONFIDENCE = 0.99;
        constexpr double INLIER_RATIO = 0.2;
        constexpr std::mt19937::result_type RANSAC_SEED = 43;

        // How many random tries to use for RANSAC to fit a circle
        const int RANSAC_ITERATIONS = cvCeil(
            std::log(1.0 - RANSAC_CONFIDENCE)
            / std::log(1.0 - std::pow(INLIER_RATIO, MINIMAL_SAMPLE_SIZE))
        );

        using MinimalSample = std::array<cv::Point2d, MINIMAL_SAMPLE_SIZE>;

        double measureOutlineCoverage(const circles::Circle& circle, const cv::Mat& distances) {
            const cv::Rect bounds({}, distances.size());
            const int samples = cvCeil(circles::measureCircumference(circle.radius));

            const auto covered = std::ranges::count_if(views::iota(0, samples), [&](int sample) {
                const double angle = 2.0 * std::numbers::pi * sample / samples;

                const cv::Point point =
                    circle.centre + cv::Point2d(std::cos(angle), std::sin(angle)) * circle.radius;

                return bounds.contains(point) && distances.at<float>(point) <= CANDIDATE_TOLERANCE;
            });

            return static_cast<double>(covered) / samples;
        }

        double measureDistanceToOutline(const circles::Circle& circle, cv::Point2d point) {
            return std::abs(cv::norm(point - circle.centre) - circle.radius);
        }

        cv::Rect boundCircle(const circles::Circle& circle, double reach, cv::Size size) {
            const cv::Point2d corner = cv::Point2d(1.0, 1.0) * reach;
            return cv::Rect(circle.centre - corner, circle.centre + corner) & cv::Rect({}, size);
        }

        void eraseOutline(cv::Mat& edges, const circles::Circle& circle) {
            const cv::Rect bounds =
                boundCircle(circle, circle.radius + CANDIDATE_TOLERANCE, edges.size());

            std::ranges::for_each(
                colour::enumeratePixels(bounds) | views::filter([&](cv::Point pixel) {
                    return measureDistanceToOutline(circle, pixel) <= CANDIDATE_TOLERANCE;
                }),
                [&](cv::Point pixel) { edges.at<uchar>(pixel) = 0; }
            );
        }

        std::vector<double> measureCoverages(
            const cv::Mat& edges,
            const std::vector<circles::Circle>& candidates
        ) {
            cv::Mat distances;
            cv::distanceTransform(edges == 0, distances, cv::DIST_L2, cv::DIST_MASK_PRECISE);

            return candidates
                | views::transform(std::bind_back(measureOutlineCoverage, distances))
                | std::ranges::to<std::vector>();
        }

        // Decide which candidate circles are real balls
        std::vector<circles::Circle> selectExplainingAway(
            const cv::Mat& edges,
            std::vector<circles::Circle> candidates
        ) {
            cv::Mat remaining = edges.clone();
            std::vector<circles::Circle> accepted;

            while (!candidates.empty()) {
                const auto coverages = measureCoverages(remaining, candidates);
                const auto best = std::ranges::max_element(coverages);

                if (*best < MIN_OUTLINE_COVERAGE) {
                    break;
                }

                const circles::Circle chosen =
                    candidates[std::ranges::distance(coverages.begin(), best)];

                accepted.push_back(chosen);
                eraseOutline(remaining, chosen);

                std::erase_if(candidates, [&](const circles::Circle& candidate) {
                    return cv::norm(
                               circles::mapToTable(candidate.centre)
                               - circles::mapToTable(chosen.centre)
                           )
                        < CENTRE_SPACING;
                });
            }

            return accepted;
        }

        std::vector<cv::Point2d> collectOutlinePoints(
            const cv::Mat& edges,
            const circles::Circle& circle
        ) {
            const double margin = circles::RADIUS_TOLERANCE * circle.radius;
            const cv::Rect bounds = boundCircle(circle, circle.radius + margin, edges.size());

            std::vector<cv::Point> pixels;
            cv::findNonZero(edges(bounds), pixels);

            return pixels
                | views::transform([&](cv::Point pixel) {
                       return cv::Point2d(pixel + bounds.tl());
                   })
                | views::filter([&](cv::Point2d point) {
                       return measureDistanceToOutline(circle, point) <= margin;
                   })
                | std::ranges::to<std::vector>();
        }

        MinimalSample pickRandomTriple(
            const std::vector<cv::Point2d>& points,
            std::mt19937& generator
        ) {
            std::uniform_int_distribution<std::size_t> pick(0, points.size() - 1);
            MinimalSample sample;

            std::ranges::generate(sample, [&] { return points[pick(generator)]; });

            return sample;
        }

        // Find a circle passing through three points
        std::optional<circles::Circle> circleThroughTriple(const MinimalSample& corners) {
            const cv::Point2d first = corners[1] - corners[0];
            const cv::Point2d second = corners[2] - corners[0];
            const double determinant = 2.0 * first.cross(second);

            if (determinant == 0.0) {
                return std::nullopt;
            }

            const cv::Point2d centre =
                cv::Point2d(
                    (second.y * first.dot(first)) - (first.y * second.dot(second)),
                    (first.x * second.dot(second)) - (second.x * first.dot(first))
                )
                / determinant;

            return circles::Circle{.centre = corners[0] + centre, .radius = cv::norm(centre)};
        }

        // Fit the best circle to a all the inliers
        std::optional<circles::Circle> fitCircleLeastSquares(
            const std::vector<cv::Point2d>& points,
            cv::Point2d origin
        ) {
            const auto equations =
                points
                | views::transform([&](cv::Point2d point) {
                      const cv::Point2d offset = point - origin;
                      return cv::Vec4d{offset.x, offset.y, 1.0, -offset.dot(offset)};
                  })
                | std::ranges::to<std::vector>();

            const cv::Mat system = cv::Mat(equations, false).reshape(1);

            cv::Mat solution;
            if (!cv::solve(system.colRange(0, 3), system.col(3), solution, cv::DECOMP_SVD)) {
                return std::nullopt;
            }

            const cv::Point2d centre(-solution.at<double>(0) / 2.0, -solution.at<double>(1) / 2.0);
            const double squared_radius = centre.dot(centre) - solution.at<double>(2);
            if (squared_radius <= 0.0) {
                return std::nullopt;
            }

            return circles::Circle{.centre = origin + centre, .radius = std::sqrt(squared_radius)};
        }

        bool isNearOutline(const circles::Circle& circle, cv::Point2d point) {
            return measureDistanceToOutline(circle, point) <= FIT_TOLERANCE;
        }

        std::ptrdiff_t countNearOutline(
            const std::vector<cv::Point2d>& points,
            const circles::Circle& circle
        ) {
            return std::ranges::count_if(points, std::bind_front(isNearOutline, circle));
        }

        bool hasPlausibleRadius(const circles::Circle& fit, const circles::Circle& predicted) {
            return std::abs(fit.radius - predicted.radius)
                <= circles::RADIUS_TOLERANCE
                * predicted.radius;
        }

        std::optional<circles::Circle> findConsensusCircle(
            const std::vector<cv::Point2d>& points,
            const circles::Circle& predicted
        ) {
            std::mt19937 generator(RANSAC_SEED);

            const auto hypotheses =
                views::iota(0, RANSAC_ITERATIONS)
                | views::transform([&](int) {
                      return circleThroughTriple(pickRandomTriple(points, generator));
                  })
                | std::ranges::to<std::vector>();

            const auto plausible = hypotheses
                | views::filter([](const auto& hypothesis) { return hypothesis.has_value(); })
                | views::transform([](const auto& hypothesis) { return *hypothesis; })
                | views::filter(std::bind_back(hasPlausibleRadius, predicted))
                | std::ranges::to<std::vector>();

            if (plausible.empty()) {
                return std::nullopt;
            }

            const auto scored = plausible
                | views::transform([&](const circles::Circle& circle) {
                                    return std::pair{countNearOutline(points, circle), circle};
                                })
                | std::ranges::to<std::vector>();

            return std::ranges::max(scored, {}, &std::pair<std::ptrdiff_t, circles::Circle>::first)
                .second;
        }

        // Attempt to refine with ransac or fall back to the hough circle if not
        circles::Circle refineByRansac(const cv::Mat& edges, const circles::Circle& predicted) {
            const auto points = collectOutlinePoints(edges, predicted);
            const int required =
                cvCeil(MIN_OUTLINE_COVERAGE * circles::measureCircumference(predicted.radius));

            if (std::ranges::ssize(points) < required) {
                return predicted;
            }

            return findConsensusCircle(points, predicted)
                .and_then([&](const circles::Circle& consensus) {
                    const auto inliers = points
                        | views::filter(std::bind_front(isNearOutline, consensus))
                        | std::ranges::to<std::vector>();

                    return fitCircleLeastSquares(inliers, predicted.centre);
                })
                .and_then([&](const circles::Circle& fit) {
                    const bool confirmed = hasPlausibleRadius(fit, predicted)
                        && countNearOutline(points, fit)
                        >= required;

                    return confirmed ? std::optional(fit) : std::nullopt;
                })
                .value_or(predicted);
        }
    }

    namespace detection {
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

        // Return cushion pixels that look the same as in the reference
        cv::Mat subtractBackground(
            const cv::Mat& relative,
            const measurement::AllBallsMeasurements& measurements
        ) {
            const cv::Mat distance =
                colour::measureLabDistance(relative - measurements.cushion_background);

            return measurements.cushion_mask
                & (distance <= circles::NOISE_SIGMAS * measurements.cloth_spread);
        }

        std::vector<cv::Point2f> locateBalls(
            const table::Image& table,
            const measurement::AllBallsMeasurements& measurements
        ) {
            cv::Mat smoothed;
            cv::medianBlur(table.image, smoothed, colour::SMOOTHING_SIZE);

            cv::Mat hls;
            cv::cvtColor(smoothed, hls, cv::COLOR_BGR2HLS);
            const cv::Mat foreground = colour::segmentNonCloth(hls, measurements.cloth_histogram);

            const cv::Mat relative = colour::measureRelativeLab(smoothed, foreground);
            const cv::Mat balls = foreground & ~subtractBackground(relative, measurements);

            const circles::EdgeImage photo = circles::detectPhotoEdges(relative);

            return refinement::selectExplainingAway(
                       photo.edges,
                       circles::keepCirclesOverBalls(
                           circles::scanHoughBands(photo), balls, measurements.ball_size
                       )
                   )
                | views::transform(
                       std::bind_front(refinement::refineByRansac, std::cref(photo.edges))
                )
                | views::transform([](const circles::Circle& circle) {
                       return circles::mapToTable(circle.centre);
                   })
                | std::ranges::to<std::vector>();
        }

        std::vector<Match> rankCandidates(const std::vector<std::vector<double>>& distances) {
            auto candidates = views::iota(0UZ, distances.size())
                | views::transform([&](std::size_t detection) {
                                  return views::iota(0UZ, distances[detection].size())
                                      | views::transform([&](std::size_t label) {
                                             return Match{
                                                 .detection = detection,
                                                 .label = label,
                                                 .distance = distances[detection][label]
                                             };
                                         });
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

        // Give each ball a different number to help identify it (closest match)
        std::vector<std::size_t> assignGreedily(const std::vector<std::vector<double>>& distances) {
            auto labels =
                distances | views::transform(findNearestHistogram) | std::ranges::to<std::vector>();

            std::ranges::for_each(
                matchGreedily(rankCandidates(distances)),
                [&](const Match& match) { labels[match.detection] = match.label; }
            );

            return labels;
        }

        std::vector<double> measureBhattacharyyaDistances(
            const cv::Mat& histogram,
            const std::vector<measurement::BallHistogram>& ball_histograms
        ) {
            return ball_histograms
                | views::transform([&](const measurement::BallHistogram& ball) {
                       return cv::compareHist(
                           histogram, ball.hsv_histogram, cv::HISTCMP_BHATTACHARYYA
                       );
                   })
                | std::ranges::to<std::vector>();
        }

        std::vector<table::Ball> identifyBalls(
            const table::Image& table,
            const std::vector<cv::Point2f>& positions,
            const measurement::AllBallsMeasurements& measurements
        ) {
            cv::Mat smoothed;
            cv::medianBlur(table.image, smoothed, colour::SMOOTHING_SIZE);

            cv::Mat hsv;
            cv::cvtColor(smoothed, hsv, cv::COLOR_BGR2HSV);

            cv::Mat hls;
            cv::cvtColor(smoothed, hls, cv::COLOR_BGR2HLS);

            const cv::Mat foreground = colour::segmentNonCloth(hls, measurements.cloth_histogram);

            const auto measureDistancesToBalls = [&](cv::Point2f position) {
                const cv::Mat region = measurement::maskOwnBallPixels(
                    foreground, position, positions, measurements.ball_size
                );

                return measureBhattacharyyaDistances(
                    colour::calculateHsvHistogram(hsv, region), measurements.ball_histograms
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

        std::vector<table::Ball> detectBalls(
            const table::Image& table,
            const measurement::AllBallsMeasurements& measurements
        ) {
            const auto positions = locateBalls(table, measurements);
            return identifyBalls(table, positions, measurements);
        }
    }

    namespace scoring {
        struct Score {
            std::ptrdiff_t labelled = 0;
            std::ptrdiff_t found = 0;
            std::ptrdiff_t false_detections = 0;
            std::ptrdiff_t identified = 0;
        };

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

        constexpr float BALL_RADIUS = table::BALL_DIAMETER / 2.0F;

        // Ball match granularities to test matching effectiveness at different sizes
        constexpr std::array<float, 3> MATCH_DISTANCES =
            {BALL_RADIUS / 4.0F, BALL_RADIUS / 2.0F, BALL_RADIUS};

        Score addScores(const Score& left, const Score& right) {
            return {
                .labelled = left.labelled + right.labelled,
                .found = left.found + right.found,
                .false_detections = left.false_detections + right.false_detections,
                .identified = left.identified + right.identified
            };
        }

        std::vector<detection::Match> listCandidates(
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
                      return detection::Match{
                          .detection = detection,
                          .label = label,
                          .distance = cv::norm(detected[detection].position - truth[label].position)
                      };
                  })
                | views::filter([&](const detection::Match& match) {
                      return match.distance < match_distance;
                  })
                | std::ranges::to<std::vector>();

            std::ranges::sort(candidates, {}, &detection::Match::distance);

            return candidates;
        }

        Score tallyScore(
            const std::vector<detection::Match>& matches,
            const std::vector<table::Ball>& detected,
            const std::vector<table::Ball>& truth
        ) {
            const auto found = std::ranges::ssize(matches);

            return {
                .labelled = std::ranges::ssize(truth),
                .found = found,
                .false_detections = std::ranges::ssize(detected) - found,
                .identified = std::ranges::count_if(matches, [&](const detection::Match& match) {
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
                detection::matchGreedily(listCandidates(detected, table.truth, match_distance));

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
                    return views::zip_transform(addScores, total, table_scores)
                        | std::ranges::to<std::vector>();
                }
            );
        }
    }

    namespace reporting {
        std::string formatRatio(std::ptrdiff_t numerator, std::ptrdiff_t denominator) {
            if (denominator == 0) {
                return "n/a";
            }

            return std::format(
                "{:.3f}", static_cast<double>(numerator) / static_cast<double>(denominator)
            );
        }

        // Work out the F1 score from the counts
        std::string formatF1(const scoring::Score& score) {
            return formatRatio(
                2 * score.found, score.found + score.labelled + score.false_detections
            );
        }

        std::string formatScore(const scoring::Score& score) {
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

        void printScores(const std::vector<scoring::Score>& scores) {
            std::ranges::for_each(
                views::zip(scoring::MATCH_DISTANCES, scores), [](const auto& pair) {
                    const auto& [distance, score] = pair;
                    std::println("  within {:>5.1f} px: {}", distance, formatScore(score));
                }
            );
        }

        void printResults(
            const table::Image& table,
            const std::vector<table::Ball>& detected,
            const std::vector<scoring::Score>& scores
        ) {
            std::println("{}", table.name);

            if (detected.empty()) {
                std::println("  no balls detected");
            }

            std::ranges::for_each(detected, [](const table::Ball& ball) {
                std::println(
                    "  ball {:>2} at row {:.0f}, col {:.0f}",
                    ball.number,
                    ball.position.y,
                    ball.position.x
                );
            });

            printScores(scores);
        }

        std::vector<scoring::Score> evaluateTable(
            const table::Image& table,
            const measurement::AllBallsMeasurements& measurements
        ) {
            const auto detected = detection::detectBalls(table, measurements);
            const auto scores = scoring::scoreDistances(table, detected);
            printResults(table, detected, scores);
            return scores;
        }

        void evaluateTables(
            const std::vector<table::Image>& tables,
            const measurement::AllBallsMeasurements& measurements
        ) {
            const auto scores = tables
                | views::transform(std::bind_back(evaluateTable, std::cref(measurements)))
                | std::ranges::to<std::vector>();

            std::println("Total");
            printScores(scoring::sumScores(scores));
        }
    }
}

void MyApplication() {
    const auto result =
        table::loadImage(table::PHOTO_DIRECTORY, measurement::REFERENCE_PHOTO)
            .transform(measurement::measureAllBallsPhoto)
            .and_then([](const measurement::AllBallsMeasurements& measurements) {
                return table::loadImages(table::PHOTO_DIRECTORY, scoring::TABLE_PHOTOS)
                    .transform(std::bind_back(reporting::evaluateTables, std::cref(measurements)));
            });

    if (!result) {
        std::println(stderr, "{}", result.error());
    }
}
