#include "../OpenCVExample/MyApplication.cpp" // NOLINT(bugprone-suspicious-include)

#include <exception>
#include <system_error>

namespace {
    namespace drawing {
        struct Figure {
            std::string name;
            cv::Mat image;
        };

        const cv::Scalar CANDIDATE_COLOUR = {0, 0, 255};
        const cv::Scalar ACCEPTED_COLOUR = {0, 255, 0};
        const cv::Scalar LABEL_COLOUR = {0, 0, 0};
        const cv::Point LABEL_OFFSET = {-10, 5};
        constexpr int OUTLINE_THICKNESS = 2;
        constexpr double LABEL_SCALE = 0.6;
        const cv::Scalar TILE_LABEL_COLOUR = {255, 255, 255};
        const cv::Scalar TILE_BORDER_COLOUR = {90, 90, 90};
        const cv::Point TILE_LABEL_ORIGIN = {6, 20};
        constexpr int HEATMAP_SCALE = 24;
        constexpr int TILE_SCALE = 10;
        constexpr int TILE_COLUMNS = 4;
        constexpr int TILE_BORDER = 4;

        cv::Mat convertToColour(const cv::Mat& grey) {
            cv::Mat converted;
            cv::cvtColor(grey, converted, cv::COLOR_GRAY2BGR);
            return converted;
        }

        cv::Mat backProjectCloth(const cv::Mat& hls, const cv::Mat& cloth) {
            std::array<const float*, 2> ranges = {
                colour::HUE_RANGE.data(), colour::BYTE_RANGE.data()
            };

            cv::Mat likelihood;
            cv::calcBackProject(
                &hls, 1, colour::HUE_SATURATION_CHANNELS.data(), cloth, likelihood, ranges.data()
            );
            return likelihood;
        }

        cv::Mat outlineTable(const cv::Mat& photo) {
            const table::Corners& corners = table::PHOTO_CORNERS;
            const std::vector<cv::Point> outline = {
                cv::Point(corners[0]),
                cv::Point(corners[2]),
                cv::Point(corners[3]),
                cv::Point(corners[1])
            };

            cv::Mat outlined = photo.clone();
            cv::polylines(outlined, std::vector{outline}, true, ACCEPTED_COLOUR, OUTLINE_THICKNESS);
            return outlined;
        }

        cv::Mat drawPhotoCircles(
            const cv::Mat& background,
            const std::vector<circles::Circle>& found,
            const cv::Scalar& outline
        ) {
            cv::Mat drawn = background.clone();
            std::ranges::for_each(found, [&](const circles::Circle& circle) {
                cv::circle(
                    drawn,
                    cv::Point(circle.centre),
                    cvRound(circle.radius),
                    outline,
                    OUTLINE_THICKNESS
                );
            });
            return drawn;
        }

        std::vector<cv::Point2f> mapCirclesToTable(const std::vector<circles::Circle>& found) {
            return found
                | views::transform([](const circles::Circle& circle) {
                       return circles::mapToTable(circle.centre);
                   })
                | std::ranges::to<std::vector>();
        }

        cv::Mat drawTableCircles(
            const cv::Mat& background,
            const std::vector<cv::Point2f>& centres,
            const cv::Scalar& outline
        ) {
            cv::Mat drawn = background.clone();
            std::ranges::for_each(centres, [&](cv::Point2f centre) {
                cv::circle(
                    drawn, cv::Point(centre), table::BALL_DIAMETER / 2, outline, OUTLINE_THICKNESS
                );
            });
            return drawn;
        }

        cv::Mat labelBalls(const cv::Mat& background, const std::vector<table::Ball>& balls) {
            cv::Mat labelled = drawTableCircles(
                background,
                balls | views::transform(&table::Ball::position) | std::ranges::to<std::vector>(),
                ACCEPTED_COLOUR
            );
            std::ranges::for_each(balls, [&](const table::Ball& ball) {
                cv::putText(
                    labelled,
                    std::to_string(ball.number),
                    cv::Point(ball.position) + LABEL_OFFSET,
                    cv::FONT_HERSHEY_SIMPLEX,
                    LABEL_SCALE,
                    LABEL_COLOUR,
                    OUTLINE_THICKNESS
                );
            });
            return labelled;
        }

        cv::Mat renderHeatmap(const cv::Mat& histogram, int scale) {
            cv::Mat rooted;
            cv::sqrt(histogram, rooted);

            cv::Mat grey;
            cv::normalize(rooted, grey, 0.0, MAX_PIXEL, cv::NORM_MINMAX, CV_8U);

            cv::Mat mapped;
            cv::applyColorMap(grey, mapped, cv::COLORMAP_INFERNO);

            cv::Mat enlarged;
            cv::resize(mapped, enlarged, {}, scale, scale, cv::INTER_NEAREST);
            return enlarged;
        }

        cv::Mat sumValueBins(const cv::Mat& histogram) {
            cv::Mat summed(histogram.size[0], histogram.size[1], CV_32F);
            std::ranges::for_each(
                views::cartesian_product(
                    views::iota(0, histogram.size[0]), views::iota(0, histogram.size[1])
                ),
                [&](const auto& bins) {
                    const auto [hue, saturation] = bins;
                    summed.at<float>(hue, saturation) = std::ranges::fold_left(
                        views::iota(0, histogram.size[2]) | views::transform([&](int value) {
                            return histogram.at<float>(hue, saturation, value);
                        }),
                        0.0F,
                        std::plus<>{}
                    );
                }
            );
            return summed;
        }

        cv::Mat frameTile(const cv::Mat& tile, int number) {
            cv::Mat framed;
            cv::copyMakeBorder(
                tile,
                framed,
                TILE_BORDER,
                TILE_BORDER,
                TILE_BORDER,
                TILE_BORDER,
                cv::BORDER_CONSTANT,
                TILE_BORDER_COLOUR
            );
            cv::putText(
                framed,
                std::to_string(number),
                TILE_LABEL_ORIGIN,
                cv::FONT_HERSHEY_SIMPLEX,
                LABEL_SCALE,
                TILE_LABEL_COLOUR
            );
            return framed;
        }

        cv::Mat renderBallHistograms(const std::vector<measurement::BallHistogram>& histograms) {
            const auto rows =
                histograms
                | views::transform([](const measurement::BallHistogram& ball) {
                      return frameTile(
                          renderHeatmap(sumValueBins(ball.hsv_histogram), TILE_SCALE), ball.number
                      );
                  })
                | views::chunk(TILE_COLUMNS)
                | views::transform([](auto row) {
                      cv::Mat joined;
                      cv::hconcat(row | std::ranges::to<std::vector>(), joined);
                      return joined;
                  })
                | std::ranges::to<std::vector>();

            cv::Mat montage;
            cv::vconcat(rows, montage);
            return montage;
        }
    }

    namespace walkthrough {
        const std::string PHOTO_NAME = "Table2.jpeg";
        const fs::path FIGURE_DIRECTORY = "figures";
        const fs::path POOL_DIRECTORY = "OpenCVExample" / table::PHOTO_DIRECTORY;

        Result<table::Photo> findTablePhoto(const std::string& name) {
            const auto found = std::ranges::find(scoring::TABLE_PHOTOS, name, &table::Photo::name);

            if (found == scoring::TABLE_PHOTOS.end()) {
                return std::unexpected("unknown table photo " + name);
            }

            return *found;
        }

        std::vector<drawing::Figure> renderFigures(
            const cv::Mat& photo,
            const table::Image& table,
            const measurement::AllBallsMeasurements& measurements
        ) {
            cv::Mat smoothed;
            cv::medianBlur(table.image, smoothed, colour::SMOOTHING_SIZE);

            cv::Mat hls;
            cv::cvtColor(smoothed, hls, cv::COLOR_BGR2HLS);

            const cv::Mat likelihood = drawing::backProjectCloth(hls, measurements.cloth_histogram);
            const cv::Mat foreground = colour::segmentNonCloth(hls, measurements.cloth_histogram);
            const cv::Mat relative = colour::measureRelativeLab(smoothed, foreground);
            const cv::Mat balls =
                foreground & ~detection::subtractBackground(relative, measurements);

            const circles::EdgeImage edges = circles::detectPhotoEdges(relative);
            const auto candidates = circles::scanHoughBands(edges);
            const auto kept =
                circles::keepCirclesOverBalls(candidates, balls, measurements.ball_size);
            const auto selected = refinement::selectExplainingAway(edges.edges, kept);
            const auto refined =
                selected
                | views::transform(
                    std::bind_front(refinement::refineByRansac, std::cref(edges.edges))
                )
                | std::ranges::to<std::vector>();

            const auto refined_centres = drawing::mapCirclesToTable(refined);
            const auto detected = detection::identifyBalls(table, refined_centres, measurements);

            return {
                {"01-photo", drawing::outlineTable(photo)},
                {"02-perspective", table.image},
                {"03-smoothed", smoothed},
                {"04-cloth-histogram",
                 drawing::renderHeatmap(measurements.cloth_histogram, drawing::HEATMAP_SCALE)},
                {"05-likelihood", likelihood},
                {"06-foreground", foreground},
                {"07-contrast", colour::measureClothContrast(relative)},
                {"08-edges", edges.edges},
                {"09-hough",
                 drawing::drawPhotoCircles(
                     drawing::convertToColour(edges.edges), candidates, drawing::CANDIDATE_COLOUR
                 )},
                {"10-ball-mask", balls},
                {"11-kept",
                 drawing::drawTableCircles(
                     table.image, drawing::mapCirclesToTable(kept), drawing::ACCEPTED_COLOUR
                 )},
                {"12-refined",
                 drawing::drawTableCircles(
                     drawing::drawTableCircles(
                         table.image,
                         drawing::mapCirclesToTable(selected),
                         drawing::CANDIDATE_COLOUR
                     ),
                     refined_centres,
                     drawing::ACCEPTED_COLOUR
                 )},
                {"13-ball-histograms", drawing::renderBallHistograms(measurements.ball_histograms)},
                {"14-identified", drawing::labelBalls(table.image, detected)}
            };
        }

        Result<void> saveFigure(const drawing::Figure& figure) {
            const fs::path path = FIGURE_DIRECTORY / (figure.name + ".png");

            if (!cv::imwrite(path.string(), figure.image)) {
                return std::unexpected("could not write " + path.string());
            }

            std::println("wrote {}", path.string());
            return {};
        }

        Result<void> saveFigures(const std::vector<drawing::Figure>& figures) {
            std::error_code failure;
            fs::create_directories(FIGURE_DIRECTORY, failure);

            if (failure) {
                return std::unexpected(failure.message());
            }

            for (const auto& figure : figures) {
                const auto saved = saveFigure(figure);

                if (!saved) {
                    return saved;
                }
            }

            return {};
        }

        Result<void> generateFigures() {
            return table::loadImage(POOL_DIRECTORY, measurement::REFERENCE_PHOTO)
                .transform(measurement::measureAllBallsPhoto)
                .and_then([](const measurement::AllBallsMeasurements& measurements) {
                    return findTablePhoto(PHOTO_NAME).and_then([&](const table::Photo& labels) {
                        return table::readPhoto(POOL_DIRECTORY / labels.name)
                            .transform([&](const cv::Mat& photo) {
                                const table::Image image = {
                                    .name = labels.name,
                                    .image = table::correctPerspective(photo, table::PHOTO_CORNERS),
                                    .truth = labels.balls
                                };
                                return renderFigures(photo, image, measurements);
                            })
                            .and_then(saveFigures);
                    });
                });
        }
    }

    namespace analysis {
        const cv::Scalar GOOD_COLOUR = {0, 255, 0};
        const cv::Scalar FAIR_COLOUR = {0, 255, 255};
        const cv::Scalar POOR_COLOUR = {0, 0, 255};
        const cv::Scalar LABEL_DOT_COLOUR = {255, 0, 255};
        const cv::Scalar BACKGROUND_COLOUR = {255, 255, 255};
        const cv::Scalar RING_COLOUR = {160, 160, 160};
        const cv::Scalar POINT_COLOUR = {200, 80, 0};
        constexpr int LABEL_DOT_RADIUS = 6;
        constexpr int PLOT_SIZE = 600;
        constexpr int POINT_RADIUS = 4;
        constexpr float PLOT_SCALE = 12.0F;
        const std::string RESULT_PREFIX = "results-";
        const std::string ERROR_FIGURE = "error-vectors";

        struct PhotoAnalysis {
            drawing::Figure overlay;
            std::vector<cv::Point2f> offsets;
        };

        const cv::Scalar& pickTierColour(double distance) {
            if (distance < scoring::MATCH_DISTANCES[0]) {
                return GOOD_COLOUR;
            }

            if (distance < scoring::MATCH_DISTANCES[1]) {
                return FAIR_COLOUR;
            }

            return POOR_COLOUR;
        }

        cv::Mat drawMatches(
            const table::Image& table,
            const std::vector<table::Ball>& detected,
            const std::vector<detection::Match>& matches
        ) {
            cv::Mat drawn = drawing::drawTableCircles(
                table.image,
                detected
                    | views::transform(&table::Ball::position)
                    | std::ranges::to<std::vector>(),
                POOR_COLOUR
            );

            std::ranges::for_each(matches, [&](const detection::Match& match) {
                cv::circle(
                    drawn,
                    cv::Point(detected[match.detection].position),
                    table::BALL_DIAMETER / 2,
                    pickTierColour(match.distance),
                    drawing::OUTLINE_THICKNESS
                );
            });

            std::ranges::for_each(table.truth, [&](const table::Ball& label) {
                cv::circle(
                    drawn, cv::Point(label.position), LABEL_DOT_RADIUS, LABEL_DOT_COLOUR, cv::FILLED
                );
            });

            return drawn;
        }

        std::vector<cv::Point2f> measureOffsets(
            const table::Image& table,
            const std::vector<table::Ball>& detected,
            const std::vector<detection::Match>& matches
        ) {
            return matches
                | views::transform([&](const detection::Match& match) {
                       return detected[match.detection].position
                           - table.truth[match.label].position;
                   })
                | std::ranges::to<std::vector>();
        }

        void printMatches(
            const table::Image& table,
            const std::vector<table::Ball>& detected,
            const std::vector<detection::Match>& matches
        ) {
            std::ranges::for_each(matches, [&](const detection::Match& match) {
                const cv::Point2f offset =
                    detected[match.detection].position - table.truth[match.label].position;
                std::println(
                    "{} ball {:>2} offset x {:+.1f} y {:+.1f} distance {:.1f}",
                    table.name,
                    table.truth[match.label].number,
                    offset.x,
                    offset.y,
                    match.distance
                );
            });
        }

        void printNearest(
            const std::string& stage,
            const std::vector<cv::Point2f>& centres,
            const std::vector<table::Ball>& truth
        ) {
            std::ranges::for_each(truth, [&](const table::Ball& label) {
                if (centres.empty()) {
                    std::println("{:>8} ball {:>2} none", stage, label.number);
                    return;
                }

                const auto distance = [&](cv::Point2f centre) {
                    return cv::norm(centre - label.position);
                };
                const cv::Point2f nearest = *std::ranges::min_element(centres, {}, distance);
                const cv::Point2f offset = nearest - label.position;
                std::println(
                    "{:>8} ball {:>2} offset x {:+6.1f} y {:+6.1f} distance {:6.1f}",
                    stage,
                    label.number,
                    offset.x,
                    offset.y,
                    distance(nearest)
                );
            });
        }

        void diagnoseTable(
            const table::Image& image,
            const measurement::AllBallsMeasurements& measurements
        ) {
            cv::Mat smoothed;
            cv::medianBlur(image.image, smoothed, colour::SMOOTHING_SIZE);

            cv::Mat hls;
            cv::cvtColor(smoothed, hls, cv::COLOR_BGR2HLS);

            const cv::Mat foreground = colour::segmentNonCloth(hls, measurements.cloth_histogram);
            const cv::Mat relative = colour::measureRelativeLab(smoothed, foreground);
            const cv::Mat balls =
                foreground & ~detection::subtractBackground(relative, measurements);

            const circles::EdgeImage edges = circles::detectPhotoEdges(relative);
            const auto candidates = circles::scanHoughBands(edges);
            const auto kept =
                circles::keepCirclesOverBalls(candidates, balls, measurements.ball_size);
            const auto selected = refinement::selectExplainingAway(edges.edges, kept);
            const auto refined =
                selected
                | views::transform(
                    std::bind_front(refinement::refineByRansac, std::cref(edges.edges))
                )
                | std::ranges::to<std::vector>();

            std::println("{}", image.name);
            printNearest("hough", drawing::mapCirclesToTable(candidates), image.truth);
            printNearest("kept", drawing::mapCirclesToTable(kept), image.truth);
            printNearest("selected", drawing::mapCirclesToTable(selected), image.truth);
            printNearest("refined", drawing::mapCirclesToTable(refined), image.truth);
        }

        Result<void> diagnoseStages() {
            return table::loadImage(walkthrough::POOL_DIRECTORY, measurement::REFERENCE_PHOTO)
                .transform(measurement::measureAllBallsPhoto)
                .and_then([](const measurement::AllBallsMeasurements& measurements) {
                    return table::loadImages(walkthrough::POOL_DIRECTORY, scoring::TABLE_PHOTOS)
                        .transform([&](const std::vector<table::Image>& tables) {
                            std::ranges::for_each(tables, [&](const table::Image& image) {
                                diagnoseTable(image, measurements);
                            });
                        });
                });
        }

        PhotoAnalysis analysePhoto(
            const table::Image& table,
            const measurement::AllBallsMeasurements& measurements
        ) {
            const auto detected = detection::detectBalls(table, measurements);
            const auto matches = detection::matchGreedily(
                scoring::listCandidates(detected, table.truth, scoring::MATCH_DISTANCES.back())
            );
            printMatches(table, detected, matches);

            return {
                .overlay =
                    {.name = RESULT_PREFIX + fs::path(table.name).stem().string(),
                     .image = drawMatches(table, detected, matches)},
                .offsets = measureOffsets(table, detected, matches)
            };
        }

        cv::Mat renderErrorPlot(const std::vector<cv::Point2f>& offsets) {
            const cv::Point centre(PLOT_SIZE / 2, PLOT_SIZE / 2);
            cv::Mat plot(PLOT_SIZE, PLOT_SIZE, CV_8UC3, BACKGROUND_COLOUR);

            cv::line(plot, {0, centre.y}, {PLOT_SIZE, centre.y}, RING_COLOUR);
            cv::line(plot, {centre.x, 0}, {centre.x, PLOT_SIZE}, RING_COLOUR);

            std::ranges::for_each(scoring::MATCH_DISTANCES, [&](float distance) {
                cv::circle(plot, centre, cvRound(distance * PLOT_SCALE), RING_COLOUR);
            });

            std::ranges::for_each(offsets, [&](cv::Point2f offset) {
                cv::circle(
                    plot,
                    centre + cv::Point(offset * PLOT_SCALE),
                    POINT_RADIUS,
                    POINT_COLOUR,
                    cv::FILLED
                );
            });

            return plot;
        }

        void printMeanOffset(const std::vector<cv::Point2f>& offsets) {
            const cv::Point2f mean = std::ranges::fold_left(offsets, cv::Point2f{}, std::plus<>{})
                / static_cast<float>(offsets.size());
            std::println("mean offset x {:+.2f} y {:+.2f}", mean.x, mean.y);
        }

        Result<void> saveAnalyses(const std::vector<PhotoAnalysis>& analyses) {
            const auto offsets = analyses
                | views::transform(&PhotoAnalysis::offsets)
                | views::join
                | std::ranges::to<std::vector>();
            printMeanOffset(offsets);

            auto figures = analyses
                | views::transform(&PhotoAnalysis::overlay)
                | std::ranges::to<std::vector>();
            figures.push_back({ERROR_FIGURE, renderErrorPlot(offsets)});

            return walkthrough::saveFigures(figures);
        }

        Result<void> generateResultFigures() {
            return table::loadImage(walkthrough::POOL_DIRECTORY, measurement::REFERENCE_PHOTO)
                .transform(measurement::measureAllBallsPhoto)
                .and_then([](const measurement::AllBallsMeasurements& measurements) {
                    return table::loadImages(walkthrough::POOL_DIRECTORY, scoring::TABLE_PHOTOS)
                        .transform([&](const std::vector<table::Image>& tables) {
                            return tables
                                | views::transform(
                                       std::bind_back(analysePhoto, std::cref(measurements))
                                )
                                | std::ranges::to<std::vector>();
                        })
                        .and_then(saveAnalyses);
                });
        }
    }
}

int main() {
    try {
        const auto result = walkthrough::generateFigures()
                                .and_then(analysis::generateResultFigures)
                                .and_then(analysis::diagnoseStages);
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
