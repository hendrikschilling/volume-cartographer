#include <deque>
#include <iomanip>
#include <limits>
#include <list>
#include <tuple>
#include <algorithm>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/filesystem.hpp"
#include "vc/core/math/StructureTensor.hpp"
#include "vc/segmentation/LocalResliceParticleSim.hpp"
#include "vc/segmentation/lrps/Common.hpp"
#include "vc/segmentation/lrps/Derivative.hpp"
#include "vc/segmentation/lrps/EnergyMetrics.hpp"
#include "vc/segmentation/lrps/FittedCurve.hpp"
#include "vc/segmentation/lrps/IntensityMap.hpp"

using namespace volcart::segmentation;
namespace fs = volcart::filesystem;
using std::begin;
using std::end;

size_t LocalResliceSegmentation::progressIterations() const
{
    auto minZPoint = std::min_element(
        startingChain_.begin(), startingChain_.end(),
        [](auto a, auto b) { return a[2] < b[2]; });
    auto startIndex = static_cast<int>(std::floor((*minZPoint)[2]));
    return static_cast<size_t>((endIndex_ - startIndex) / stepSize_);
}

std::ostream& operator<<(std::ostream& os, const Voxel& voxel) {
    os << "(" << voxel[0] << ", " << voxel[1] << ", " << voxel[2] << ")";
    return os;
}

std::vector<Voxel> LocalResliceSegmentation::computeCurve(
    FittedCurve currentCurve,
    Chain& currentVs,
    int zIndex) 
{
    // Extract 2D image slices at zIndex and zIndex+1
    cv::Mat slice1 = vol_->getSliceData(zIndex);
    cv::Mat slice2 = vol_->getSliceData(zIndex+1);

    // Calculate the bounding box of the curve to define the region of interest
    int x_min = std::numeric_limits<int>::max();
    int y_min = std::numeric_limits<int>::max();
    int x_max = std::numeric_limits<int>::min();
    int y_max = std::numeric_limits<int>::min();

    for (int i = 0; i < int(currentCurve.size()); ++i) {
        // Get the current point
        Voxel pt_ = currentCurve(i);
        x_min = std::min(x_min, static_cast<int>(pt_[0]));
        y_min = std::min(y_min, static_cast<int>(pt_[1]));
        x_max = std::max(x_max, static_cast<int>(pt_[0]));
        y_max = std::max(y_max, static_cast<int>(pt_[1]));
    }

    // Add a margin to the bounding box to avoid edge effects
    int margin = 15;
    x_min = std::max(0, x_min - margin);
    y_min = std::max(0, y_min - margin);
    x_max = std::min(slice1.cols - 1, x_max + margin);
    y_max = std::min(slice1.rows - 1, y_max + margin);

    // Extract the region of interest
    cv::Rect roi(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1);
    cv::Mat roiSlice1 = slice1(roi);
    cv::Mat roiSlice2 = slice2(roi);

    // Convert to grayscale and normalize the slices
    cv::Mat gray1, gray2;
    cv::normalize(roiSlice1, gray1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(roiSlice2, gray2, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Compute dense optical flow using Farneback method
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 7, 1.2, 0);

    // Initialize the updated curve
    std::vector<Voxel> nextVs;

    for (int i = 0; i < int(currentCurve.size()); ++i) {
        // Get the current point
        Voxel pt_ = currentCurve(i);
        // pt as a cv::Point2f
        cv::Point2f pt(pt_[0], pt_[1]);
        // Convert pt to ROI coordinates
        cv::Point2f roiPt = pt - cv::Point2f(x_min, y_min);
        // Estimate normal
        const cv::Vec3d normal = estimate_normal_at_index_(currentCurve, i);

        // Get the optical flow vector at the current point
        cv::Vec2f flowVec = flow.at<cv::Vec2f>(roiPt);

        // Project flow vector onto the normal direction
        float flowMagnitude = flowVec.dot(cv::Vec2f(normal[0], normal[1]));

        // Move the point along the normal direction with respect to the optical flow vector
        cv::Point2f updatedPt = cv::Vec2f(pt_[0], pt_[1]) + flowMagnitude * cv::Vec2f(normal[0], normal[1]);

        // Add the updated point to the updated curve
        nextVs.push_back(Voxel(updatedPt.x, updatedPt.y, zIndex + 1));
    }

    /////////////////////////////////////////////////////////
    // 3. Clamp points that jumped too far back to a good (interpolated)
    // position. Do this by looking for places where the square of the
    // second derivative is large, and move them back. Tentatively, I'm only
    // going to move back points whose D2^2 evaluates to > 10.
    //
    // Currently, linear interpolation between the previous/next point is
    // used. This could be upgraded to some kind of other fit, possibly
    // cubic interpolation. The end points are linearly extrapolated from
    // their two closest neighbors.

    // Take initial second derivative
    auto secondDeriv = D2(nextVs);
    std::vector<double> normDeriv2(secondDeriv.size());
    std::transform(
        begin(secondDeriv) + 1, end(secondDeriv) - 1, begin(normDeriv2),
        [](auto d) { return cv::norm(d) * cv::norm(d); });

    // Don't resettle points at the beginning or end of the chain
    auto maxVal =
        std::max_element(begin(normDeriv2) + 1, end(normDeriv2) - 1);
    int settlingIters = 0;

    // Iterate until we move all out-of-place points back into place
    while (*maxVal > 10.0 && settlingIters++ < 100) {
        Voxel newPoint;
        int i = maxVal - begin(normDeriv2);
        Voxel diff = 0.5 * nextVs[size_t(i) + 1] - 0.5 * nextVs[i - 1];
        newPoint = nextVs[i - 1] + diff;
        nextVs[i] = newPoint;

        // Re-evaluate second derivative of new curve
        secondDeriv = D2(nextVs);
        std::transform(
            begin(secondDeriv), end(secondDeriv), begin(normDeriv2),
            [](auto d) { return cv::norm(d) * cv::norm(d); });

        // Don't resettle points at the beginning or end of the chain
        maxVal =
            std::max_element(begin(normDeriv2) + 1, end(normDeriv2) - 1);
    }

    // Return the updated vector of Voxel points
    return nextVs;
}

// std::vector<Voxel> LocalResliceSegmentation::computeCurve(
//     FittedCurve currentCurve,
//     Chain& currentVs,
//     int zIndex) {

//     /////////////////////////////////////////////////////////
//     // 1. Generate all candidate positions for all particles
//     std::vector<std::deque<Voxel>> nextPositions;
//     nextPositions.reserve(currentCurve.size());
//     // XXX DEBUG
//     std::vector<IntensityMap> maps;
//     std::vector<Reslice> reslices;
//     maps.reserve(currentCurve.size());
//     reslices.reserve(currentCurve.size());
//     // XXX DEBUG
//     for (int i = 0; i < int(currentCurve.size()); ++i) {
//         // Estimate normal and reslice along it
//         const cv::Vec3d normal = estimate_normal_at_index_(currentCurve, i);
//         const auto reslice = vol_->reslice(
//             currentCurve(i), normal, {0, 0, 1}, resliceSize_, resliceSize_);
//         reslices.push_back(reslice);
//         auto resliceIntensities = reslice.sliceData();

//         // Make the intensity map `stepSize_` layers down from current
//         // position and find the maxima
//         const cv::Point2i center{
//             resliceIntensities.cols / 2, resliceIntensities.rows / 2};
//         const int nextLayerIndex = center.y + static_cast<int>(stepSize_);
//         IntensityMap map(
//             resliceIntensities, static_cast<int>(stepSize_),
//             peakDistanceWeight_, considerPrevious_);
//         const auto allMaxima = map.sortedMaxima();
//         maps.push_back(map);

//         // Handle case where there's no maxima - go straight down
//         if (allMaxima.empty()) {
//             nextPositions.emplace_back(
//                 std::deque<Voxel>{reslice.sliceToVoxelCoord<int>(
//                     {center.x, nextLayerIndex})});
//             continue;
//         }

//         // Convert maxima to voxel positions
//         std::deque<Voxel> maximaQueue;
//         for (auto&& maxima : allMaxima) {
//             maximaQueue.emplace_back(reslice.sliceToVoxelCoord<double>(
//                 {maxima.first, nextLayerIndex}));
//         }
//         nextPositions.push_back(maximaQueue);
//     }

//     /////////////////////////////////////////////////////////
//     // 2. Construct initial guess using top maxima for each next position
//     std::vector<Voxel> nextVs;
//     nextVs.reserve(currentVs.size());
//     for (int i = 0; i < int(nextPositions.size()); ++i) {
//         nextVs.push_back(nextPositions[i].front());
//         maps[i].setChosenMaximaIndex(0);
//     }
//     FittedCurve nextCurve(nextVs, zIndex + 1);

//     // Calculate energy of the current curve
//     double minEnergy = std::numeric_limits<double>::max();

//     // Derivative of energy measure - keeps the previous three measurements
//     // and will evaluate the central difference (when there's enough
//     // measurements)
//     std::deque<double> dEnergy;
//     dEnergy.push_back(minEnergy);

//     /////////////////////////////////////////////////////////
//     // 3. Optimize
//     std::vector<int> indices(currentVs.size());
//     std::iota(begin(indices), end(indices), 0);

//     // - Go until either some hard limit or change in energy is minimal
//     int n = 0;

//     iters_start:
//     while (n++ < numIters_) {

//         // Break if our energy gradient is leveling off
//         dEnergy.push_back(minEnergy);
//         if (dEnergy.size() > 3) {
//             dEnergy.pop_front();
//         }
//         if (dEnergy.size() == 3 &&
//             0.5 * (dEnergy[0] - dEnergy[2]) < DEFAULT_MIN_ENERGY_GRADIENT) {
//             break;
//         }

//         // - Sort paired index-Voxel in increasing local internal energy
//         auto pairs = Zip(indices, SquareDiff(currentVs, nextVs));
//         std::sort(begin(pairs), end(pairs), [](auto p1, auto p2) {
//             return p1.second < p2.second;
//         });

//         // - Go through the sorted list in reverse order, optimizing each
//         // particle. If we find an optimum, then start over with the new
//         // optimal positions. Do this until convergence or we hit a cap on
//         // number of iterations.
//         while (!pairs.empty()) {
//             int maxDiffIdx;
//             double _;
//             std::tie(maxDiffIdx, _) = pairs.back();
//             pairs.pop_back();

//             // Go through each combination for the maximal difference
//             // particle, iterate until you find a new optimum or don't find
//             // anything.
//             while (!nextPositions[maxDiffIdx].empty()) {
//                 std::vector<Voxel> combVs(begin(nextVs), end(nextVs));
//                 combVs[maxDiffIdx] = nextPositions[maxDiffIdx].front();
//                 nextPositions[maxDiffIdx].pop_front();
//                 FittedCurve combCurve(combVs, zIndex + 1);

//                 // Found a new optimum?
//                 double newE = EnergyMetrics::TotalEnergy(
//                     combCurve, alpha_, k1_, k2_, beta_, delta_);
//                 if (newE < minEnergy) {
//                     minEnergy = newE;
//                     maps[maxDiffIdx].incrementMaximaIndex();
//                     nextVs = combVs;
//                     nextCurve = combCurve;
//                 }
//             }
//             goto iters_start;
//         }
//     }

//     /////////////////////////////////////////////////////////
//     // 3. Clamp points that jumped too far back to a good (interpolated)
//     // position. Do this by looking for places where the square of the
//     // second derivative is large, and move them back. Tentatively, I'm only
//     // going to move back points whose D2^2 evaluates to > 10.
//     //
//     // Currently, linear interpolation between the previous/next point is
//     // used. This could be upgraded to some kind of other fit, possibly
//     // cubic interpolation. The end points are linearly extrapolated from
//     // their two closest neighbors.

//     // Take initial second derivative
//     auto secondDeriv = D2(nextVs);
//     std::vector<double> normDeriv2(secondDeriv.size());
//     std::transform(
//         begin(secondDeriv) + 1, end(secondDeriv) - 1, begin(normDeriv2),
//         [](auto d) { return cv::norm(d) * cv::norm(d); });

//     // Don't resettle points at the beginning or end of the chain
//     auto maxVal =
//         std::max_element(begin(normDeriv2) + 1, end(normDeriv2) - 1);
//     int settlingIters = 0;

//     // Iterate until we move all out-of-place points back into place
//     while (*maxVal > 10.0 && settlingIters++ < 100) {
//         Voxel newPoint;
//         int i = maxVal - begin(normDeriv2);
//         Voxel diff = 0.5 * nextVs[size_t(i) + 1] - 0.5 * nextVs[i - 1];
//         newPoint = nextVs[i - 1] + diff;
//         nextVs[i] = newPoint;

//         // Re-evaluate second derivative of new curve
//         secondDeriv = D2(nextVs);
//         std::transform(
//             begin(secondDeriv), end(secondDeriv), begin(normDeriv2),
//             [](auto d) { return cv::norm(d) * cv::norm(d); });

//         // Don't resettle points at the beginning or end of the chain
//         maxVal =
//             std::max_element(begin(normDeriv2) + 1, end(normDeriv2) - 1);
//     }

//     // Return the updated vector of Voxel points
//     return nextVs;
// }

LocalResliceSegmentation::PointSet LocalResliceSegmentation::compute()
{
    // reset progress
    progressStarted();

    // Duplicate the starting chain
    auto currentVs = startingChain_;

    // Update the user-defined boundary
    bb_.setUpperBoundByIndex(2, endIndex_ + 1);

    // Check that incoming points are all within bounds
    if (std::any_of(begin(currentVs), end(currentVs), [this](auto v) {
            return !bb_.isInBounds(v) || !vol_->isInBounds(v);
        })) {
        status_ = Status::ReturnedEarly;
        progressComplete();
        return create_final_pointset_({currentVs});
    }

    const fs::path outputDir("debugvis");
    const fs::path wholeChainDir(outputDir / "whole_chain");
    if (dumpVis_) {
        fs::create_directory(outputDir);
        fs::create_directory(wholeChainDir);
    }

    // Calculate the starting index
    auto minZPoint = std::min_element(
        begin(currentVs), end(currentVs),
        [](auto a, auto b) { return a[2] < b[2]; });
    auto startIndex = static_cast<int>(std::floor((*minZPoint)[2]));

    if (endIndex_ <= startIndex) {
        throw std::domain_error("end index <= start index");
    }

    // Collection to hold all positions
    std::vector<std::vector<Voxel>> points;
    points.reserve(
        (endIndex_ - startIndex + 1) / static_cast<uint64_t>(stepSize_));
    points.push_back(currentVs);

    // std::cout << "Maximum slices size in Cache: " << vol_->getCacheCapacity() << std::endl;
    auto radius = static_cast<int>(std::ceil(materialThickness_ / vol_->voxelSize()) * 0.5) + 10;
    // std::cout << "Need to load at least: " << radius << "slices above" << std::endl;
    auto sliceing_width = resliceSize_ + 10;
    // std::cout << "Sliceing width: " << sliceing_width << std::endl;
    auto preload_nr = std::max(radius, sliceing_width);
    // std::cout << "Preload: " << preload_nr << "slices" << std::endl;
    // Iterate over z-slices
    size_t iteration{0};
    for (int zIndex = startIndex; zIndex <= endIndex_; zIndex += stepSize_) {
        // preload the next few slices
        for(int u = 0; u <= preload_nr; u++) {
            // skip already loaded slices
            if(zIndex != startIndex && u < preload_nr - zIndex + stepSize_) {
                continue;
            }
            else if (zIndex == startIndex && zIndex - u >= 0) {
                vol_->getSliceData(zIndex - u);
            }
            vol_->getSliceData(zIndex + u);
        }
        
        // std::cout << "Start Z Slice " << zIndex << std::endl;
        // Update progress
        progressUpdated(iteration++);

        // Directory to dump vis
        std::stringstream ss;
        ss << std::setw(std::to_string(endIndex_).size()) << std::setfill('0')
           << zIndex;
        const fs::path zIdxDir = outputDir / ss.str();

        //////////////////////////////////////////////////////////
        // 0. Resample current positions so they are evenly spaced
        FittedCurve currentCurve(currentVs, zIndex);
        currentVs = currentCurve.evenlySpacePoints();
        // std::cout << "Initial curve size: " << currentVs.size() << std::endl;

        // Dump entire curve for easy viewing
        if (dumpVis_) {
            ss.str(std::string());
            ss << std::setw(std::to_string(endIndex_).size())
                << std::setfill('0') << zIndex << "_chain.png";
            const auto wholeChainPath = wholeChainDir / ss.str();
            cv::imwrite(
                wholeChainPath.string(),
                draw_particle_on_slice_(currentCurve, zIndex, -1, true));
        }

        // Split the curve into subsegments
        // Calculate num_threads and segment_length
        const int min_points_per_thread = 15;
        int total_points = currentVs.size();
        int num_threads = std::max(1, std::min(static_cast<int>(std::floor(((float)total_points) / (float)min_points_per_thread)), static_cast<int>(std::thread::hardware_concurrency()) - 1));
        int base_segment_length = static_cast<int>(std::floor(((float)total_points) / (float)num_threads));
        int num_threads_with_extra_point = total_points % num_threads;
        
        std::vector<std::vector<Voxel>> subsegment_points(num_threads);

        // Parallel computation of curve segments
        std::vector<std::thread> threads(num_threads);
        std::vector<std::vector<Voxel>> subsegment_vectors(num_threads);
        int start_idx = 0;
        for (int i = 0; i < num_threads; ++i)
        {
            int segment_length = base_segment_length + (i < num_threads_with_extra_point ? 1 : 0);
            int end_idx = start_idx + segment_length;
            // Change start_idx and end_idx to include overlap
            int start_idx_padded = (i == 0) ? 0 : (start_idx - 2);
            int end_idx_padded = (i == num_threads - 1) ? total_points : (end_idx + 2);
            std::vector<Voxel> subsegment(currentVs.begin() + start_idx_padded, currentVs.begin() + end_idx_padded);
            subsegment_vectors[i] = subsegment;
            // std::cout << "Subsegment " << i << " size: " << subsegment.size() << std::endl;
            start_idx = end_idx;
        }

        for (int i = 0; i < num_threads; ++i)
        {

            threads[i] = std::thread([&, i]()
            {
                Chain subsegment_chain(subsegment_vectors[i]);
                FittedCurve subsegmentCurve(subsegment_chain, zIndex);
                std::vector<Voxel> subsegmentNextVs = computeCurve(subsegmentCurve, subsegment_chain, zIndex);
                subsegment_points[i] = subsegmentNextVs;
            });
        }

        // Join threads and stitch curve segments together
        for (auto& thread : threads)
        {
            // std::cout << "Join Thread " << std::endl;
            thread.join();
        }
        // std::cout << "joined all threads" << std::endl;
        std::vector<Voxel> stitched_curve;
        stitched_curve.reserve(currentVs.size());

        // Stitch curve segments together, discarding overlapping points
        for (int i = 0; i < num_threads; ++i)
        {
            if (i > 0) {
                subsegment_points[i].erase(subsegment_points[i].begin(), subsegment_points[i].begin() + 2);
            }
            if (i < num_threads - 1) {
                subsegment_points[i].erase(subsegment_points[i].end() - 2, subsegment_points[i].end());
            }
            stitched_curve.insert(stitched_curve.end(), subsegment_points[i].begin(), subsegment_points[i].end());
        }
        
        // std::cout << "Stitched curve size: " << stitched_curve.size() << std::endl;
        // std::cout << "stitched curve together" << std::endl;
        // Generate nextVs by evenly spacing points in the stitched curve
        FittedCurve stitchedFittedCurve(stitched_curve, zIndex);
        std::vector<Voxel> nextVs = stitchedFittedCurve.evenlySpacePoints();
        // std::cout << "evenly spaced curve" << std::endl;
        // std::cout << "Next curve size: " << nextVs.size() << std::endl;
        // Computation of the next set of points
        // std::vector<Voxel> nextVs = computeCurve(currentCurve, currentVs, zIndex);

        // Check if any points in nextVs are outside volume boundaries. If so,
        // stop iterating and dump the resulting pointcloud.
        if (std::any_of(begin(nextVs), end(nextVs), [this](auto v) {
                return !bb_.isInBounds(v) || !vol_->isInBounds(v);
            })) {
            // std::cout << "Returned early due to out-of-bounds points" << std::endl;
            status_ = Status::ReturnedEarly;
            return create_final_pointset_(points);
        }

        /////////////////////////////////////////////////////////
        // 4. Visualize if specified by user
        if (visualize_) {
            // Since points can change due to 2nd deriv optimization after main
            // optimization, refit a curve and draw that
            FittedCurve newChain(nextVs, zIndex + 1);
            auto chain = draw_particle_on_slice_(newChain, zIndex + 1);
            cv::namedWindow("Next curve", cv::WINDOW_NORMAL);
            cv::imshow("Next curve", chain);
            cv::waitKey(0);
        }

        /////////////////////////////////////////////////////////
        // 5. Set up for next iteration
        currentVs = nextVs;
        points.push_back(nextVs);
    }

    /////////////////////////////////////////////////////////
    // Update progress
    progressComplete();

    // 6. Output final mesh
    // std::cout << "compute() end" << std::endl;
    return create_final_pointset_(points);
}

cv::Vec3d LocalResliceSegmentation::estimate_normal_at_index_(
    const FittedCurve& currentCurve, int index)
{
    auto currentVoxel = currentCurve(index);
    auto radius = static_cast<int>(
        std::ceil(materialThickness_ / vol_->voxelSize()) * 0.5);
    auto eigenPairs = ComputeSubvoxelEigenPairs(vol_, currentVoxel, radius);
    double exp0 = std::log10(eigenPairs[0].first);
    double exp1 = std::log10(eigenPairs[1].first);
    cv::Vec3d normal;
    if (std::abs(exp0 - exp1) > 2.0) {
        normal = eigenPairs[0].second;
    } else {
        auto tan3d = D1At(currentCurve.points(), index, 3);
        normal = tan3d.cross(cv::Vec3d{0, 0, 1});
    }

    // Normalize the normal vector
    double norm = cv::norm(normal);
    if (norm > 0) {
        normal /= norm;
    }

    return normal;
}

LocalResliceSegmentation::PointSet
LocalResliceSegmentation::create_final_pointset_(
    const std::vector<std::vector<Voxel>>& points)
{
    auto rows = points.size();
    auto cols = points[0].size();
    std::vector<cv::Vec3d> tempRow;
    result_.clear();
    result_.setWidth(cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            Voxel v = points[i][j];
            tempRow.emplace_back(v(0), v(1), v(2));
        }
        result_.pushRow(tempRow);
        tempRow.clear();
    }
    return result_;
}

cv::Mat LocalResliceSegmentation::draw_particle_on_slice_(
    const FittedCurve& curve,
    int sliceIndex,
    int particleIndex,
    bool showSpline) const
{
    auto pkgSlice = vol_->getSliceDataCopy(sliceIndex);
    pkgSlice.convertTo(
        pkgSlice, CV_8UC3, 1.0 / std::numeric_limits<uint8_t>::max());
    cv::cvtColor(pkgSlice, pkgSlice, cv::COLOR_GRAY2BGR);

    // Superimpose interpolated currentCurve on window
    if (showSpline) {
        const int n = 500;
        double sum = 0;
        int i = 0;
        std::vector<cv::Point> contour;
        while (i < n && sum <= 1.0) {
            contour.emplace_back(curve.eval(sum));
            sum += 1.0 / (n - 1);
        }
        cv::polylines(pkgSlice, contour, false, BGR_BLUE, 1, cv::LINE_AA);
    } else {
        // Draw circles on the pkgSlice window for each point
        for (size_t i = 0; i < curve.size(); ++i) {
            cv::Point real{int(curve(i)(0)), int(curve(i)(1))};
            cv::circle(pkgSlice, real, 2, BGR_GREEN, -1);
        }
    }

    // Only highlight a point if particleIndex isn't default -1
    if (particleIndex != -1) {
        const Voxel particle = curve(particleIndex);
        cv::circle(
            pkgSlice, {int(particle(0)), int(particle(1))},
            (showSpline ? 2 : 1), BGR_RED, -1);
    }

    return pkgSlice;
}