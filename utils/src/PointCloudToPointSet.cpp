/* vc_convert_pcd_to_ps
 * Utility that converts all PCL point clouds in a
 * volume package into the new VC pointset type */

#include <iostream>
#include <boost/filesystem/path.hpp>
#include <pcl/io/pcd_io.h>

#include "common/io/PointSetIO.h"
#include "common/types/OrderedPointSet.h"
#include "common/types/Point.h"
#include "common/types/PointSet.h"
#include "volumepkg/volumepkg.h"

using namespace volcart;
namespace fs = boost::filesystem;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage:" << std::endl
                  << "    " << argv[0] << " old.volpkg" << std::endl;
        std::exit(1);
    }

    fs::path volpkgPath{argv[1]};
    VolumePkg pkg{volpkgPath};

    for (const auto& seg : pkg.getSegmentations()) {
        pkg.setActiveSegmentation(seg);
        std::cout << "Processing " << pkg.getActiveSegPath() << std::endl;
        auto cloud = pkg.openCloud();

        // Convert to OrderedPointSet
        OrderedPointSet<Point3d> ps(cloud->width, cloud->height);
        for (size_t j = 0; j < cloud->height; ++j) {
            std::vector<Point3d> points;
            points.reserve(cloud->width);
            for (size_t i = 0; i < cloud->width; ++i) {
                points.push_back(
                    {(*cloud)(i, j).x, (*cloud)(i, j).y, (*cloud)(i, j).z});
            }
            ps.push_row(points);
        }

        // Write to disk
        auto psPath = pkg.getActiveSegPath() / "pointset.vcps";
        PointSetIO<Point3d>::WriteOrderedPointSet(psPath, ps);

        // Read back, verify it's correct
        auto newPs = PointSetIO<Point3d>::ReadOrderedPointSet(psPath);
        for (size_t j = 0; j < newPs.height(); ++j) {
            for (size_t i = 0; i < newPs.width(); ++i) {
                assert(newPs(i, j)[0] == (*cloud)(i, j).x);
                assert(newPs(i, j)[1] == (*cloud)(i, j).y);
                assert(newPs(i, j)[2] == (*cloud)(i, j).z);
            }
        }
    }
}