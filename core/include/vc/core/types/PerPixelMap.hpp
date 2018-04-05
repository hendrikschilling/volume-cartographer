#pragma once

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>

#include "OrderedPointSet.hpp"
#include "UVMap.hpp"

namespace volcart
{
/**
 * @class PerPixelMap
 * @author Seth Parker
 * @date 3/17/16
 *
 * @brief A raster of a UVMap that provides a per-pixel mapping between a Volume
 * and a Texture generated from that volume
 *
 * After a segmentation mesh is flattened, the resulting UV space is sampled at
 * a specific resolution in order to generate a texture space (e.g. image). A
 * texturing algorithm is responsible for filtering information from the volume
 * and placing it into this space, the intended result being an image of a
 * manuscript's text. The transformation that maps 2D coordinates in texture
 * space to 3D coordinates in volume space is defined by the per-vertex
 * transformation generated by flattening, however, there are numerous ways that
 * the points \em between vertices can be mapped back into the volume. Since the
 * calculation of this mapping can be expensive, it is often desirable to
 * perform this operation only once.
 *
 * The PerPixelMap (PPM) provides a method for storing the result of this
 * calculation. It has the same dimensions as texture space, and each pixel
 * holds the mapped, 3D position for that pixel in volume space. It additionally
 * holds 3 extra `double` elements, typically used to store the surface normal
 * vector for the 3D position (calculated from the segmentation mesh).
 *
 * The position and normal vector are stored in a `cv::Vec6d`:
 * `{x, y, z, nx, ny, nz}`.
 *
 * The texturing::PPMGenerator class generates a PerPixelMap by mapping
 * pixels through the barycentric coordinates of the mesh's triangular faces.
 *
 * @ingroup Types
 */
class PerPixelMap
{
public:
    /** Convenience structure for a single pixel's mapping information */
    struct PixelMap {
        /** Default constructor */
        PixelMap() = default;

        /** Construct and initialize */
        PixelMap(int x, int y, cv::Vec6d value)
            : x{x}
            , y{y}
            , pos{value[0], value[1], value[2]}
            , normal{value[3], value[4], value[5]}
        {
        }

        /** PPM Pixel position X */
        int x{0};
        /** PPM Pixel position Y */
        int y{0};
        /** Mapped Volume position */
        cv::Vec3d pos;
        /** Surface normal at mapped Volume position */
        cv::Vec3d normal;
    };

    /**@{*/
    /** @brief Default constructor */
    PerPixelMap() = default;

    /** @brief Constructor with width and height parameters */
    PerPixelMap(size_t height, size_t width) : height_{height}, width_{width}
    {
        initialize_map_();
    }

    /**
     * @brief Return whether the PerPixelMap has been initialized
     *
     * The map is initialized as soon as its width and height have been set.
     */
    bool initialized() const
    {
        return width_ == map_.width() && height_ == map_.height() &&
               width_ > 0 && height_ > 0;
    }
    /**@}*/

    /**@{*/
    /** @brief Get the mapping for a pixel by x, y coordinate */
    const cv::Vec6d& operator()(size_t y, size_t x) const { return map_(y, x); }

    /** @copydoc operator()() */
    cv::Vec6d& operator()(size_t y, size_t x) { return map_(y, x); }

    /**
     * @brief Return whether there is a mapping for the pixel at x, y
     *
     * Returns `true` is the pixel mask has not been set or is empty
     */
    bool hasMapping(size_t y, size_t x)
    {
        if (mask_.empty()) {
            return true;
        }

        return mask_.at<uint8_t>(y, x) == 255;
    }

    /**
     * @brief Get all valid pixel mappings as a sorted list of PixelMap
     *
     * Uses hasMapping() to determine which pixels in the PPM are valid. The
     * resulting list is then sorted using an element of the stored position
     * value. The `sortElement` should be 0, 1, or 2, which correspond to X, Y,
     * and Z respectively.
     */
    std::vector<PixelMap> getSortedMappings(size_t sortElement = 2);
    /**@}*/

    /**@{*/
    /**
     * @brief Set the dimensions of the map
     *
     * @warning Changing the size of the PerPixelMap will clear it of data.
     * Setting either dimension to 0 will result in undefined behavior.
     */
    void setDimensions(size_t h, size_t w);

    /**
     * @brief Set the width of the map
     * @copydetails setDimensions()
     */
    void setWidth(size_t w);

    /**
     * @brief Set the height of the map
     * @copydetails setDimensions()
     */
    void setHeight(size_t h);

    /** @brief Get the width of the map */
    size_t width() const { return width_; }

    /** @brief Get the height of the map */
    size_t height() const { return height_; }
    /**@}*/

    /**@{*/
    /**
     * @brief Get the UVMap
     *
     * Generally the UVMap from which this PPM was generated.
     */
    const UVMap& uvMap() const { return uvMap_; }

    /** @copydoc uvMap() const */
    UVMap& uvMap() { return uvMap_; }

    /**
     * @brief Set the UVMap
     *
     * Useful for keeping a copy of the the UVMap that generated this PPM.
     */
    void setUVMap(const UVMap& u) { uvMap_ = u; }

    /** @brief Get the pixel mask */
    const cv::Mat mask() const { return mask_; }

    /**
     * @brief Set the pixel mask
     *
     * If the pixel mask is not set or is empty, every pixel is assumed to have
     * a mapping.
     *
     * Not every pixel in the PerPixelMap will have a mapped value. The pixel
     * mask is an 8bpc, single channel image that indicates which pixels do and
     * do not have mappings. 0 = No mapping, 255 = Has mapping.
     */
    void setMask(const cv::Mat& m) { mask_ = m.clone(); }
    /**@}*/

    /**@{*/
    /** @brief Write a PerPixelMap to disk */
    static void WritePPM(
        const boost::filesystem::path& path, const PerPixelMap& map);
    /** @brief Read a PerPixelMap from disk */
    static PerPixelMap ReadPPM(const boost::filesystem::path& path);
    /**@}*/

private:
    /**
     * Initialize the map for value assignment
     *
     * Does nothing if either the height or width are 0.
     */
    void initialize_map_();

    /** Height of the map */
    size_t height_{0};
    /** Width of the map */
    size_t width_{0};
    /** Map data storage */
    volcart::OrderedPointSet<cv::Vec6d> map_;

    /** Pixel mask
     *
     * The pixel mask is an 8bpc, single channel image that indicates which
     * pixels do and do not have mappings. 0 = No mapping, 255 = Has mapping.
     */
    cv::Mat mask_;

    /** UVMap used to generate this map */
    UVMap uvMap_{};
};
}  // namespace volcart
