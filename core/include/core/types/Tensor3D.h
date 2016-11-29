#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <opencv2/core/core.hpp>

namespace volcart
{

template <typename DType>
class Tensor3D
{
private:
    std::vector<cv::Mat_<DType>> tensor_;

public:
    int32_t dx;
    int32_t dy;
    int32_t dz;

    Tensor3D<DType>() = default;

    Tensor3D<DType>(int32_t x, int32_t y, int32_t z, bool zero = true)
        : dx(x), dy(y), dz(z)
    {
        tensor_.reserve(dz);

        // XXX check if these dimensions are too large?
        if (zero) {
            // Static vars are automatically initialized to zero, so it's a
            // convenient way to zero-initialize the tensor
            static DType d;
            for (int32_t i = 0; i < dz; ++i) {
                tensor_.emplace_back(dy, dx, d);
            }
        } else {
            for (int32_t i = 0; i < dz; ++i) {
                tensor_.emplace_back(dy, dx);
            }
        }
    }

    const cv::Mat_<DType>& xySlice(int32_t z) const { return tensor_[z]; }
    cv::Mat_<DType>& xySlice(int32_t z) { return tensor_[z]; }
    cv::Mat_<DType> xzSlice(int32_t layer) const
    {
        cv::Mat_<DType> zSlice(dz, dx);
        for (int32_t z = 0; z < dz; ++z) {
            tensor_[z].row(layer).copyTo(zSlice.row(z));
        }
        return zSlice;
    }

    const DType& operator()(int32_t x, int32_t y, int32_t z) const
    {
        assert(x < dx && x >= 0 && "index out of range");
        assert(y < dy && y >= 0 && "index out of range");
        assert(z < dz && z >= 0 && "index out of range");
        return tensor_[z](y, x);
    }

    DType& operator()(int32_t x, int32_t y, int32_t z)
    {
        assert(x < dx && x >= 0 && "index out of range");
        assert(y < dy && y >= 0 && "index out of range");
        assert(z < dz && z >= 0 && "index out of range");
        return tensor_[z](y, x);
    }

    std::unique_ptr<DType[]> buffer() const
    {
        auto buf = std::unique_ptr<DType[]>(new DType[dx * dy * dz]);
        for (int32_t z = 0; z < dz; ++z) {
            for (int32_t y = 0; y < dy; ++y) {
                for (int32_t x = 0; x < dx; ++x) {
                    buf[z * dx * dy + y * dx + x] = tensor_[z](y, x);
                }
            }
        }
        return buf;
    }
};
}

template <typename DType>
std::ostream& operator<<(
    std::ostream& s, const volcart::Tensor3D<DType>& tensor)
{
    for (int32_t z = 0; z < tensor.dz; ++z) {
        s << tensor.xySlice(z) << std::endl;
    }
    return s;
}
