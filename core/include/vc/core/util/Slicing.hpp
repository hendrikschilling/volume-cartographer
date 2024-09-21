#pragma once

#include <xtensor/xarray.hpp>
#include <opencv2/core.hpp>

namespace z5
{
    class Dataset;
}

class CoordGenerator
{
public:
    //given input volume shape, fill a coord slice
    void gen_coords(xt::xarray<float> &coords, int w, int h) const;
    void gen_coords(xt::xarray<float> &coords, const cv::Rect &roi, float render_scale = 1.0, float coord_scale = 1.0) const;
    // virtual void gen_coords(float i, float j, int x, int y, float render_scale, float coord_scale) const = 0;
    virtual void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) const = 0;
};

class PlaneCoords : public CoordGenerator
{
public:
    PlaneCoords() {};
    PlaneCoords(cv::Vec3f origin_, cv::Vec3f normal_);
    virtual float pointDist(cv::Vec3f wp);
    virtual cv::Vec3f project(cv::Vec3f wp, const cv::Rect &roi, float render_scale = 1.0, float coord_scale = 1.0);
    virtual void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) const;
    // float plane_mul() const;
    virtual float scalarp(cv::Vec3f point) const;
    virtual float height(cv::Vec3f point) const { return 0.0; };
    // virtual void gen_coords(float i, float j, int x, int y, float render_scale, float coord_scale) const = 0;
    using CoordGenerator::gen_coords;
    cv::Vec3f origin = {0,0,0};
    cv::Vec3f normal = {1,1,1};
};

class IDWHeightPlaneCoords : public PlaneCoords
{
public:
    IDWHeightPlaneCoords(std::vector<cv::Vec3f> *control_points_) : control_points(control_points_) {};
    virtual float scalarp(cv::Vec3f point) const;
    virtual float height(cv::Vec3f point) const;
    virtual void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) const;
    std::vector<cv::Vec3f> *control_points;
    // float pointDist(cv::Vec3f wp);
    // cv::Vec3f project(cv::Vec3f wp, const cv::Rect &roi, float render_scale = 1.0, float coord_scale = 1.0);
    // virtual void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) const;
    // virtual void gen_coords(float i, float j, int x, int y, float render_scale, float coord_scale) const = 0;
    // using CoordGenerator::gen_coords;
};

class PlaneIDWSegmentator
{
public:
    PlaneIDWSegmentator();
    //add a new point to the model
    void add(cv::Vec3f wp, cv::Vec3f normal);
    //move void move(...) (this can be in 2d or 3d and in-plane or out-of-plane .. think about which variants are relevant...)
    PlaneCoords *generator() const;
    std::vector<cv::Vec3f> control_points;
private:
    std::vector<std::pair<cv::Vec2f,cv::Vec3f>> _points;
    IDWHeightPlaneCoords *_generator = nullptr;
};

void find_intersect_segments(std::vector<std::vector<cv::Point2f>> &segments_roi, const PlaneCoords *other, const CoordGenerator *roi_gen, const cv::Rect roi, float render_scale, float coord_scale);


//TODO generation overrun
//TODO groupkey overrun
class ChunkCache
{
public:
    ChunkCache(size_t size) : _size(size) {};
    ~ChunkCache();
    
    //get key for a subvolume - should be uniqueley identified between all groups and volumes that use this cache.
    //for example by using path + group name
    uint64_t groupKey(std::string name);
    
    //key should be unique for chunk and contain groupkey (groupkey sets highest 16bits of uint64_t)
    void put(uint64_t key, xt::xarray<uint8_t> *ar);
    xt::xarray<uint8_t> *get(uint64_t key);
private:
    uint64_t _generation = 0;
    size_t _size = 0;
    size_t _stored = 0;
    std::unordered_map<uint64_t,xt::xarray<uint8_t>*> _store;
    //store generation number
    std::unordered_map<uint64_t,uint64_t> _gen_store;
    //store group keys
    std::unordered_map<std::string,uint64_t> _group_store;
};

//NOTE depending on request this might load a lot (the whole array) into RAM
void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache = nullptr);