#pragma once

#include <QObject>
#include <opencv2/core.hpp>

class Surface;

namespace ChaoVis
{
    
struct POI
{
    cv::Vec3f p = {0,0,0};
    Surface *src = nullptr;
    cv::Vec3f n = {0,0,0};
};

struct Intersection
{
    std::vector<std::vector<cv::Vec3f>> lines;
};

struct string_pair_hash {
    size_t operator()(const std::pair<std::string,std::string>& p) const
    {
        size_t hash1 = std::hash<std::string>{}(p.first);
        size_t hash2 = std::hash<std::string>{}(p.second);
        
        //magic numbers from boost. should be good enough
        return hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    }
};

//this class shall handle all the (gui) interactions for its stored objects but does not itself provide the gui
//slices: all the defined slices of all kinds
//Segmentators: segmentations and interactions with segments
//POIs : e.g. active constrol points or slicing focus points
class CSurfaceCollection : public QObject
{
    Q_OBJECT
    
public:
    void setSurface(const std::string &name, Surface*);
    void setPOI(const std::string &name, POI *poi);
    void setIntersection(const std::string &a, const std::string &b, Intersection *intersect);
    Surface *surface(const std::string &name);
    Intersection *intersection(const std::string &a, const std::string &b);
    POI *poi(const std::string &name);
    std::vector<std::string> surfaces();
    std::vector<std::string> pois();
    std::vector<std::pair<std::string,std::string>> intersections(const std::string &a = "");
    
signals:
    void sendSurfaceChanged(std::string, Surface*);
    void sendPOIChanged(std::string, POI*);
    void sendIntersectionChanged(std::string, std::string, Intersection*);
    
protected:
    bool _regular_pan = false;
    std::unordered_map<std::string, Surface*> _surfs;
    std::unordered_map<std::string, POI*> _pois;
    std::unordered_map<std::pair<std::string,std::string>, Intersection*, string_pair_hash> _intersections;
};

}
