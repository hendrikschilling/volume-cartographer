#include "CSurfaceCollection.hpp"

#include "vc/core/util/Slicing.hpp"

using namespace ChaoVis;

void CSurfaceCollection::setSurface(const std::string &name, Surface* surf)
{
    _surfs[name] = surf;
    sendSurfaceChanged(name, surf);
}

void CSurfaceCollection::setPOI(const std::string &name, POI *poi)
{
    _pois[name] = poi;
    sendPOIChanged(name, poi);
}

Surface* CSurfaceCollection::surface(const std::string &name)
{
    if (!_surfs.count(name))
        return nullptr;
    return _surfs[name];
}

POI *CSurfaceCollection::poi(const std::string &name)
{
    if (!_pois.count(name))
        return nullptr;
    return _pois[name];
}

std::vector<std::string> CSurfaceCollection::surfaces()
{
    std::vector<std::string> keys;
    for(auto &it : _surfs)
        keys.push_back(it.first);
    
    return keys;
}

std::vector<std::string> CSurfaceCollection::pois()
{
    std::vector<std::string> keys;
    for(auto &it : _pois)
        keys.push_back(it.first);

    return keys;
}

void CSurfaceCollection::setIntersection(const std::string &a, const std::string &b, Intersection *intersect)
{
    _intersections[{a,b}] = intersect;
    sendIntersectionChanged(a, b, intersect);
}

Intersection *CSurfaceCollection::intersection(const std::string &a, const std::string &b)
{
    if (_intersections.count({a,b}))
        return _intersections[{a,b}];
        
    if (_intersections.count({b,a}))
        return _intersections[{b,a}];
    
    return nullptr;
}

std::vector<std::pair<std::string,std::string>> CSurfaceCollection::intersections(const std::string &a)
{
    std::vector<std::pair<std::string,std::string>> res;

    if (!a.size()) {
        for(auto item : _intersections)
            res.push_back(item.first);
    }
    else
        for(auto item : _intersections) {
            if (item.first.first == a)
                res.push_back(item.first);
            else if (item.first.second == a)
                res.push_back(item.first);
        }
    return res;
}
