// CVolumeViewer.cpp
// Chao Du 2015 April
#include "CVolumeViewer.hpp"
#include "UDataManipulateUtils.hpp"
#include "HBase.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>

#include "CVolumeViewerView.hpp"
#include "SegmentationStruct.hpp"
#include "CSurfaceCollection.hpp"

#include "vc/core/util/Slicing.hpp"

using namespace ChaoVis;
using qga = QGuiApplication;

#define BGND_RECT_MARGIN 8
#define DEFAULT_TEXT_COLOR QColor(255, 255, 120)
// #define ZOOM_FACTOR 1.148698354997035
#define ZOOM_FACTOR 2.0 //1.414213562373095

// Constructor
CVolumeViewer::CVolumeViewer(CSurfaceCollection *slices, QWidget* parent)
    : QWidget(parent)
    , fCanvas(nullptr)
    // , fScrollArea(nullptr)
    , fGraphicsView(nullptr)
    , fZoomInBtn(nullptr)
    , fZoomOutBtn(nullptr)
    , fResetBtn(nullptr)
    , fNextBtn(nullptr)
    , fPrevBtn(nullptr)
    , fImgQImage(nullptr)
    , fBaseImageItem(nullptr)
    , fScanRange(1)
    , _slice_col(slices)
{
    fBaseImageItem = nullptr;

    // Create graphics view
    fGraphicsView = new CVolumeViewerView(this);
    
    
    fGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    fGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    
    fGraphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);
    
    fGraphicsView->setRenderHint(QPainter::Antialiasing);
    // setFocusProxy(fGraphicsView);
    connect(fGraphicsView, &CVolumeViewerView::sendScrolled, this, &CVolumeViewer::onScrolled);
    connect(fGraphicsView, &CVolumeViewerView::sendVolumeClicked, this, &CVolumeViewer::onVolumeClicked);
    connect(fGraphicsView, &CVolumeViewerView::sendZoom, this, &CVolumeViewer::onZoom);
    connect(fGraphicsView, &CVolumeViewerView::sendCursorMove, this, &CVolumeViewer::onCursorMove);

    // Create graphics scene
    fScene = new QGraphicsScene({-2500,-2500,5000,5000}, this);

    // Set the scene
    fGraphicsView->setScene(fScene);

    QSettings settings("VC.ini", QSettings::IniFormat);
    // fCenterOnZoomEnabled = settings.value("viewer/center_on_zoom", false).toInt() != 0;
    // fScrollSpeed = settings.value("viewer/scroll_speed", false).toInt();
    fSkipImageFormatConv = settings.value("perf/chkSkipImageFormatConvExp", false).toBool();

    QVBoxLayout* aWidgetLayout = new QVBoxLayout;
    aWidgetLayout->addWidget(fGraphicsView);

    setLayout(aWidgetLayout);
}

// Destructor
CVolumeViewer::~CVolumeViewer(void)
{
    deleteNULL(fGraphicsView);
    deleteNULL(fScene);
}

void CVolumeViewer::SetImage(const QImage& nSrc)
{
    if (fImgQImage == nullptr) {
        fImgQImage = new QImage(nSrc);
    } else {
        *fImgQImage = nSrc;
    }

    // Create a QPixmap from the QImage
    QPixmap pixmap = QPixmap::fromImage(*fImgQImage, fSkipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);

    // Add the QPixmap to the scene as a QGraphicsPixmapItem
    if (!fBaseImageItem) {
        fBaseImageItem = fScene->addPixmap(pixmap);
    } else {
        fBaseImageItem->setPixmap(pixmap);
    }
    update();
}

void round_scale(float &scale)
{
    if (abs(scale-round(log2(scale))) < 0.02)
        scale = pow(2,round(log2(scale)));
}

//get center of current visible area in scene coordinates
QPointF visible_center(QGraphicsView *view)
{
    QRectF bbox = view->mapToScene(view->viewport()->geometry()).boundingRect();
    return bbox.topLeft() + QPointF(bbox.width(),bbox.height())*0.5;
}


void CVolumeViewer::onCursorMove(QPointF scene_loc)
{
    if (!_slice)
        return;
    
    cv::Vec3f slice_loc = {scene_loc.x()/_ds_scale, scene_loc.y()/_ds_scale,0};

    POI *cursor = _slice_col->poi("cursor");
    if (!cursor)
        cursor = new POI;
    
    cursor->p = _slice->coord_legacy(slice_loc);
    
    _slice_col->setPOI("cursor", cursor);
}

void CVolumeViewer::onZoom(int steps, QPointF scene_loc, Qt::KeyboardModifiers modifiers)
{
    //TODO don't invalidate if only _scene_scale chagned
    invalidateVis();
    invalidateIntersect();
    
    if (!_slice)
        return;
    
    if (modifiers & Qt::ShiftModifier) {
        _slice->setOffsetZ(_slice->offsetZ()+steps);
        _slice_col->setSlice(_slice_name, _slice);
    }
    else {
        float zoom = pow(ZOOM_FACTOR, steps);
        
        _scale *= zoom;
        round_scale(_scale);
        
        if (_scale >= _max_scale) {
            _ds_scale = _max_scale;
            _ds_sd_idx = -log2(_ds_scale);
            _scene_scale = _scale/_ds_scale;
        }
        else if (_scale < _min_scale) {
            _ds_scale = _min_scale;
            _ds_sd_idx = -log2(_ds_scale);
            _scene_scale = _scale/_ds_scale;
        }
        else {
            _ds_sd_idx = -log2(_scale);
            _ds_scale = pow(2,-_ds_sd_idx);
            _scene_scale = _scale/_ds_scale;
        }
        
        QTransform M = fGraphicsView->transform();
        if (_scene_scale != M.m11()) {
            double delta_scale = _scene_scale/M.m11();
            M.scale(delta_scale,delta_scale);
            fGraphicsView->setTransform(M);
        }
        
        curr_img_area = {0,0,0,0};
        QPointF center = visible_center(fGraphicsView) * zoom;
        
        //FIXME get correct size for slice!
        int max_size = std::max(volume->sliceWidth(), std::max(volume->numSlices(), volume->sliceHeight()))*_ds_scale + 512;
        fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
        
        fGraphicsView->centerOn(center);
        renderVisible();
    }
}

void CVolumeViewer::OnVolumeChanged(volcart::Volume::Pointer volume_)
{
    volume = volume_;
    
    printf("sizes %d %d %d\n", volume_->sliceWidth(), volume_->sliceHeight(), volume_->numSlices());
    
    int max_size = std::max(volume_->sliceWidth(), std::max(volume_->numSlices(), volume_->sliceHeight()))*_ds_scale + 512;
    printf("max size %d\n", max_size);
    fGraphicsView->setSceneRect(-max_size/2,-max_size/2,max_size,max_size);
    
    //FIXME currently hardcoded
    _max_scale = 0.5;
    _min_scale = pow(2.0,1.-volume->numScales());

    renderVisible(true);
}

cv::Vec3f loc3d_at_imgpos(volcart::Volume *vol, CoordGenerator *slice, QPointF loc, float scale)
{
    xt::xarray<float> coords;
    
    int sd_idx = 1;
    
    float round_scale = 0.5;
    while (0.5*round_scale >= scale && sd_idx < vol->numScales()-1) {
        sd_idx++;
        round_scale *= 0.5;
    }
    
    slice->gen_coords(coords, loc.x()*round_scale/scale,loc.y()*round_scale/scale, 1, 1, scale/round_scale, round_scale);
    
    coords /= round_scale;
    
    return {coords(0,0,2),coords(0,0,1),coords(0,0,0)};
}

void CVolumeViewer::onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    if (!_slice)
        return;

    cv::Vec3f slice_loc = {scene_loc.x()/_ds_scale, scene_loc.y()/_ds_scale,0};
    
    cv::Vec3f n = _slice->normal_legacy(slice_loc);
    cv::Vec3f p = _slice->coord_legacy(slice_loc);
    

    sendVolumeClicked(p, n, _slice, slice_loc, buttons, modifiers);
}

void CVolumeViewer::setCache(ChunkCache *cache_)
{
    cache = cache_;
}

void CVolumeViewer::setSlice(const std::string &name)
{
    _slice_name = name;
    _slice = nullptr;
    onSliceChanged(name, _slice_col->slice(name));
}


void CVolumeViewer::invalidateVis()
{
    _slice_vis_valid = false;    
    for(auto &item : slice_vis_items) {
        fScene->removeItem(item);
        delete item;
    }
    slice_vis_items.resize(0);
}

void CVolumeViewer::invalidateIntersect()
{
    _intersect_valid = false;    
    for(auto &item : _intersect_items) {
        fScene->removeItem(item);
        delete item;
    }
    _intersect_items.resize(0);
}

void CVolumeViewer::onSliceChanged(std::string name, CoordGenerator *slice)
{
    //TODO distinguis different elements for rendering! (slice, intersect, control points) completely separately!
    if (_slice_name == "segmentation")
        invalidateIntersect();
    
    if (_slice_name == name) {
        _slice = slice;
        if (!_slice)
            fScene->clear();
        else
            invalidateVis();
    }
    
    //FIXME do not re-render slice if only segmentation changed?
    if (name == _slice_name || name == "segmentation") {
        curr_img_area = {0,0,0,0};
        renderVisible();
    }
}

QGraphicsItem *cursorItem()
{
    QPen pen(QBrush(Qt::cyan), 3);
    QGraphicsLineItem *parent = new QGraphicsLineItem(-10, 0, -5, 0);
    parent->setZValue(10);
    parent->setPen(pen);
    QGraphicsLineItem *line = new QGraphicsLineItem(10, 0, 5, 0, parent);
    line->setPen(pen);
    line = new QGraphicsLineItem(0, -10, 0, -5, parent);
    line->setPen(pen);
    line = new QGraphicsLineItem(0, 10, 0, 5, parent);
    line->setPen(pen);
    
    return parent;
}

QGraphicsItem *crossItem()
{
    QPen pen(QBrush(Qt::red), 1);
    QGraphicsLineItem *parent = new QGraphicsLineItem(-5, -5, 5, 5);
    parent->setZValue(10);
    parent->setPen(pen);
    QGraphicsLineItem *line = new QGraphicsLineItem(-5, 5, 5, -5, parent);
    line->setPen(pen);
    
    return parent;
}

//TODO make poi tracking optional and configurable
void CVolumeViewer::onPOIChanged(std::string name, POI *poi)
{    
    if (!poi)
        return;
    
    if (name == "focus") {
        PlaneCoords *plane = dynamic_cast<PlaneCoords*>(_slice);
        
        if (!plane)
            return;
        
        fGraphicsView->centerOn(0,0);
        
        if (poi->p == plane->origin)
            return;
        
        plane->origin = poi->p;
        
        _slice_col->setSlice(_slice_name, plane);
    }
    else if (name == "cursor") {
        PlaneCoords *slice_plane = dynamic_cast<PlaneCoords*>(_slice);
        
        if (slice_plane) {
            
            if (!_cursor) {
                _cursor = cursorItem();
                fScene->addItem(_cursor);
            }
            
            float dist = slice_plane->pointDist(poi->p);
            
            if (dist < 100.0/_ds_scale) {
                cv::Vec3f sp = slice_plane->project(poi->p, 1.0, _ds_scale);
                
                _cursor->setPos(sp[0], sp[1]);
                _cursor->setOpacity(1.0-dist*_ds_scale/100.0);
            }
            else
                _cursor->setOpacity(0.0);
        }
    }
}

void CVolumeViewer::onSegmentatorChanged(std::string name, ControlPointSegmentator *seg)
{
    if (name != "default")
        return;

    _seg_tool = seg;
}

cv::Mat CVolumeViewer::render_area(const cv::Rect &roi)
{
    xt::xarray<float> coords;
    xt::xarray<uint8_t> img;

    _slice->gen_coords(coords, roi, 1.0, _ds_scale);
    readInterpolated3D(img, volume->zarrDataset(_ds_sd_idx), coords, cache);
    cv::Mat m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    
    return m.clone();
}

class LifeTime
{
public:
    LifeTime(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~LifeTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

void CVolumeViewer::renderVisible(bool force)
{
    if (!volume || !volume->zarrDataset() || !_slice)
        return;
    
    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();
    
    if (!force && QRectF(curr_img_area).contains(bbox))
        return;
    
    curr_img_area = {bbox.left()-128,bbox.top()-128, bbox.width()+256, bbox.height()+256};
    invalidateIntersect();
    
    cv::Mat img = render_area({curr_img_area.x(), curr_img_area.y(), curr_img_area.width(), curr_img_area.height()});
    
    QImage qimg = Mat2QImage(img);
    
    QPixmap pixmap = QPixmap::fromImage(qimg, fSkipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);
    //     
    // Add the QPixmap to the scene as a QGraphicsPixmapItem
    if (!fBaseImageItem)
        fBaseImageItem = fScene->addPixmap(pixmap);
    else
        fBaseImageItem->setPixmap(pixmap);
    
    if (!_center_marker) {
        _center_marker = fScene->addEllipse({-10,-10,20,20}, QPen(Qt::yellow, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _center_marker->setZValue(11);
        
    }

    
    _center_marker->setParentItem(fBaseImageItem);
    
    fBaseImageItem->setOffset(curr_img_area.topLeft());
    PlaneCoords *slice_plane = dynamic_cast<PlaneCoords*>(_slice);
    GridCoords *slice_segment = dynamic_cast<GridCoords*>(_slice_col->slice("segmentation"));
    
    if (!_intersect_valid && slice_plane && slice_segment) {
        std::vector<std::vector<cv::Vec2f>> xy_seg_;
        std::vector<std::vector<cv::Vec3f>> intersections;
        
        cv::Rect plane_roi = {curr_img_area.x()/_ds_scale, curr_img_area.y()/_ds_scale, curr_img_area.width()/_ds_scale, curr_img_area.height()/_ds_scale};
        
        find_intersect_segments(intersections, xy_seg_, slice_segment->_points, slice_plane, plane_roi, 4/_ds_scale);
    

        for (auto seg : intersections) {
            QColor col(128+rand()%127, 128+rand()%127, 128+rand()%127);
            QPainterPath path;

            bool first = true;
            for (auto wp : seg)
            {
                cv::Vec3f p = slice_plane->project(wp, 1.0, _ds_scale);
                if (first)
                    path.moveTo(p[0],p[1]);
                else
                    path.lineTo(p[0],p[1]);
                first = false;
            }
            auto item = fGraphicsView->scene()->addPath(path, QPen(Qt::yellow, 1/_scene_scale));
            item->setZValue(5);
            _intersect_items.push_back(item);
        }
    }
        
    /*if (!_slice_vis_valid && _seg_tool && slice_plane) {
#pragma omp parallel for
        for (auto &wp : _seg_tool->control_points) {
            float dist = slice_plane->pointDist(wp);
            
            if (dist > 0.5)
                continue;
            
            cv::Vec3f p = slice_plane->project(wp, 1.0, _ds_scale);
            
#pragma omp critical
            {
                // auto item = crossItem();
                // item->setPos(p[0], p[1]);
                auto item = fGraphicsView->scene()->addEllipse({-1,-1,2,2}, QPen(Qt::red, 1));
                item->setZValue(8);
                item->setPos(p[0],p[1]);
                //FIXME rename/clean
                slice_vis_items.push_back(item);
            }
        }
        
        if (_seg_tool->control_points.size())
            _slice_vis_valid = true;
    }*/
}

void CVolumeViewer::onScrolled()
{
    renderVisible();
}

cv::Mat CVolumeViewer::getCoordSlice()
{
    return cv::Mat();
}
