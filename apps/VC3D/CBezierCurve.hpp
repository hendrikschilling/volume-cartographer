// CBezierCurve.h
// Chao Du 2015 April
#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "MathUtils.hpp"
#include <QGraphicsEllipseItem>
#include <QGraphicsView>
#include <QGraphicsScene>

namespace ChaoVis
{

// REVISIT - NOTE - we use order 3 Bezier curve, which has 4 control points and
// 3 convex line segments

class CBezierCurve
{

public:
    CBezierCurve(int nSampleInterval = 5.0);
    CBezierCurve(
        const std::vector<Vec2<double>>& nControlPoints,
        int nSampleInterval = 5.0);

    void SetControlPoints(const std::vector<Vec2<double>>& nControlPoints);
    void SetControlPoints(const std::vector<cv::Vec2f>& nControlPoints);
    void GetSamplePoints(std::vector<Vec2<double>>& nSamplePoints);
    void GetSamplePoints(std::vector<cv::Vec2f>& nSamplePoints);

    void DrawOnImage(QGraphicsScene* scene, const QColor& color = QColor(0, 0, 255));

protected:
private:
    int fNumControlPoints;  // constantly equals 4
    std::vector<Vec2<double>> fControlPoints;
    double fSampleInterval;

};  // class CBezierCurve

}  // namespace ChaoVis
