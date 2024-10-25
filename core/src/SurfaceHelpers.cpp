#include "SurfaceHelpers.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/calib3d.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"

static std::ostream& operator<< (std::ostream& out, const std::vector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

static std::ostream& operator<< (std::ostream& out, const std::vector<int> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

template <size_t N>
std::ostream& operator<< (std::ostream& out, const std::array<size_t,N> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

static std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

/*static void timed_plane_slice(PlaneCoords &plane, z5::Dataset *ds, size_t size, ChunkCache *cache, std::string msg)
{
    xt::xarray<float> coords;
    xt::xarray<uint8_t> img;
    
    auto start = std::chrono::high_resolution_clock::now();
    plane.gen_coords(coords, size, size);
    auto end = std::chrono::high_resolution_clock::now();
    // std::cout << std::chrono::duration<double>(end-start).count() << "s gen_coords() " << msg << std::endl;
    start = std::chrono::high_resolution_clock::now();
    readInterpolated3D(img, ds, coords, cache);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s slicing " << msg << std::endl; 
}*/

static cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;
    
    cv::Vec3f p00 = points(y,x);
    cv::Vec3f p01 = points(y,x+1);
    cv::Vec3f p10 = points(y+1,x);
    cv::Vec3f p11 = points(y+1,x+1);
    
    cv::Vec3f p0 = (1-fx)*p00 + fx*p01;
    cv::Vec3f p1 = (1-fx)*p10 + fx*p11;
    
    return (1-fy)*p0 + fy*p1;
}

static float atf_int(const cv::Mat_<float> &points, cv::Vec2f p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;

    float p00 = points(y,x);
    float p01 = points(y,x+1);
    float p10 = points(y+1,x);
    float p11 = points(y+1,x+1);

    float p0 = (1-fx)*p00 + fx*p01;
    float p1 = (1-fx)*p10 + fx*p11;

    return (1-fy)*p0 + fy*p1;
}

static float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d.dot(d);
}

static void min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, bool z_search = true)
{
    // std::cout << "start minlo" << loc << std::endl;
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        // loc = {-1,-1};
        // printf("init fail %d %d\n", loc[0], loc[1]);
        return;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    // std::cout << "at" << loc << val << std::endl;
    out = val;
    float best = sdist(val, tgt);
    printf("init dist %f\n", best);
    float res;
    
    std::vector<cv::Vec2f> search;
    if (z_search)
        search = {{0,-1},{0,1},{-1,0},{1,0}};
    else
        search = {{1,0},{-1,0}};

    float step = 1.0;
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!boundary.contains({cand[0],cand[1]})) {
                out = {-1,-1,-1};
                // loc = {-1,-1};
                return;
            }
                
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = sdist(val,tgt);
            if (res < best) {
                // std::cout << res << tgt << val << step << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
                // std::cout << "(" << res << tgt << val << step << "\n";
        }
        
        if (!changed && step > 0.125) {
            step *= 0.5;
            changed = true;
        }
    }
    
    std::cout << "best" << best << tgt << out << "\n" <<  std::endl;
}

static float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist)
{
    cv::Vec3f d = a-b;
    float l = sqrt(d.dot(d));
    
    return abs(l-t_dist);
}

static float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, const std::vector<float> ws = {})
{
    if (!ws.size()) {
        float sum = 0;
        for(int i=0;i<tgts.size();i++) {
            float d = tdist(v, tgts[i], tds[i]);
            sum += d*d;///tds[i]*20;
        }
        
        return sum;
    }
    else {
        float sum = 0;
        for(int i=0;i<tgts.size();i++) {
            float d = tdist(v, tgts[i], tds[i]);
            sum += d*d*ws[i];
        }
        
        return sum;
    }
}

static void min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, bool z_search = true)
{
    // std::cout << "start minlo" << loc << std::endl;
    cv::Rect boundary(1,1,points.cols-1,points.rows-1);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        return;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds);
    float res;
    
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    std::vector<cv::Vec2f> search;
    if (z_search)
        search = {{0,-1},{0,1},{-1,0},{1,0}};
    else
        search = {{1,0},{-1,0}};
    float step = 16.0;
    
    
    // std::cout << "init " << best << tgts[0] << val << loc << "\n";
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!boundary.contains({cand[0],cand[1]})) {
                out = {-1,-1,-1};
                loc = {-1,-1};
                return;
            }
            
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds);
            if (res < best) {
                // std::cout << res << tgts[0] << val << step << cand << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
                // std::cout << "(" << res << tgts[0] << val << step << cand << "\n";
        }
        
        if (!changed && step > 0.125) {
            step *= 0.5;
            changed = true;
        }
    }
    
    // std::cout << "best" << best << tgts[0] << out << "\n" <<  std::endl;
}

//this works surprisingly well, though some artifacts where original there was a lot of skew
//FIXME add to api ...
static cv::Mat_<cv::Vec3f> derive_regular_region_stupid_gauss(cv::Mat_<cv::Vec3f> points)
{
    cv::Mat_<cv::Vec3f> out = points.clone();
    cv::Mat_<cv::Vec3f> blur(points.cols, points.rows);
    cv::Mat_<cv::Vec2f> locs(points.size());
    
    cv::Mat trans = out.t();
    
#pragma omp parallel for
    for(int j=0;j<trans.rows;j++) 
        cv::GaussianBlur(trans({0,j,trans.cols,1}), blur({0,j,trans.cols,1}), {255,1}, 0);

    blur = blur.t();
    
#pragma omp parallel for
    for(int j=1;j<points.rows;j++)
        for(int i=1;i<points.cols-1;i++) {
            // min_loc(points, {i,j}, out(j,i), {out(j,i)[0],out(j,i)[1],out(j,i)[2]});
            cv::Vec2f loc = {i,j};
            min_loc(points, loc, out(j,i), blur(j,i), false);
        }
        
    return out;
}

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

static float min_loc_dbg(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneSurface *plane, cv::Vec2f init_step, float min_step_f, const std::vector<float> &ws = {}, bool robust_edge = false, const cv::Mat_<float> &used = cv::Mat())
{
    // std::cout << "start minlo" << loc << std::endl;
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds, ws);
    if (plane) {
        float d = plane->pointDist(val);
        best += d*d;
    }
    if (!used.empty())
        best += atf_int(used, loc)*100.0;
    float res;
    
    //TODO add more search patterns, compare motion estimatino for video compression, x264/x265, ...
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    cv::Vec2f step = init_step;
    
    
    // std::cout << "init " << best << tgts[0] << val << loc << "\n";
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);
            
            if (!boundary.contains({cand[0],cand[1]})) {
                if (!robust_edge || (step[0] < min_step_f*init_step[0])) {
                    out = {-1,-1,-1};
                    loc = {-1,-1};
                    return -1;
                }
                else
                    //skip to next scale
                    break;
            }
            
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds, ws);
            if (!used.empty())
                res += atf_int(used, loc)*100.0;
            if (plane) {
                float d = plane->pointDist(val);
                res += d*d;
            }
            if (res < best) {
                // std::cout << res << val << step << cand << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
            // std::cout << "(" << res << val << step << cand << "\n";
        }
        
        if (changed)
            continue;
        
        step *= 0.5;
        changed = true;
        
        if (step[0] < min_step_f*init_step[0])
            break;
    }
    
    // std::cout << "best" << best << out << "\n" <<  std::endl;
    return sqrt(best/tgts.size());
}

template<typename T> std::vector<T> join(const std::vector<T> &a, const std::vector<T> &b)
{
    std::vector<T> c = a;
    c.insert(c.end(), b.begin(), b.end());
    
    return c;
}

static float multi_step_search(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds_, PlaneSurface *plane, cv::Vec2f init_step, const std::vector<cv::Vec3f> &opt_t, const std::vector<float> &opt_d, int &failstate, const std::vector<float> &ws, float th, const cv::Mat_<float> &used = cv::Mat())
{
    // std::cout << init << loc << std::endl;
    failstate = 0;
    cv::Vec2f init_loc = loc;
    cv::Vec3f first_res;
    
    // std::vector<cv::Vec3f> t2 = join(tgts, opt_t);
    // std::vector<float> d2 = join(tds, opt_d);
    std::vector<float> w2 = ws;
    std::vector<float> tds = tds_;
    float w_sum = 0;
    for(auto &w : w2) {
        w *= w;
        w_sum += w;
    }
    // w2.push_back(10);
    // tds.push_back(0);
    
    float res1 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
    float res2 = 0;
    float res3;
    // res3 = min_loc_dbg(points, loc, out, join(tgts,{out}), tds, plane, init_step*0.1, 0.1, w2);
    cv::Vec3f val;
    if (res1 >= 0) {
        val = at_int(points, loc);
        res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
    }
    else
        res3 = res1;
    
    if (res3 < th)
        return res3;

    // printf("start it %f\n", res3);
    
    float best_res = res3;
    cv::Vec2f best_loc = loc;
    for(int i=0;i<10;i++)
    {
        cv::Vec2f off = {rand()%100,rand()%100};
        off -= cv::Vec2f(50,50);
        off = mul(off, init_step)*100/50;
        loc = init_loc + off;
        
        res1 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
        // res3 = min_loc_dbg(points, loc, out, join(tgts,{out}), tds, plane, init_step*0.1, 0.1, w2);
        // res3 = min_loc_dbg(points, loc, out, join(tgts,{out}), tds, plane, init_step*0.1, 0.1, w2);
        if (res1 >= 0) {
            val = at_int(points, loc);
            res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
            if (res1 < best_res) {
                best_res = res1;
                best_loc = loc;
            }
        }
        else
            res3 = res1;
        
        // printf("   %f %f\n", res3, res1);
        
        if (res3 < th) {
            // printf("  it %f (%f)\n", res3, res1);
            return res3;
        }
    }
    
    loc = best_loc;
    res1 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
    // res3 = min_loc_dbg(points, loc, out, join(tgts,{out}), tds, plane, init_step*0.1, 0.1, w2);
    if (res1 >= 0) {
        val = at_int(points, loc);
        res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
    }
    else
        res3 = res1;
    // res3 = min_loc_dbg(points, loc, out, join(tgts,{out}), tds, plane, init_step*0.1, 0.1, w2);
    // printf("  fallback %f (%f %f)\n", res3, res2, res1);
    
    
    //lets brute force this for once
    /*for (int j=0;j<100;j++)
        for (int i=0;i<100;i++) {
            loc = init_loc + 0.1*mul(init_step,cv::Vec2f(j-50,i-50));
            // res1 = min_loc_dbg(points, loc, out, t2, d2, plane, init_step, 0.1, ws);
            cv::Vec3f val = at_int(points, loc);
            res1 = sqrt(tdist_sum(val, t2, d2, ws)/t2.size());
            printf("brute %f\n", res1);
        }*/
    
    
    
    failstate = 1;
    
    return -abs(res3);
}

//lets try again
//FIXME mark covered regions as not available so we can't repeat them'
cv::Mat_<cv::Vec3f> derive_regular_region_largesteps(const cv::Mat_<cv::Vec3f> &points, cv::Mat_<cv::Vec2f> &locs, int seed_x, int seed_y, float step_size, int w, int h)
{
    double sx, sy;
    vc_segmentation_scales(points, sx, sy);
    
    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;
    std::vector<cv::Vec2i> setfail;
    std::vector<cv::Vec2i> collected_failures;
    std::vector<cv::Vec2i> skipped;
    
    std::cout << "input avg step " << sx << " " << sy << points.size() << std::endl;
    
    //TODO use scaling and average diffeence vecs for init?
    float D = sqrt(2);
    
    float T = step_size;
    
    float th = T/4;
    
    int r = 4;
    
    cv::Vec2f step = {sx*T/10, sy*T/10};
    
    cv::Mat_<cv::Vec3f> out(h,w);
    locs.create(h,w);
    cv::Mat_<uint8_t> state(h,w);
    cv::Mat_<float> dbg(h,w);
    cv::Mat_<float> x_curv(h,w);
    cv::Mat_<float> y_curv(h,w);
    cv::Mat_<float> used(points.size());
    out.setTo(-1);
    used.setTo(0);
    state.setTo(0);
    dbg.setTo(0);
    x_curv.setTo(1);
    y_curv.setTo(1);
    
    cv::Rect src_bounds(0,0,points.cols-3,points.rows-3);
    if (!src_bounds.contains({seed_x,seed_y}))
        return out;

    //FIXME the init locations are probably very important!
    
    //FIXME local search can be affected by noise/artefacts in data, add some re-init random initilizations if we see failures?
    
    int x0 = w/2;
    int y0 = h/2;

    cv::Rect used_area(x0,y0,1,1);
    locs(y0,x0) = {seed_x, seed_y};
    out(y0,x0) = at_int(points, locs(y0,x0));
    
    float res;
    
    
    //first point to the right
    locs(y0,x0+1) = locs(y0,x0)+cv::Vec2f(1,0);
    res = min_loc_dbg(points, locs(y0,x0+1), out(y0,x0+1), {out(y0,x0)}, {T}, nullptr, step, 0.01);
    std::cout << res << std::endl;
    
    //bottom left
    locs(y0+1,x0) = locs(y0,x0)+cv::Vec2f(0,1);
    res = min_loc_dbg(points, locs(y0+1,x0), out(y0+1,x0), {out(y0,x0),out(y0,x0+1)}, {T,D*T}, nullptr, step, 0.01);
    std::cout << res << std::endl;
    
    //bottom right
    locs(y0+1,x0+1) = locs(y0,x0)+cv::Vec2f(1,1);
    res = min_loc_dbg(points, locs(y0+1,x0+1), out(y0+1,x0+1), {out(y0,x0),out(y0,x0+1),out(y0+1,x0)}, {D*T,T,T}, nullptr, step, 0.01);
    std::cout << res << std::endl;
    
    std::cout << out(y0,x0) << out(y0,x0+1) << std::endl;
    std::cout << out(y0+1,x0) << out(y0+1,x0+1) << std::endl;
    
    // locs(j,i) = locs(j-1,i);
    
    // std::vector<cv::Vec3f> refs;
    // std::vector<float> dists;
    
    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};
    cv::Rect bounds(0,0,h-1,w-1);
    
    state(y0,x0) = 1;
    state(y0+1,x0) = 1;
    state(y0,x0+1) = 1;
    state(y0+1,x0+1) = 1;
    
    fringe.push_back({y0,x0});
    fringe.push_back({y0+1,x0});
    fringe.push_back({y0,x0+1});
    fringe.push_back({y0+1,x0+1});
    
    int succ = 0;
    int total_fail = 0;
    int last_round_updated = 4;
    
    std::vector<cv::Vec2f> all_locs;
    bool skipped_from_skipped = false;
    int generation = 0;
    int stop_gen = -1;
    bool ignore_failures = false;
    
    while (fringe.size() || (!skipped_from_skipped && skipped.size()) || collected_failures.size()) {
        generation++;
        if (generation == stop_gen)
            break;
        last_round_updated = 0;

        //first: regular fring
        if (fringe.size()) {
            for(auto p : fringe)
            {
                if (state(p) != 1)
                    continue;
                
                // std::cout << "check " << p << std::endl;
                
                for(auto n : neighs)
                    if (bounds.contains(p+n) && state(p+n) == 0) {
                        state(p+n) = 2;
                        cands.push_back(p+n);
                        // std::cout << "cand  " << p+n << std::endl;
                    }
            }
            for(auto p : cands)
                state(p) = 0;
            printf("gen %d processing %d fringe cands (total succ/fail %d/%d fringe: %d skipped: %d failures: %d\n", generation, cands.size(), succ, total_fail, fringe.size(), skipped.size(), collected_failures.size());
            fringe.resize(0);
            skipped_from_skipped = false;
        }
        else if ((last_round_updated && skipped.size()) || collected_failures.size()) {
            //skipped && failed points are processed at the same time so we expand smoothly
            for(auto p : skipped)
                state(p) = 0;
            cands = skipped;
            printf("gen %d processing %d skipped cands (total succ/fail %d/%d fringe: %d skipped: %d failures: %d\n", generation, cands.size(), succ, total_fail, fringe.size(), skipped.size(), collected_failures.size());
            skipped.resize(0);
            skipped_from_skipped = true;

            for(auto p : collected_failures) {
                state(p) = 0;
                cands.push_back(p);
            }
            printf("gen %d processing %d fail cands (total succ/fail %d/%d fringe: %d skipped: %d failures: %d\n", generation, cands.size(), succ, total_fail, fringe.size(), skipped.size(), collected_failures.size());
            collected_failures.resize(0);
            ignore_failures = true;
            skipped_from_skipped = false;
        }
        else
            break;

        cv::Mat_<cv::Vec3f> curv_data(2*r+1,2*r+1);
        cv::Mat_<uint8_t> curv_valid(2*r+1,2*r+1);
        
        for(auto p : cands) {
            if (state(p))
                continue;
            
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            std::vector<float> ws;
            std::vector<cv::Vec2i> dbg_outp;
            cv::Vec2f loc_sum = 0;
            int fail = 0;
            
            curv_valid.setTo(0);
            
            // printf("run %d %d\n",p[1],p[0]);
            
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,out.rows-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,out.cols-1);ox++)
                    if (state(oy,ox) == 1) {
                        int dy = oy-p[0];
                        int dx = ox-p[1];
                        curv_valid(dy+r,dx+r) = 1;
                        curv_data(dy+r,dx+r) = out(oy,ox);
                    }
            
            float x_curve_sum = 0;
            int x_curve_count = 0;
            for(int j=0;j<2*r+1;j++)
                for(int i=0;i<2*r+1-2;i++) {
                    if (curv_valid(j,i) && curv_valid(j,i+1) && curv_valid(j,i+2)) {
                        x_curve_sum += sqrt(sdist(curv_data(j,i),curv_data(j,i+2)))/(2*T);
                        x_curve_count++;
                        // printf("%f\n",sqrt(sdist(curv_data(j,i),curv_data(j,i+2)))/(2*T));
                    }
                }
            if (x_curve_count)
                x_curv(p) = sqrt(std::min(1.0f,x_curve_sum/x_curve_count));
            
            float y_curve_sum = 0;
            int y_curve_count = 0;
            for(int j=0;j<2*r+1-2;j++)
                for(int i=0;i<2*r+1;i++) {
                    if (curv_valid(j,i) && curv_valid(j+1,i) && curv_valid(j+2,i)) {
                        y_curve_sum += sqrt(sdist(curv_data(j,i),curv_data(j+2,i)))/(2*T);
                        y_curve_count++;
                        // printf("%f\n",sqrt(sdist(curv_data(j,i),curv_data(j,i+2)))/(2*T));
                    }
                }
                if (y_curve_count)
                    y_curv(p) = sqrt(std::min(1.0f,y_curve_sum/y_curve_count));
            
            // printf("avg curv xy %f %f\n",x_curv(p),y_curv(p));
            
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,out.rows-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,out.cols-1);ox++)
                    if (state(oy,ox) == 1) {
                        refs.push_back(out(oy,ox));
                        float curv_pow_x = pow(x_curv(p),abs(ox-p[1]));
                        float curv_pow_y = pow(y_curv(p),abs(oy-p[0]));
                        float dy = abs(oy-p[0])*curv_pow_x;
                        float dx = abs(ox-p[1])*curv_pow_y;
                        // float dy = (oy-p[0])*y_curv(p);
                        // float dx = (ox-p[1])*x_curv(p);
                        // float dy = (oy-p[0]);
                        // float dx = (ox-p[1]);
                        float d = sqrt(dy*dy+dx*dx);
                        dists.push_back(T*d);
                        loc_sum += locs(oy,ox);
                        // float w = 1.0/(std::max(abs(dbg(oy,ox)),1.0f));
                        float w = 1*curv_pow_x*curv_pow_y/d;
                        ws.push_back(w);
                        dbg_outp.push_back({dy,dx});
                    }
                    else if (state(oy,ox) == 10)
                        fail++;
            // else if (state(oy,ox) == 10)
            // fail++;
                    
            locs(p) = loc_sum*(1.0/dists.size());
            
            if (fail >= 2 && !ignore_failures) {
                setfail.push_back(p);
                continue;
            }
            
            if (!ignore_failures && succ > 200 && dists.size()-4*fail <= 12) {
                skipped.push_back(p);
                continue;
            }
                    
            int failstate = 0;
            res = multi_step_search(points, locs(p), out(p), refs, dists, nullptr, step, {}, {}, failstate, ws, th, used);
            all_locs.push_back(locs(p));
            
            // printf("%f\n", res);

            dbg(p) = -res;
                    
            if (failstate && !ignore_failures) {
                printf("fail %f %d %d\n", res, p[1]*5, p[0]*5);
                setfail.push_back(p);
                total_fail++;
                
                // std::vector<cv::Vec3f> succ_ps;
                // for(int j=0;j<10;j++)
                //     for(int i=0;i<10;i++) {
                //         cv::Vec2i l = p+cv::Vec2i(j-5,i-5);
                //         if (state(l) == 1)
                //             succ_ps.push_back(out(l));
                //     }
                    
                /*std::vector<cv::Vec3f> input_ps;
                std::vector<cv::Vec3f> rounded_ps;
                cv::Vec2i ref_loc = loc_sum*(1.0/dists.size());
                // cv::Vec2i ref_loc = locs(some_point);
                for(int j=0;j<50;j++)
                    for(int i=0;i<50;i++) {
                        cv::Vec2i l = ref_loc+cv::Vec2i(j-25,i-25);
                        input_ps.push_back(points(l[1],l[0]));
                    }
                // for(auto l : dbg_outp)
                    // input_ps.push_back(points(locs(l)[0],locs(l)[1]));
                // for(auto l : all_locs) 
                //     input_ps.push_back(at_int(points,l));
                // for(auto l : all_locs) 
                //     rounded_ps.push_back(points(l[1],l[0]));
                
                for(int n=0;n<refs.size();n++) {
                    std::cout << refs[n] << dbg_outp[0] << " " << dbg_outp[1] << std::endl;
                }
                
                write_ply("surf.ply", refs);
                write_ply("points.ply", input_ps);
                write_ply("res.ply", {out(p)});*/
                // write_ply("points_nearest.ply", rounded_ps);
                
                // cv::imwrite("xcurv.tif",x_curv);
                // cv::imwrite("ycurv.tif",y_curv);
                // return out;
            }
            else if (res < 0 && !failstate) {
                //image edge encountered
                state(p) = 11;
                out(p) = -1;
            }
            else {
                last_round_updated++;
                succ++;
                state(p) = 1;
                fringe.push_back(p);
                if (!used_area.contains({p[1],p[0]})) {
                    used_area = used_area | cv::Rect(p[1],p[0],1,1);
                }
                cv::Rect roi = {locs(p)[0]-80,locs(p)[1]-80,160,160};
                roi = roi & src_bounds;
                for(int j=roi.y;j<roi.br().y;j++)
                    for(int i=roi.x;i<roi.br().x;i++) {
                        // used(j,i) = std::max(used(j,i), float(1.0-1.0/T*sqrt(sdist(points(locs(p)[1],locs(p)[0]), points(j,i))+1e-2)));
                        used(j,i) = std::min(1.0f, used(j,i) + std::max(0.0f, float(1.0-1.0/T*sqrt(sdist(points(locs(p)[1],locs(p)[0]), points(j,i))+1e-2))));
                    }
            }
            
        }
        cands.resize(0);
  
        
        for(auto p : setfail) {
            dbg(p) = -1;
            state(p) = 10;
            out(p) = -1;
            collected_failures.push_back(p);
        }
        setfail.resize(0);
        
        // if (ignore_failures)
        //     stop_gen = generation+10;
        
        printf("-> total succ/fail %d/%d fringe: %d skipped: %d failures: %d\n", succ, total_fail, fringe.size(), skipped.size(), collected_failures.size());
    }
    
    // cv::resize(dbg, dbg, {0,0}, 10.0, 10.0, cv::INTER_NEAREST);
    // cv::imwrite("dbg.tif", dbg);
    
    // cv::imwrite("xcurv.tif",x_curv);
    // cv::imwrite("ycurv.tif",y_curv);
    
    //now lets expand a whole row
    /*for(int i=2;i<w;i++) {
//         float res;
//         //predict upper loc
//         // locs(0,i) = 2*locs(0,i-1)-locs(0,i-2);
//         locs(0,i) = locs(0,i-1);
//         
//         //FIXME this may take a curve?
//         res = multi_step_search(points, locs(0,i), out(0,i), {out(0,i-1),out(1,i-1)}, {T,D*T}, nullptr, step, {out(0,i-2)}, {2*T});
//         
//         if (res == -1)
//             break;
//         
// //         res = min_loc_dbg(points, locs(0,i), out(0,i), {out(0,i-2),out(0,i-1),out(1,i-1)}, {2*T,T,D*T}, nullptr, step, 0.01);
// //         
// //         std::cout << res << std::endl;
// // 
// //         //predict lower loc
//         // locs(1,i) = 2*locs(1,i-1)-locs(1,i-2);
//         locs(1,i) = locs(1,i-1);
//         //         res = min_loc_dbg(points, locs(1,i), out(1,i), {out(0,i),out(0,i-1),out(1,i-1)}, {T,D*T,T}, nullptr, step, 0.01);
//         res = multi_step_search(points, locs(1,i), out(1,i), {out(0,i),out(0,i-1),out(1,i-1)}, {T,D*T,T}, nullptr, step, {out(1,i-2)}, {2*T});
//         //         std::cout << res << std::endl;
//         if (res == -1)
//             break;
        {
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            
            int j = 0;
            
            for(int oy=std::max(j-r,0);oy<=2;oy++)
                for(int ox=std::max(i-r,0);ox<=std::min(i+r,out.cols-1);ox++)
                    if (out(oy,ox)[0] != -1 && (ox != i || oy != j)) {
                        refs.push_back(out(oy,ox));
                        int dy = oy-j;
                        int dx = ox-i;
                        dists.push_back(T*sqrt(dy*dy+dx*dx));
                    }
                    
            locs(j,i) = locs(j,i-1);
            // if (dists.size() < 4) {
            //     out(j,i) = 0;
            // }
            // else {
                res = multi_step_search(points, locs(j,i), out(j,i), refs, dists, nullptr, step, {}, {});
                
            //     if (res == -1)
            //         out(j,i) = -1;
            // }
        }
        
        {
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            
            int j = 1;
            
            for(int oy=std::max(j-r,0);oy<=2;oy++)
                for(int ox=std::max(i-r,0);ox<=std::min(i+r,out.cols-1);ox++)
                    if (out(oy,ox)[0] != -1 && (ox != i || oy != j)) {
                        refs.push_back(out(oy,ox));
                        int dy = oy-j;
                        int dx = ox-i;
                        dists.push_back(T*sqrt(dy*dy+dx*dx));
                    }
                    
            locs(j,i) = locs(j,i-1);
            // if (dists.size() < 4) {
            //     out(j,i) = 0;
            // }
            // else {
                res = multi_step_search(points, locs(j,i), out(j,i), refs, dists, nullptr, step, {}, {});
                
            //     if (res == -1)
            //         out(j,i) = -1;
            // }
        }
        
    }
    
    //now lets expand the rest
    for(int j=2;j<h;j++) {
        float res;
        
        for(int i=0;i<w;i++) {
            locs(j,i) = locs(j-1,i);
            
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            
            
            for(int oy=std::max(j-r,0);oy<=j;oy++)
                for(int ox=std::max(i-r,0);ox<=std::min(i+r,out.cols-1);ox++)
                    if (out(oy,ox)[0] != -1 && (ox != i || oy != j)) {
                        refs.push_back(out(oy,ox));
                        int dy = oy-j;
                        int dx = ox-i;
                        dists.push_back(T*sqrt(dy*dy+dx*dx));
                    }
                    
            // if (dists.size() < 4) {
            //     out(j,i) = -1;
            //     continue;
            // }
                    
            res = multi_step_search(points, locs(j,i), out(j,i), refs, dists, nullptr, step, {}, {});
            
            // res = multi_step_search(points, locs(j,i), out(j,i), {out(j-1,i),out(j-1,i-1),out(j,i-1)}, {T,D*T,T}, &plane, step, {out(j-2,i),out(j,i-2)}, {2*T,2*T});
            
            printf("%f\n", res);
            
            // if (res == -1) {
            //     out(j,i) = -1;
            //     // locs(j,i) = locs(j-1,i);
            //     // return out;
            //     continue;
            // }
            
        }
    }*/
    
    return out(used_area).clone();
}

cv::Mat_<cv::Vec3f> upsample_with_grounding_simple(const cv::Mat_<cv::Vec3f> &small, cv::Mat_<cv::Vec2f> &locs, const cv::Size &tgt_size, const cv::Mat_<cv::Vec3f> &points, double sx, double sy)
{
    std::cout << "upsample with simple interpolation " << small.size() << " -> " << tgt_size << std::endl; 
    cv::Mat_<cv::Vec3f> large;
    // cv::Mat_<cv::Vec2f> locs(small.size());
    // large = small;
    // cv::resize(small, large, small.size()*2, cv::INTER_CUBIC);
    cv::resize(small, large, tgt_size, cv::INTER_CUBIC);
    
    cv::Vec2f step = {sx*10, sy*10};
    cv::resize(locs, locs, large.size(), cv::INTER_CUBIC);

#pragma omp parallel for
    for(int j=0;j<large.rows;j++) {
        for(int i=0;i<large.cols;i++) {
            cv::Vec3f tgt = large(j,i);
            if (tgt[0] == -1)
                continue;
            float res;
            
            res = min_loc_dbg(points, locs(j,i), large(j,i), {tgt}, {0}, nullptr, step, 0.001, {}, true);
        }
    }
    
    return large;
}

float dist2(int x, int y)
{
    return sqrt(x*x+y*y);
}

float dist2(const cv::Vec2f &v)
{
    return sqrt(v[0]*v[0]+v[1]*v[1]);
}

cv::Mat_<cv::Vec3f> upsample_with_grounding_skip(const cv::Mat_<cv::Vec3f> &small, cv::Mat_<cv::Vec2f> &locs, int scale, const cv::Mat_<cv::Vec3f> &points, double sx, double sy)
{
    std::cout << "upsample with interpolation & search " << small.size() << " x " << scale << std::endl; 
    // cv::Mat_<cv::Vec2f> locs(small.size());
    
    cv::Vec2f step_large = {sx*128, sy*128};
    cv::Vec2f step = {sx*10, sy*10};
    
    int rdone = 0;
    
    cv::Size tgt_size = small.size() * scale;
    cv::Mat_<cv::Vec3f> large(tgt_size);
    
    cv::resize(locs, locs, large.size(), cv::INTER_CUBIC);
    
    large.setTo(-1);
    
#pragma omp parallel for
    for(int j=0;j<small.rows-1;j++)
        for(int i=0;i<small.cols-1;i++) {
            cv::Vec3f tgt1 = small(j,i);
            cv::Vec3f tgt2 = small(j,i+1);
            cv::Vec3f tgt3 = small(j+1,i);
            cv::Vec3f tgt4 = small(j+1,i+1);
            // dx /= scale;
            // dy /= scale;
            //TODO same for the others
            if (tgt1[0] == -1)
                continue;
            if (tgt2[0] == -1)
                continue;
            if (tgt3[0] == -1)
                continue;
            if (tgt4[0] == -1)
                continue;
            
            float dx1 = sqrt(sdist(small(j,i+1),small(j,i)))/scale;
            float dx2 = sqrt(sdist(small(j+1,i+1),small(j+1,i)))/scale;
            float dy1 = sqrt(sdist(small(j,i),small(j+1,i)))/scale;
            float dy2 = sqrt(sdist(small(j,i+1),small(j+1,i+1)))/scale;

            //TODO had a neat idea of using perspective transform to get good target distances ... not quite sure which coords to use though so until then use linear interpolation for the target distance
            //we are only interested in distances, so regard the quad as the length of four sides!
            // std::vector<cv::Vec2f> dist_coords = {{0,0},{0,scale},{scale,0},{scale,scale}};
            // std::vector<cv::Vec3f> dist_loc = {tgt1,tgt2,tgt3,tgt4};
            // cv::Mat m = cv::findHomography(dist_coords,dist_loc);
            
            //TODO this will overwrite points on the border between two blocks ... maybe average those?
            //TODO ideally we would do a local fit ... and that do that anyways also already on the initial search whe creating the mesh ...
            for(int ly=0;ly<scale;ly++)
                for(int lx=0;lx<scale;lx++) {
                    // large(j*scale+ly,i*scale+lx) = at_int(points, locs(j*scale+ly,i*scale+lx));
                    // continue;
                    if (!lx && !ly) {
                        large(j*scale,i*scale) = small(j,i);
                        continue;
                    }
                    int nx = scale-lx;
                    int ny = scale-ly;
                    
                    // std::vector<cv::Vec2f> dist_loc;
                    // dist_coords = {{lx,ly}, {nx,ly},{lx,ny},{nx,ny}};
                    
                    // cv::perspectiveTransform(dist_coords, dist_loc, m);
                    
                    // std::vector<float> dists = {dist2(dist_loc[0]),dist2(dist_loc[1]),dist2(dist_loc[3]),dist2(dist_loc[3])};
                    // std::vector<float> dists = {1,0,0,0};
                    float fy = float(ly)/scale;
                    float fx = float(lx)/scale;
                
                    float dx = (1-fy)*dx1+fy*dx2;
                    float dy = (1-fx)*dy1+fx*dy2;
                    std::vector<float> dists = {dist2(lx*dx,ly*dy),dist2(nx*dx,ly*dy),dist2(lx*dx,ny*dy),dist2(nx*dx,ny*dy)};
                    std::vector<float> ws = {1/dists[0], 1/dists[1],1/dists[2],1/dists[3]};
                    float res = min_loc_dbg(points, locs(j*scale+ly,i*scale+lx), large(j*scale+ly,i*scale+lx), {tgt1,tgt2,tgt3,tgt4}, dists, nullptr, step, 0.001, ws, true);
                }
            
        }
        
    return large;
}

cv::Mat_<cv::Vec3f> upsample_with_grounding(cv::Mat_<cv::Vec3f> &small, cv::Mat_<cv::Vec2f> &locs, const cv::Size &tgt_size, const cv::Mat_<cv::Vec3f> &points, double sx, double sy)
{
    int scale = std::max(tgt_size.width/small.cols, tgt_size.height/small.rows);
    cv::Size int_tgt = tgt_size*scale;
    cv::Mat_<cv::Vec3f> large = small;

    while (scale >= 2) {
        scale = 2;
        int_tgt = tgt_size*scale;

        if (small.size() != int_tgt)
            large = upsample_with_grounding_skip(small, locs, scale, points, sx, sy);

        small = large;
        scale = std::max(tgt_size.width/small.cols, tgt_size.height/small.rows);
    }

    if (int_tgt != tgt_size)
        large = upsample_with_grounding_simple(large, locs, tgt_size, points, sx, sy);

    return large;
}


void refine_normal(const std::vector<std::pair<cv::Vec2i,cv::Vec3f>> &refs, cv::Vec3f &point, cv::Vec3f &normal, cv::Vec3f &vx, cv::Vec3f &vy, const std::vector<float> &ws) {};
