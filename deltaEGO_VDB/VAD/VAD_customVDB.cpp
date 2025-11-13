#include "VAD_customVDB.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <array>
#include <functional>
#include <numeric>
#include <string_view>


/*
This function reads VAD data in VAD folder and convert it into Emotion and node.
If it succeed to load, it will return True flag.
*/
bool KDTree::load_data(const std::string& json_path)
{
    std::cout << "-----------Loading VAD emotion data-----------" << std::endl;
        
    // open file
    std::ifstream ifs(json_path);
    if(!ifs.is_open())
        return false;
    
    // get json array
    json j;
    try 
    {
        j = nlohmann::json::parse(ifs);
    } 
    // parsing fail
    catch (const nlohmann::json::parse_error& e) 
    {
        std::cerr << "parse error: " << e.what() << "\n";
        return false;
    }

    // is it array?
    if (!j.is_array()) 
        return false;
        
    // size of array
    int size = j.size();

    // allocate heap     
    this->Emotions.clear();
    this->Emotions.reserve(size);

    // save it!
    for (const auto& element : j) 
    {
        Emotion emo;
        emo.term = element.at("term").get<std::string>();

        emo.point.x = element.at("valence").get<double>();
        emo.point.y = element.at("arousal").get<double>();
        emo.point.z = element.at("dominance").get<double>();

        Emotions.emplace_back(std::move(emo));
    }

    // bulid KD-Tree with index vector
    // I used P_buffer because I've heard this is a kind of permutation buffer 
    std::vector<int> P_buffer(this->Emotions.size());   // a vector that saves the emotions index
    std::iota(P_buffer.begin(), P_buffer.end(), 0);     // this will save index value to vector
    this->root = this->build_tree_with_iterative(P_buffer);

    this->axis_scale = this->compute_axis_std();

    std::cout << "--------Loading VAD emotion data Success!--------\n";
    return true;   
}
/*
This will bulid k-d Tree data structure in non-recursive way (heap based)
It was VERY HARD for me
It will use vector array as stack
*/
int KDTree::build_tree_with_iterative(std::vector<int>& P_buffer)
{
    // initialize tree (delete previous nodes)
    this->nodes.clear();
    // number of nodes == number of data -> reserve capacity(prevent reallocation)
    this->nodes.reserve(P_buffer.size());

    // a frame that can alter recursion
    struct Frame
    {
        int 
        l,              // a start point of range to process(inclusive), range of P_buffer is [l, r)
        r,              // a end point of range to process  (exclusive)
        depth,          // current subtree's depth -> used for axis calculation (axis = depth % 3)
        parent;         // parent node's index. if it is root, index will be -1
        bool is_left;   // is this parent node's left node or not? --- it is used for connecting with parent
    };

    // stack for iteration(heap vector) --> it will process with push/pop the frame
    std::vector<Frame> work;

    work.reserve(64);   // the size 64 is for preventing multiple realloc

    // it is pushing a new Frame => l = 0, r = (int)P_buffer.size(), depth = 0, parent = -1, is_left = false       
    work.push_back({0, (int)P_buffer.size(), 0, -1, false}); 

    // root node's index
    int local_root = -1;

    // build start
    while(!work.empty())
    {
        // Pull out very last frame and process (LIFO)
        Frame f = work.back();
        work.pop_back();

        //if it is empty range, skip
        if (f.l >= f.r) 
            continue;
            
        // gets axis of data based on depth
        int axis = axis_of(f.depth);

        // gets median of l and r (Split point location). The actual split point value will be decided at nth_element
        int median = (f.l + f.r) / 2;

        // declare lambda function (returns axis' value based on given data index. ex. if axis = x -> returns x value)
        auto axisVal_lambda = [&](int data_idx)
        {
            const auto& p = Emotions[data_idx].point;
            return (axis == 0) ? p.x : (axis == 1) ? p.y : p.z;
        }; 

        // reallocate P_buffer's [l, r) range -> P_buffer[median] is median data index 
        std::nth_element(P_buffer.begin() + f.l,
                         P_buffer.begin() + median,
                         P_buffer.begin() + f.r,
                         // another lambda
                         [&](int a, int b){ return axisVal_lambda(a) < axisVal_lambda(b);}
                        );

        // declares new node: split point is a data pointed by P_buffer[median]    
        int mid_idx = (int)nodes.size();
        this->nodes.push_back(Node{ P_buffer[median], (uint8_t)axis, -1, -1 });

        // connect parent-child node (if it is root, parent == -1 -> save it to local_root)
        if (f.parent >= 0) 
        {
            if (f.is_left) 
                nodes[f.parent].left = mid_idx;
            else           
                nodes[f.parent].right = mid_idx;
        } 
        else 
        {
            local_root = mid_idx;
        }

        // push right subtree[median + 1, f.r) for processing it later
        if (median + 1 < f.r) 
            work.push_back({ median + 1, f.r, f.depth + 1, mid_idx, false }); // right

        // push left subtree[f.l, median) for processing it later
        if (f.l < median)     
            work.push_back({ f.l,     median, f.depth + 1, mid_idx, true  }); // left

    }
    return local_root;
}

AxisScale KDTree::compute_axis_std() const
{
    const unsigned int size_of_data = this->Emotions.size();

    double mx = 0, my = 0, mz = 0;
    for(auto& emotion: this->Emotions)
    {
        mx += emotion.point.x;
        my += emotion.point.y;
        mz += emotion.point.z;
    }
    // get everage x,y,z value of emotion data
    mx /= size_of_data; 
    my /= size_of_data;
    mz /= size_of_data;

    double vx = 0, vy = 0, vz = 0;
    for (auto& emotion : this->Emotions)
    {
        double dx = emotion.point.x - mx;
        double dy = emotion.point.y - my;
        double dz = emotion.point.z - mz;

        vx += dx * dx; vy += dy * dy; vz += dz * dz;
    }

    vx /= static_cast<double>(std::max<unsigned int>(1, size_of_data - 1));
    vy /= static_cast<double>(std::max<unsigned int>(1, size_of_data - 1));
    vz /= static_cast<double>(std::max<unsigned int>(1, size_of_data - 1));

    auto temp_lambda = [](double s){return (s < 1e-6) ? 1e-6 : s;};

    return AxisScale{
        temp_lambda(std::sqrt(vx)),
        temp_lambda(std::sqrt(vy)),
        temp_lambda(std::sqrt(vz))
    };
}

//--------------------------------------------------------------------------------------------------

inline double KDTree::get_axis(const Point3D& point, int axis)
{
    return (axis == 0) ? point.x : (axis == 1) ? point.y : point.z;
}
inline double KDTree::distance_pow2(const Point3D& a, const Point3D& b)
{
    double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}
inline std::vector<std::string> KDTree::parse_option(std::string_view opt) 
{
    this->trim_opt(opt);

    // seperate main/flag
    size_t sp = opt.find(' ');
    std::string_view main  = (sp == std::string_view::npos) ? opt : opt.substr(0, sp);
    std::string_view fpart = (sp == std::string_view::npos) ? std::string_view{} : opt.substr(sp+1);
    this->trim_opt(main);
    this->trim_opt(fpart);

    // seperate main with visit~sim
    size_t til = main.find('~');
    std::string visit = (til == std::string_view::npos) ? std::string(main) : std::string(main.substr(0, til));
    std::string sim   = (til == std::string_view::npos) ? "none" : std::string(main.substr(til+1));

    if (visit.empty()) 
        visit = "knn";

    // handle one flag
    std::string flag;
    if (!fpart.empty()) 
    {
        size_t j = 0;
        while (j < fpart.size() && !std::isspace((unsigned char)fpart[j])) 
            ++j;

        std::string_view tok = fpart.substr(0, j);
        if (tok.size() == 2 && tok[0] == '-') 
        {
            flag.assign(1, tok[1]); // one letter like"E" 
        }
    }

    return { std::move(visit), std::move(sim), std::move(flag) };

}
inline void KDTree::trim_opt(std::string_view& str) 
{
    size_t a = 0, b = str.size();

    while (a < b && std::isspace((unsigned char)str[a])) 
        ++a;

    while (b > a && std::isspace((unsigned char)str[b-1])) 
        --b;

    str = str.substr(a, b-a);
}
// compute similarity ------------------------------------------------------------
inline int KDTree::similarity_percent_relative(const Point3D& q, const Point3D& p, double d)
{
    // get percentage how close it is base on d 
    if (d <= 0.0) 
    return 0;
    
    double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
    double d2 = dx*dx + dy*dy + dz*dz;
    
    if (d2 >= d*d) 
    return 0;
    
    double sim = 1.0 - (d2 / (d*d));
    
    return static_cast<int>(std::lround(sim * 100.0));
}
inline int KDTree::similarity_percent_abs_L2(const Point3D& q, const Point3D& p)
{
    // L2 normalization
    double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
    double d = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    constexpr double DMAX = 2.0 * 1.7320508075688772; // 2*sqrt(3)
    double sim = 1.0 - d/DMAX;
    
    if (sim < 0) 
    sim = 0; 
    if (sim > 1) 
    sim = 1;
    
    return static_cast<int>(std::lround(sim * 100.0));
}
inline int KDTree::similarity_percent_cosine(const Point3D& q, const Point3D& p)
{
    // cosine simularity
    double dot = q.x*p.x + q.y*p.y + q.z*p.z;
    
    double nq  = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
    double np  = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    
    if (nq==0.0 || np==0.0) 
    return 0;
    
    double cosv = dot/(nq * np);           
    double sim = 0.5*(cosv + 1.0);
    
    return static_cast<int>(std::lround(sim * 100.0));
}
inline int KDTree::similarity_percent_gauss_l2(const Point3D& q, const Point3D& p, double SIGMA)
{
    if(SIGMA <= 0.0)
        return 0;

    const double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
    const double d_2 = dx*dx + dy*dy + dz*dz;
    const double sim = std::exp(-d_2 / (2.0 * SIGMA * SIGMA));
    
    return static_cast<int>(std::lround(std::clamp(sim, 0.0, 1.0) * 100.0));
}
inline int KDTree::similarity_percent_gauss_whitened(const Point3D& q, const Point3D& p, double SIGMA)
{
    if(this->axis_scale.sx <= 0 
    || this->axis_scale.sy <= 0
    || this->axis_scale.sz <= 0
    || SIGMA <= 0)
        return 0;

    const double dx=(q.x-p.x)/this->axis_scale.sx;
    const double dy=(q.y-p.y)/this->axis_scale.sy;
    const double dz=(q.z-p.z)/this->axis_scale.sz;
    const double d2 = dx*dx + dy*dy + dz*dz;

    const double sim = std::exp(- d2 / (2.0 * SIGMA * SIGMA));

    return static_cast<int>(std::lround(std::clamp(sim, 0.0, 1.0) * 100.0));
}
// compute similarity ------------------------------------------------------------
inline int KDTree::compute_similarity_pct(const std::string& sim_key,
                                  const Point3D& q, const Point3D& p, double d, double SIGMA) 
{
    if (sim_key == "d")       
        return similarity_percent_relative(q,p,d); 
    
    else if (sim_key == "l2")      
        return similarity_percent_abs_L2(q,p);    

    else if (sim_key == "cos")     
        return similarity_percent_cosine(q,p);

    else if (sim_key == "gauss")
        return similarity_percent_gauss_l2(q,p,SIGMA);             
    
    else if (sim_key == "none")    
        return similarity_percent_abs_L2(q,p);

    else if (sim_key == "gauss_w")
        return similarity_percent_gauss_whitened(q, p, SIGMA);

    else
        return similarity_percent_abs_L2(q,p);
}
inline std::string KDTree::get_compute_similarity_algorithm(const std::string& key)
{
    if (key == "d")
        return "Relative similarity based on d";

    else if(key == "l2")
        return "L2 normalization";

    else if(key == "cos")
        return "Cosine similarity";

    else if(key == "gauss")
        return "RBF with plain L2";

    else if(key == "gauss_w")
        return "Whitened / Axis-scaled Gaussian";
        
    else if(key == "none")
        return "L2 normalization";

    else
        return "L2 normalization"; 
}

std::string KDTree::VAD_search_near_k(double V,       /* Valance */
                                      double A,       /* Arousal */
                                      double D,       /* Dominance */
                                      int k           /* how many? */,
                                      double d        /* how near */,
                                      double SIGMA    /* For gaussian*/, 
                                      std::string opt /* search option */)
{
    // if root == -1, tree is empty
    if(this->root < 0)
        return R"({"error":"empty_tree"})";

    if(V == A && A == D && V == 0.0)
    {
        return R"({"emotion":"neutral","magnitude":0,"similarity":1})";
    }

    if (k <= 0) 
        return R"({"error":"k is 0 or minus"})";
    if (k > Emotions.size())
        // prevent error 
        k = Emotions.size();
    

    // prepare for search-----------------------
    const Point3D input_p {V,A,D};
    MaxHeap heap;

    // parse option: input = knn_d~l2 -> knn_d and l2
    struct ParsedOpt 
    {
        std::string visit_key; // knn, knn_d or etc.
        std::string sim_key;   // l2, cos, ccos, gauss or etc.
        std::string flag; // -E, -D, ""
    };
    std::vector<std::string> parsed_opt = parse_option(static_cast<std::string_view>(opt));
    ParsedOpt user_opt{parsed_opt[0], parsed_opt[1], parsed_opt[2]};

    // get a lambda function
    auto lambda_compare = get_search_func(user_opt.visit_key);

    const bool does_use_d = (user_opt.visit_key == "knn_d");
    const double r2 = d * d;

    // make a stack for iteration loop
    std::vector<int> stk;
    stk.reserve(64);
    stk.push_back(this->root);

    //search
    try
    {
        while(!stk.empty())
        {
            int i = stk.back();
            stk.pop_back();
    
            // root has inserted
            if(i < 0)
                continue;
    
            // compare and update
            lambda_compare(input_p, i, k, d, heap);
            
            const Node& current_node = this->nodes[i];
            const Point3D& current_point3d = this->Emotions[current_node.idx].point;
    
            // near? far?
            double delta = this->get_axis(input_p, current_node.axis)
                                    - this->get_axis(this->Emotions[this->nodes[i].idx].point, current_node.axis);
        
            int near_child = (delta <= 0) ? current_node.left : current_node.right;
            int  far_child = (delta <= 0) ? current_node.right : current_node.left;
    
            double threshold = (heap.size() == k) ? heap.top().first : std::numeric_limits<double>::infinity();
            if (does_use_d) threshold = std::min(threshold, r2);
    
            // add stack
            if (far_child >= 0 && (delta * delta) <= threshold) // if it is too far, don't add it to stack
                stk.push_back(far_child);
            if (near_child >= 0)                              
                stk.push_back(near_child);
        }
    }
    catch (...)
    {
        return R"({"error":"search fail"})";
    }
    
    // sort
    std::vector<Hit> tmp; 
    tmp.reserve(heap.size());

    while (!heap.empty()) 
    { 
        tmp.push_back(heap.top()); 
        heap.pop(); 
    }

    std::sort(tmp.begin(), tmp.end(), [](const Hit& a, const Hit& b){ return a.first < b.first; });
    // prevent too big k
    const int limit = std::min<int>(static_cast<int>(tmp.size()), k);

    if (tmp.size() > static_cast<std::size_t>(k)) 
        tmp.resize(static_cast<std::size_t>(k));

    // return value
    json out;
    out["query"] = {{"V",V},{"A",A},{"D",D}};
    out["mode"]  = {{"input_visit",user_opt.visit_key},{"input_sim",user_opt.sim_key},{"flag", user_opt.flag},{"k",k},{"d",d}};

    json arr = json::array();

    if (user_opt.flag == "B")
    {
        for (int i = 0; i < limit; i++)
        {
            int idx = tmp[i].second;
            double d2 = tmp[i].first;
            const Point3D& p = this->Emotions[idx].point;
    
            json item = 
            {
                {"rank", i + 1},
                {"emotion", Emotions[idx].term},
                {"distance_pow2", d2},
                {"VAD", {{"V",p.x},{"A",p.y},{"D",p.z}}}
            };

            int similarity = this->compute_similarity_pct(user_opt.sim_key, input_p, p, d);

            item["similarity_percent"] = similarity;
            item["similarity_metric"]  = this->get_compute_similarity_algorithm(user_opt.sim_key);

            if(user_opt.sim_key == "gauss" || user_opt.sim_key == "gauss_w")
                item["emotion_simplified"] = this->get_str_expression(similarity) + " " + this->Emotions[idx].term;

            arr.push_back(std::move(item));
        }
    }
    else if (user_opt.flag == "D") //TODO
    {
        for (int i = 0; i < limit; i++)
        {
            int idx = tmp[i].second;
            double d2 = tmp[i].first;
            const Point3D& p = this->Emotions[idx].point;
    
            json item = 
            {
                {"rank", i + 1},
                {"emotion", Emotions[idx].term},
                {"distance_pow2", d2},
                {"VAD", {{"V",p.x},{"A",p.y},{"D",p.z}}}
            };

            int similarity = this->compute_similarity_pct(user_opt.sim_key, input_p, p, d);


            item["similarity_percent"] = similarity;
            item["similarity_metric"]  = this->get_compute_similarity_algorithm(user_opt.sim_key);

            if(user_opt.sim_key == "gauss" || user_opt.sim_key == "gauss_w")
                item["emotion_simplified"] = this->get_str_expression(similarity) + " " + this->Emotions[idx].term;

            arr.push_back(std::move(item));
        }
    }
    else if (user_opt.flag == "S")
    {
        for (int i = 0; i < limit; i++)
        {
            int idx = tmp[i].second;
            double d2 = tmp[i].first;
            const Point3D& p = this->Emotions[idx].point;
    
            json item = 
            {
                {"rank", i + 1},
                {"emotion", Emotions[idx].term},
                {"distance_pow2", d2},
                {"VAD", {{"V",p.x},{"A",p.y},{"D",p.z}}}
            };

            int similarity = this->compute_similarity_pct(user_opt.sim_key, input_p, p, d);

            item["emotion_simplified"] = this->get_str_expression(similarity) + " " + this->Emotions[idx].term;
            item["similarity_metric"]  = this->get_compute_similarity_algorithm(user_opt.sim_key);

            arr.push_back(std::move(item));
        }
    }
    else    // default E
    {
        for (int i = 0; i < limit; i++)
        {
                        int idx = tmp[i].second;
            double d2 = tmp[i].first;
            const Point3D& p = this->Emotions[idx].point;
    
            json item = 
            {
                {"rank", i + 1},
                {"emotion", Emotions[idx].term},
                {"distance_pow2", d2},
                {"VAD", {{"V",p.x},{"A",p.y},{"D",p.z}}}
            };

            int similarity = this->compute_similarity_pct(user_opt.sim_key, input_p, p, d);

            item["similarity_percent"] = similarity;
            item["similarity_metric"]  = this->get_compute_similarity_algorithm(user_opt.sim_key);
            
            if(user_opt.sim_key == "gauss" || user_opt.sim_key == "gauss_w")
                item["emotion_simplified"] = this->get_str_expression(similarity) + " " + this->Emotions[idx].term;

            arr.push_back(std::move(item));
        }
    }
    out["result"] = std::move(arr);
    out["count"]  = (int)out["result"].size();

    return out.dump();
}

/* Is it ideal? --> idk let's ask */
std::function<void(
                    const Point3D&, /* input */
                    int             /* node index */, 
                    int             /* k: how many? */,
                    double          /* d: how near */,
                    MaxHeap&        /* searched */
    )> KDTree::get_search_func(std::string& option)
{
    if (option == "knn") 
    {
        // k-NN without d
        return [this](const Point3D& q, int nodeIdx, int k, double /* d */, MaxHeap& heap) 
        {
            const Node& temp_node = this->nodes[nodeIdx];
            const auto& p  = this->Emotions[temp_node.idx].point;

            const double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            const double d2 = dx * dx + dy * dy + dz * dz;

            // k is int , but heap.size() is unsigned
            if (heap.size() < static_cast<std::size_t>(k)) 
                heap.emplace(d2, temp_node.idx);

            else if (d2 < heap.top().first) 
            { 
                heap.pop(); 
                heap.emplace(d2, temp_node.idx); 
            }
        }; 
    }

    else if(option == "knn_d")
    {
        return [this](const Point3D& q, int nodeIdx, int k, double radius, MaxHeap& heap)
        {
            const Node& temp_node = this->nodes[nodeIdx];
            const auto& p  = this->Emotions[temp_node.idx].point;

            const double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            const double d2 = dx * dx + dy * dy + dz * dz;

            const double r2 = radius * radius;
            // if it is not near enough, skip          
            if (d2 > r2) 
                return;

            if (heap.size() < static_cast<std::size_t>(k)) 
                heap.emplace(d2, temp_node.idx);
            
            else if(d2 < heap.top().first)
            {
                heap.pop(); 
                heap.emplace(d2, temp_node.idx);
            }
        };
    }

    else
    {
        // k-NN without d
        return [this](const Point3D& q, int nodeIdx, int k, double /* d */, MaxHeap& heap) 
        {
            const Node& temp_node = this->nodes[nodeIdx];
            const auto& p  = this->Emotions[temp_node.idx].point;

            const double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            const double d2 = dx * dx + dy * dy + dz * dz;

            // k is int , but heap.size() is unsigned
            if (heap.size() < static_cast<std::size_t>(k)) 
                heap.emplace(d2, temp_node.idx);

            else if (d2 < heap.top().first) 
            { 
                heap.pop(); 
                heap.emplace(d2, temp_node.idx); 
            }
        }; 
    }
}

inline std::string KDTree::get_str_expression(const int& percentage)
{
    if (percentage >= 0 && percentage <= 5)
        return "negligible";

    else if (percentage > 5 && percentage <= 20)
        return "mild";

    else if (percentage > 20 && percentage <= 40)
        return "somewhat";

    else if (percentage > 40 && percentage <= 60)
        return "moderate";
    
    else if (percentage > 60 && percentage <= 80)
        return "quite";

    else if (percentage > 80 && percentage <= 95)
        return "intense";

    else
        return "absolute";
}