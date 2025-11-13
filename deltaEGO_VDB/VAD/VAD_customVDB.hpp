#ifndef VAD_CUSTOMVDB_HPP
#define VAD_CUSTOMVDB_HPP

#include <nlohmann/json.hpp>
#include <string>
#include <array>
#include <functional>
#include <queue>


using json = nlohmann::json;


struct Point3D 
{
    double x, y, z;     // Valence, Arousal, Dominance
};

struct Emotion 
{
    std::string term;   // name of emotion
    Point3D point;          // VAD value
};

struct Node 
{
    int idx;        // items[]'s index 
    uint8_t axis;   // 0:x, 1:y, 2:z
    int left;       // nodes[]'s index, if it doesn't exists -> -1
    int right;      // nodes[]'s index, if it doesn't exists -> -1
};

//-----------for search-----------
using Hit = std::pair<double,int>;

struct searched_data
{
    Emotion emo;
    double emo_magnitude = 0;  // 0 ~ 1, magnitude of vector
    double simularity = 0; // 0 ~ 1
};

struct WorseFirst 
{                 // Max heap: the farthest will be top()
    bool operator()(const Hit& a, const Hit& b) const 
    {
        return a.first < b.first;   // bigger dist will come first
    }
};

using MaxHeap = std::priority_queue<Hit, std::vector<Hit>, WorseFirst>;
//-----------for search-----------

//-----------for Whitened / axis scaled Gaussian ------------
struct AxisScale 
{ 
    double sx, sy, sz; 
};

class KDTree 
{
    public:
    std::vector<Emotion> Emotions;
    std::vector<Node> nodes;
    int root = -1;
    AxisScale axis_scale;
    // constructor
    KDTree() : root(-1) {}
    
    // methods goes here
    
    //--------------------Buliding Tree--------------------
    static inline int axis_of(int depth)
    { 
        return depth % 3; 
    }
    /*
    This function reads VAD data in VAD folder and convert it into Emotion and node.
    If it succeed to load, it will return True flag.
    */
    bool load_data(const std::string& json_path);
    /*
    This will bulid k-d Tree data structure in non-recursive way (heap based)
    I used std::vector instead of std::stack because vector is saved in heap

    Complexity:
        * Time: average O(N log N)
            -> for each level, nth_element is O(subrange), and total level is approx. log N.
        * Space: 
            * nodes = N
            * P_buffer = N index
            * work stack = O(log N) frame

    If it was recursive:

    Node* build(l, r, depth):
        if l >= r: return null
        axis = depth % 3
        median = (l + r) / 2
        nth_element(P_buffer[l:r), median, key=axis)
        node = new Node(P_buffer[median], axis)
        node->left  = build(l, median, depth+1)
        node->right = build(median+1, r, depth+1)
        return node

    How it builds:

    1. At first loop, push entire range as whole frame(range: [0,N) )

    2. Pop one from stack and find that range's axis (axis = depth % 3 --> x,y,z).

    3. Establish a baseline with location of median = (l + r) /2 and call std::nth_element(P_buffer.begin()+l ...)
       --> with nth_element, sort partialy to put median value at that location on that axis
        * It will ensure ... : From P_buffer[median], left wull be <= and right will be >=

    4. With median value point, generate 1 Node and push_back to nodes.
        * If it has parent, connect left/right child nodes.
        * If not, this node will be local_root.
    
    5. Push right range ( [median+1, r) ) and left range( [l, median) ) as new frame

    6. loop it until stack is empty

    ==> 1 frame = making 1 sub tree and find each sub tree's root with nth_element

    Main logic mapping:

        * axis_of(f.depth) : depth % 3 --> it's 3d (V,A,D)
        * median = (f.l + f.r) / 2 : The mid index location of current range
        * key_lambda(data_idx) : return current data's axis value
        * nth_element(...): Partialy sort P_buffer[f.l:f.r) to come median index in perspective of axis at P_buffer[median]
        * nodes.push_back(Node{ P_buffer[median], (uint8_t)axis, -1, -1 }); --> Make a new node with fixed median value
        * Connect parent: if f.parent is true, "nodes[f.parent].left/right = mid_idx;" based on is_left
        * Push child frame:
            * right: [median+1, f.r)
            * left : [f.l, median)
            
            => why push right first?
                * Because of LIFO, left need to push first if left needs to be process first
                * In current code, right will be pushed first and left after. -> left will be poped first
          

    */
   int build_tree_with_iterative(std::vector<int>& P_buffer);
   // after build compute axis scale
   AxisScale compute_axis_std() const;
    //----------------------------------------Buliding Tree----------------------------------------
    

    //----------------------------------------Searching data---------------------------------------

    inline double distance_pow2(const Point3D& a, const Point3D& b);
    inline std::vector<std::string> parse_option(std::string_view opt);
    inline double get_axis(const Point3D& point, int axis);
    inline void trim_opt(std::string_view& str);
    inline int similarity_percent_relative(const Point3D& q, const Point3D& p, double d);
    inline int similarity_percent_abs_L2(const Point3D& q, const Point3D& p);
    inline int similarity_percent_cosine(const Point3D& q, const Point3D& p);
    inline int similarity_percent_gauss_l2(const Point3D& q, const Point3D& p, double SIGMA);
    inline int similarity_percent_gauss_whitened(const Point3D& q, const Point3D& p, double SIGMA);
    inline int compute_similarity_pct(const std::string& sim_key, const Point3D& q, const Point3D& p, double d, double SIGMA = 0.5);
    inline std::string get_compute_similarity_algorithm(const std::string& key);

    std::string VAD_search_near_k(double V,       /* Valance */
                                  double A,       /* Arousal */
                                  double D,       /* Dominance */
                                  int k           /* how many? */,
                                  double d        /* how near */, 
                                  double SIGMA    /* For gaussian*/,
                                  std::string opt  = "knn" /* search option */);

    std::function<void(const Point3D&, int/* node index */, int /* k: how many? */, double /* how near */, MaxHeap&)> 
    get_search_func(std::string& option);

    inline std::string get_str_expression(const int& percentage);
};

#endif