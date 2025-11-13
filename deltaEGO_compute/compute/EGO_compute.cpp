#include "EGO_compute.hpp"
// TODO: do oposite of now

// struct ---------------------------------------------------------------------------
struct O1_Tasks_Result 
{
    VADPoint delta;
    double instant_stress;
    double instant_reward;
    double instant_ratio_total;
    double instant_stress_ratio;
    double instant_reward_ratio;
    double affective_lability;
    double deviation;
};

struct Ratio
{
    double stress_raw;
    double reward_raw;
    double ratio_total;
    double stress_ratio;
    double reward_ratio;
};
// struct ---------------------------------------------------------------------------


// inline function ------------------------------------------------------------------
/*
 * This function returns Euclidian distance between points in 3d area.
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
inline double get_distance(const VADPoint& a, const VADPoint& b)
{
    return std::sqrt((a.v - b.v)*(a.v - b.v) + (a.a - b.a)*(a.a - b.a) + (a.d - b.d)*(a.d - b.d));
}
inline double sigmoid(double x)
{
    if (x >= 0) 
    {
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    } 
    else 
    {
        double e = std::exp(x);
        return e / (1.0 + e);
    }
}
/**
 * This function analizes ratio.
 * Time complexity = O(1)
 * Space complexity = O(1)
 */
Ratio inline get_stress_reward_ratio(const double& stress, const double& dopamine)
{
    double total = stress + dopamine;
    double ratio_stress = 0.0;
    double ratio_reward = 0.0;

    // if it tries to devide with 0
    if (total > 1e-9)
    {
        ratio_stress = stress / total;
        ratio_reward = dopamine / total;
    }

    return Ratio{stress, dopamine, total, ratio_stress, ratio_reward};
}
// inline function ------------------------------------------------------------------


// O(1) ------------------------------------------------------------------------------
/*
 * This function returns delta value of current and prev emotion
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
VADPoint calculate_delta(const VADPoint& prev, const VADPoint& current)
{
    double dt = current.timestamp - prev.timestamp;

    if (dt <= 0) 
        dt = 1.0;

    return {
        (current.v - prev.v) / dt,
        (current.a - prev.a) / dt,
        (current.d - prev.d) / dt,
        current.timestamp
    };
}

/*
 * This function returns instant stress level
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
double calculate_instant_stress(const VADPoint& current, 
                                const VADPoint& baseline, 
                                double stabilityRadius, 
                                double weightA_stress, 
                                double weightV_stress,
                                double dampening_factor)
{
    double distance = get_distance(current, baseline);

    // if emotion is in emotion stability area, level of stress will decrease
    dampening_factor = (distance <= stabilityRadius) ? dampening_factor : 1.0;
    
    // stress from valance 
    double stressV = weightV_stress * ((1.0 - current.v) / 2.0); 
    // stress from arousal
    double stressA = weightA_stress * current.a;

    // total stress
    double normal_stress = std::min(1.0, std::max(0.0, stressV + stressA));
    
    return normal_stress * dampening_factor;
}

/*
 * This function returns affective lability(emotional whiplash)
 * Used sigmoid
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
double calculate_affective_lability(const VADPoint& delta, const double& weight_k, const double& theta_0)
{
    // get Slope in 3d
    double horizon_h = std::sqrt(delta.v*delta.v + delta.a*delta.a);
    double theta = std::atan2(delta.d, horizon_h);
    double z = weight_k * (theta - theta_0);

    return sigmoid(z);
}

/*
 * This function returns a 'Reward Index'. Emulates dopamine
 * Based on High Valence (Pleasure) and High Arousal (Energy)
 * Time complexity: O(1)
 */
double calculate_reward_index(const VADPoint& current, 
                              double weightV_reward, 
                              double weightA_reward)
{
    double rewardV = weightV_reward * ((current.v + 1.0) / 2.0);    
    double rewardA = weightA_reward * current.a;

    return std::min(1.0, std::max(0.0, rewardV + rewardA));
}

/*
 * This funtion calculate O(1) executions.
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
O1_Tasks_Result get_O1_functions_async(
    const std::optional<VADPoint>& prev,
    const VADPoint& current,
    const VADPoint& baseline,
    double stabilityRadius,
    double weightA_stress,
    double weightV_stress,
    double weightV_reward,
    double weightA_reward,
    double dampening_factor,
    double weight_k,
    double theta_0)
{
    // delta
    VADPoint delta_reval = (prev.has_value()) ? calculate_delta(prev.value(), current) : VADPoint{ 0.0, 0.0, 0.0, 0.0 };

    // Instant Stress
    double instant_stress_reval = calculate_instant_stress(
        current, baseline, stabilityRadius, weightA_stress, weightV_stress, dampening_factor);

    // Affective Lability
    double affective_lability_reval = calculate_affective_lability(
        delta_reval, weight_k, theta_0);
    
    // Instant reward
    double instant_reward_reval = calculate_reward_index(current, weightV_reward, weightA_reward);

    // ratio
    Ratio ratio = get_stress_reward_ratio(instant_stress_reval, instant_reward_reval);

    // deviation
    double deviation = get_distance(current, baseline);

    return O1_Tasks_Result{ 
        delta_reval, 
        instant_stress_reval, 
        instant_reward_reval,      
        ratio.ratio_total, 
        ratio.stress_ratio, 
        ratio.reward_ratio, 
        affective_lability_reval, 
        deviation 
    };    
}    
// O(1) ------------------------------------------------------------------------------



// O(n) ------------------------------------------------------------------------------
/*
 * This funtion returns average emotion area based on history.
 * History must be up-to-date for accuracy.
 * Time complexity: O(n) = 2T(n)
 * Space complexity: O(1)
 */ 
VAD_ave calculate_average(const std::vector<VADPoint>& history)
{
    int history_size = static_cast<int>(history.size());
    // if history is empty
    if(history_size == 0)
    {
        return VAD_ave{0.0, 0.0, 0.0, 0.05};
    }    
    
    // calculate average emotion's center
    double v = 0, a = 0, d = 0;
    for(const auto& element: history)
    {
        v += element.v;
        a += element.a;
        d += element.d;
    }    
    v /= history_size; a /= history_size; d /= history_size;
    
    // calculate radius of average emotion
    VADPoint average_center{v,a,d,0.0};
    double r = 0;
    for(const auto& element: history)
    {
        r += get_distance(average_center, element);
    }    
    r /= history_size;
    
    return VAD_ave{v,a,d,r};
}    

/*
 * This funtion returns cumulative stress with anti-diff
 * Time complexity: O(n) = T(n)
 * Space complexity: O(1)
 */ 
double calculate_cumulative_stress(const std::vector<VADPoint>& history, 
                                    const VADPoint& baseline, 
                                    const double& stabilityRadius,
                                    const double& weightA_stress, 
                                    const double& weightV_stress,
                                    const double& dampening_factor)                                    
{
    size_t history_size = history.size();

    // if sample is too small
    if (history_size < 2) 
        return 0.0;

    double cumulative_stress = 0.0;
    for(size_t i = 1; i < history_size; i++)
    {
        const auto& current = history[i];
        const auto& prev = history[i-1];

        double dt = current.timestamp - prev.timestamp;
        if (dt <= 0) 
            dt = 0.1;
        
        // get interval's stress and anti-diff
        double instant_stress = calculate_instant_stress(current, 
                                                        baseline, 
                                                        stabilityRadius, 
                                                        weightA_stress, 
                                                        weightV_stress,
                                                        dampening_factor);
        cumulative_stress += instant_stress * dt;
    }

    return cumulative_stress;
}
/*
 * This funtion returns cumulative reward with anti-diff
 * Time complexity: O(n) = T(n)
 * Space complexity: O(1)
 */
double calculate_cumulative_reward(const std::vector<VADPoint>& history, 
                                   const double& weightV_reward, 
                                   const double& weightA_reward)
{
    size_t history_size = history.size();

    // if sample is too small
    if (history_size < 2) 
        return 0.0;

    double cumulative_reward = 0.0;
    for(size_t i = 1; i < history_size; i++)
    {
        const auto& current = history[i];
        const auto& prev = history[i-1];

        double dt = current.timestamp - prev.timestamp;
        if (dt <= 0) 
            dt = 0.1;
        
        double instant_reward = calculate_reward_index(current, 
                                                         weightV_reward, 
                                                         weightA_reward);
        
        cumulative_reward += instant_reward * dt;
    }

    return cumulative_reward;
}
/**
 * This function is package of get_cumulative_data, stress, and reward.
 * This will run as multithreading
 * Time complexity = O(n) = T(n) + T(n) + T(1)
 * Space complexity = O(1)
 */
Ratio get_Tn_functions_async(const std::vector<VADPoint>& history, 
                                        const VADPoint& baseline, 
                                        const double& stabilityRadius,
                                        const double& weightA_stress, 
                                        const double& weightV_stress,
                                        const double& dampening_factor,
                                        const double& weightV_reward, 
                                        const double& weightA_reward)
{
    double cumcul_stress = calculate_cumulative_stress(history, baseline, stabilityRadius, weightA_stress, weightV_stress, dampening_factor);
    double cumcul_reward = calculate_cumulative_reward(history, weightV_reward, weightA_reward);

    return get_stress_reward_ratio(cumcul_stress, cumcul_reward);
}
// O(n) ------------------------------------------------------------------------------


// analize 
AnalysisResult EGO_compute(const compute_in& user_in)
{
    EGO_axis base = user_in.emotion_base.value_or(EGO_axis{});
    weight w = user_in.weights.value_or(weight{});
    variable v = user_in.variables.value_or(variable{});

    // thread executes O(n) = 2T(n) task
    auto thread_average = std::async(std::launch::async,
        calculate_average,
        std::ref(user_in.history)
    );

    // thread executes O(n) = T(n) + T(n) + T(1)
    auto thread_Tn_funcs = std::async(std::launch::async,
        get_Tn_functions_async,
        std::ref(user_in.history),
        std::ref(base.baseline),
        base.stabilityRadius,
        w.weightA_stress,
        w.weightV_stress,
        v.dampening_factor,
        w.weightV_reward,
        w.weightA_reward
    );

    // thread executes O(1) bundle
    auto thread_O1_funcs = std::async(std::launch::async,
        get_O1_functions_async,
        std::ref(user_in.prev),
        std::ref(user_in.current),
        std::ref(base.baseline),
        base.stabilityRadius,
        w.weightA_stress,
        w.weightV_stress,
        w.weightV_reward,
        w.weightA_reward,
        v.dampening_factor,
        w.weight_k,
        v.theta_0
    );

    // get result from threads
    VAD_ave average_result =  thread_average.get();
    Ratio n_results = thread_Tn_funcs.get();
    O1_Tasks_Result o1_results = thread_O1_funcs.get();

    // build result
    AnalysisResult final_result;
    // InstantMetrics
    final_result.instant.stress = o1_results.instant_stress;
    final_result.instant.reward = o1_results.instant_reward;
    final_result.instant.ratio_total = o1_results.instant_ratio_total;
    final_result.instant.stress_ratio = o1_results.instant_stress_ratio;
    final_result.instant.reward_ratio = o1_results.instant_reward_ratio;
    final_result.instant.deviation = o1_results.deviation;

    // DynamicMetrics
    final_result.dynamics.delta = o1_results.delta;
    final_result.dynamics.affective_lability = o1_results.affective_lability;

    // CumulativeMetrics
    final_result.cumulative.average_area = average_result;
    final_result.cumulative.stress = n_results.stress_raw;
    final_result.cumulative.reward = n_results.reward_raw;
    final_result.cumulative.total = n_results.ratio_total;
    final_result.cumulative.stress_ratio = n_results.stress_ratio;
    final_result.cumulative.reward_ratio = n_results.reward_ratio;

    return final_result;
}