#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <vector>
#include <future>
#include <variant>
#include <cmath>
#include <algorithm>
#include "VAD.hpp"
// return struct------------------------------------------------------------
struct InstantMetrics 
{
    double stress;
    double reward;
    double ratio_total;
    double stress_ratio;
    double reward_ratio;
    double deviation;
};

struct DynamicMetrics 
{
    VADPoint delta;
    double affective_lability;
};

struct CumulativeMetrics 
{
    VAD_ave average_area;
    double stress;
    double reward;
    double total;
    double stress_ratio;
    double reward_ratio;
};

struct AnalysisResult
{
    InstantMetrics instant;
    DynamicMetrics dynamics;
    CumulativeMetrics cumulative;
};
// return struct------------------------------------------------------------

// input struct-------------------------------------------------------------
struct weight
{
    double weightA_stress = 0.7;
    double weightV_stress = 0.3;
    double weightV_reward = 0.5;
    double weightA_reward = 0.5;
    double weight_k = 0.5;
    
};
struct variable
{
    double theta_0 = 0;
    double dampening_factor = 0.08;
};
struct EGO_axis
{
    VADPoint baseline{0.0, 0.0, 0.0, 0.0};
    double stabilityRadius = 0.3;
};
struct compute_in
{
    VADPoint current;
    std::vector<VADPoint> history;

    std::optional<VADPoint> prev;

    std::optional<EGO_axis> emotion_base;
    std::optional<variable> variables;
    std::optional<weight> weights; 
};
// input struct-------------------------------------------------------------

// main --------------------------------------------------------------------
AnalysisResult EGO_compute(const compute_in& user_in);