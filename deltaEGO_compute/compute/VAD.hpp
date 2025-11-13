#pragma once
#include <string>

struct VADPoint
{
    double v;
    double a;
    double d;
    double timestamp;
    std::string owner;
};

struct VAD_ave
{
    double x;
    double y;
    double z;
    double radius;
};
