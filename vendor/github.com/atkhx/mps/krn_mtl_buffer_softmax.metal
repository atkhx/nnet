#include <metal_stdlib>

using namespace metal;

struct Parameters {
    uint width;
};

kernel void exp(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    destinationBuffer[id] = exp(sourceBuffer[id]);
}

kernel void sum(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sumOutBuffer [[ buffer(1) ]],
    constant struct Parameters &params [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * params.width;
    uint endIdx = startIdx + params.width;

    float sumExp = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        sumExp += destinationBuffer[i];
    }

    sumOutBuffer[gid.y] = sumExp;
}

kernel void div(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sumOutBuffer [[ buffer(1) ]],
    constant struct Parameters &params [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * params.width+gid.x;
    destinationBuffer[startIdx] /= sumOutBuffer[gid.y];
}
