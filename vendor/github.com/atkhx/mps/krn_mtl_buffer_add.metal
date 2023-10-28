#include <metal_stdlib>

using namespace metal;

kernel void add(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] += srcBuffer[id];
}

kernel void addTo(
    device float *dstBuffer [[ buffer(0) ]],
    device float *aBuffer [[ buffer(1) ]],
    device float *bBuffer [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = aBuffer[id] + bBuffer[id];
}