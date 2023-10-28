#include <metal_stdlib>

using namespace metal;

kernel void copy(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = srcBuffer[id];
}
