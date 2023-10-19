#include <metal_stdlib>

using namespace metal;

kernel void reluBwd(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    device float *maskBuffer [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (maskBuffer[id] > 0) {
        destinationBuffer[id] += sourceBuffer[id];
    }
}