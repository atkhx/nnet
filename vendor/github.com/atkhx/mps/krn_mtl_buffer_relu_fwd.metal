#include <metal_stdlib>

using namespace metal;

kernel void reluFwd(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (sourceBuffer[id] < 0) {
        destinationBuffer[id] = 0;
    } else {
        destinationBuffer[id] = sourceBuffer[id];
    }
}