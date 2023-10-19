#include <metal_stdlib>

using namespace metal;

kernel void dropout(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    device float *maskOutBuffer [[ buffer(2) ]],
    constant float& probability [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float randomValue = fract(sin(float(id)) * 43758.5453123);

    if (randomValue < probability) {
        destinationBuffer[id] = 0.0;
        maskOutBuffer[id] = 0.0;
    } else {
        destinationBuffer[id] = sourceBuffer[id];
        maskOutBuffer[id] = 1.0;
    }
}
