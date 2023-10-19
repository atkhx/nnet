#include <metal_stdlib>

using namespace metal;

kernel void mul(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *multiplierBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    destinationBuffer[id] *= multiplierBuffer[id];
}