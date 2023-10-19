#include <metal_stdlib>

using namespace metal;

kernel void fill(
    device float *buffer [[ buffer(0) ]],
    const uint id [[ thread_position_in_grid ]],
    constant float& value [[ buffer(1) ]])
{
    buffer[id] = value;
}