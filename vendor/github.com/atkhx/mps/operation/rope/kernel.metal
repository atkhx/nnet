#include <metal_stdlib>

using namespace metal;

kernel void rope(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& headIndex [[ buffer(2) ]],
    constant uint& headSize [[ buffer(3) ]],
    constant uint& contextLength [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i = gid.z*headSize*contextLength + gid.y*headSize + gid.x * 2;

    float freq = 1.0 / pow(10000.0, float((headIndex*headSize + gid.x*2) % headSize)/float(headSize));
    float val = float(gid.y) * freq;

    float fcr = cos(val);
    float fci = sin(val);

    outputData[i]   = inputData[i]*fcr - inputData[i+1]*fci;
    outputData[i+1] = inputData[i]*fci + inputData[i+1]*fcr;
}

kernel void ropeGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& headIndex [[ buffer(2) ]],
    constant uint& headSize [[ buffer(3) ]],
    constant uint& contextLength [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i = gid.z*headSize*contextLength + gid.y*headSize + gid.x * 2;

    float freq = 1.0 / pow(10000.0, float((headIndex*headSize + gid.x*2) % headSize)/float(headSize));
    float val = float(gid.y) * freq;

    float fcr = cos(val);
    float fci = sin(val);

    float outputGrad0 = outputGrad[i];
    float outputGrad1 = outputGrad[i+1];

    inputGrad[i]   +=  outputGrad0*fcr + outputGrad1*fci;
    inputGrad[i+1] += -outputGrad0*fci + outputGrad1*fcr;
}
