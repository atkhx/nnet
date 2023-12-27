#include <metal_stdlib>

using namespace metal;

struct Parameters {
    uint width;
};

kernel void copy(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = srcBuffer[id];
}

kernel void copyWHD(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    constant uint& height [[ buffer(3) ]],
    constant uint& square [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    dstBuffer[gid.z*square + gid.y*width + gid.x] = srcBuffer[gid.z*square + gid.y*width + gid.x];
}

kernel void fill(
    device float *dstBuffer [[ buffer(0) ]],
    const uint id [[ thread_position_in_grid ]],
    constant float& value [[ buffer(1) ]])
{
    dstBuffer[id] = value;
}

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

kernel void addToWHD(
    device float *dstBuffer [[ buffer(0) ]],
    device float *aBuffer [[ buffer(1) ]],
    device float *bBuffer [[ buffer(2) ]],
    constant float& k [[ buffer(3) ]],
    constant uint& width [[ buffer(4) ]],
    constant uint& square [[ buffer(5) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint offset = gid.z*square + gid.y*width + gid.x;
    // dstBuffer[offset] = aBuffer[offset] + bBuffer[offset];
    dstBuffer[offset] = (k * dstBuffer[offset]) + aBuffer[offset] + bBuffer[offset];
}

kernel void addToWHDBwd(
    device float *aGrad [[ buffer(0) ]],
    device float *bGrad [[ buffer(1) ]],
    device float *oGrad [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    constant uint& square [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint offset = gid.z*square + gid.y*width + gid.x;
    aGrad[offset] += oGrad[offset];
    bGrad[offset] += oGrad[offset];
}

kernel void addScalar(
    device float *dstBuffer [[ buffer(0) ]],
    constant float& value [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] += value;
}

kernel void mul(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] *= srcBuffer[id];
}

kernel void maxByRow(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx + width;

    float maxValue = srcBuffer[startIdx];
    for (uint i = startIdx+1; i < endIdx; ++i) {
        if (maxValue < srcBuffer[i]) {
            maxValue = srcBuffer[i];
        }
    }

    dstBuffer[gid.y] = maxValue;
}

kernel void subMaxByRow(
    device float *dstBuffer [[ buffer(0) ]],
    device float *maxBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    dstBuffer[gid.y * width+gid.x] -= maxBuffer[gid.y];
}

kernel void exp(
    device float *destinationBuffer [[ buffer(0) ]],
    device float *sourceBuffer [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    destinationBuffer[id] = exp(sourceBuffer[id]);
}

kernel void sumByRow(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx + width;

    float sumValue = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        sumValue += srcBuffer[i];
    }

    dstBuffer[gid.y] = sumValue;
}

kernel void divOnSum(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    device float *sumBuffer [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    dstBuffer[gid.y * width+gid.x] = srcBuffer[gid.y * width+gid.x] / sumBuffer[gid.y];
}

kernel void mulAndDivOnSum(
    device float *srcBuffer [[ buffer(0) ]],
    device float *dstBuffer [[ buffer(1) ]],
    device float *mulBuffer [[ buffer(2) ]],
    device float *sumBuffer [[ buffer(3) ]],
    constant uint& width [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    // dstBuffer[gid.y * width+gid.x] = mulBuffer[gid.x] * srcBuffer[gid.y * width+gid.x] / sumBuffer[gid.y];
    dstBuffer[gid.y * width+gid.x] = srcBuffer[gid.y * width+gid.x] / sumBuffer[gid.y];
}

kernel void updateWithAdam(
    device float *dataBuffer [[ buffer(0) ]],
    device float *gradBuffer [[ buffer(1) ]],
    device float *mBuffer [[ buffer(2) ]],
    device float *vBuffer [[ buffer(3) ]],
    constant float& beta1 [[ buffer(4) ]],
    constant float& beta2 [[ buffer(5) ]],
    constant float& beta1powIterationLR [[ buffer(6) ]],
    constant float& beta2powIteration [[ buffer(7) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    mBuffer[id] = beta1*mBuffer[id] + (1 - beta1)*gradBuffer[id];
    vBuffer[id] = beta2*vBuffer[id] + (1 - beta2)*gradBuffer[id]*gradBuffer[id];

    dataBuffer[id] -= mBuffer[id] * beta1powIterationLR / (sqrt(vBuffer[id] * beta2powIteration) + 0.000000001);
}

kernel void softmaxTril(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float max = srcBuffer[startIdx];
    for (uint i = startIdx+1; i < endIdx; ++i) {
        if (max < srcBuffer[i]) {
            max = srcBuffer[i];
        }
    }

    float sumExp = 0.0;
    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] = exp(srcBuffer[i]-max);
        sumExp += dstBuffer[i];
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] /= sumExp;
    }
}

kernel void softmaxBufferTrilBwd(
    device float *dstBuffer [[ buffer(0) ]],
    device float *srcBuffer [[ buffer(1) ]],
    device float *smxBuffer [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint startIdx = gid.y * width;
    uint endIdx = startIdx +  gid.y+1;

    float g = 0;
    float s = 0;

    for (uint i = startIdx; i < endIdx; ++i) {
        g = smxBuffer[i] * srcBuffer[i];
        s += g;
        dstBuffer[i] += g;
    }

    for (uint i = startIdx; i < endIdx; ++i) {
        dstBuffer[i] -= smxBuffer[i] * s;
    }
}

kernel void nllByPos(
    device float *dstBuffer [[ buffer(0) ]],
    device float *smxBuffer [[ buffer(1) ]],
    device float *tgtBuffer [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    dstBuffer[id] = -log(smxBuffer[id * width + int(tgtBuffer[id])]);
}

kernel void nllByPosBwd(
    device float *oGrad [[ buffer(0) ]],
    device float *aGrad [[ buffer(1) ]],
    device float *tgtBuffer [[ buffer(2) ]],
    device float *smxBuffer [[ buffer(3) ]],
    constant uint& chunkSize [[ buffer(4) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    int rowIdx = id / chunkSize;
    float softmaxI = smxBuffer[id];
    int tgtIndex = tgtBuffer[rowIdx];

    if (id == tgtIndex + (rowIdx * chunkSize)) {
        aGrad[id] += oGrad[rowIdx] * (softmaxI - 1.0);
    } else {
        aGrad[id] += oGrad[rowIdx] * (softmaxI - 0.0);
    }
}

kernel void crossEntropyPosBwd(
    device float *oGrad [[ buffer(0) ]],
    device float *aGrad [[ buffer(1) ]],
    device float *tgtBuffer [[ buffer(2) ]],
    device float *smxBuffer [[ buffer(3) ]],
    constant uint& chunkSize [[ buffer(4) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    int rowIdx = id / chunkSize;
    float softmaxI = smxBuffer[id];
    int tgtIndex = tgtBuffer[rowIdx];

    if (id == tgtIndex + (rowIdx * chunkSize)) {
        aGrad[id] += oGrad[rowIdx] * (softmaxI - 1);
    } else {
        aGrad[id] += oGrad[rowIdx] * softmaxI;
    }
}

kernel void embeddings(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    device float *posEmbedding [[ buffer(2) ]],
    device float *tokenEmbedding [[ buffer(3) ]],
    constant uint& featuresCount [[ buffer(4) ]],
    constant uint& contextLength [[ buffer(5) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    int outputOffset = gid.y*featuresCount + gid.x;

    int posEmbeddingY      = gid.y - (contextLength * int(gid.y/contextLength));
    int posEmbeddingOffset = (posEmbeddingY*featuresCount) + gid.x;

    int tokenEmbeddingY      = inputData[gid.y];
    int tokenEmbeddingOffset = (tokenEmbeddingY * featuresCount) + gid.x;

    outputData[outputOffset] = posEmbedding[posEmbeddingOffset] + tokenEmbedding[tokenEmbeddingOffset];
}

kernel void embeddingsBwd(
    device float *inputData [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    device float *tokenEmbeddingGrad [[ buffer(2) ]],
    constant uint& featuresCount [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    int outputOffset = gid.y*featuresCount + gid.x;

    int tokenEmbeddingY      = inputData[gid.y];
    int tokenEmbeddingOffset = (tokenEmbeddingY * featuresCount) + gid.x;

    tokenEmbeddingGrad[tokenEmbeddingOffset] += outputGrad[outputOffset];
}

kernel void transposeTo(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    constant uint& height [[ buffer(3) ]],
    constant uint& square [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    outputData[gid.z*square + gid.x*height + gid.y] = inputData[gid.z*square + gid.y*width + gid.x];
}

kernel void transposeAndAddTo(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    constant uint& height [[ buffer(3) ]],
    constant uint& square [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    outputData[gid.z*square + gid.x*height + gid.y] += inputData[gid.z*square + gid.y*width + gid.x];
}
