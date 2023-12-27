#ifndef posEmbeddingsKernel_h
#define posEmbeddingsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol posEmbeddingsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        posEmbedding:(id<MTLBuffer>)posEmbedding
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputGrad:(id<MTLBuffer>)outputGrad
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGrad
        featuresCount:(uint)featuresCount;

@end


@interface posEmbeddingsKernelImpl : NSObject <posEmbeddingsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* posEmbeddingsKernel_h */
