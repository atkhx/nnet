#ifndef RopeKernel_h
#define RopeKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol RopeKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        headIndex:(uint)headIndex
        headSize:(uint)headSize
        contextLength:(uint)contextLength;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        headIndex:(uint)headIndex
        headSize:(uint)headSize
        contextLength:(uint)contextLength;

@end


@interface RopeKernelImpl : NSObject <RopeKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* RopeKernel_h */
