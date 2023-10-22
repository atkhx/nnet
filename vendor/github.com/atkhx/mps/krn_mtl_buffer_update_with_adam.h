#ifndef KernelMTLBufferUpdateWithAdam_h
#define KernelMTLBufferUpdateWithAdam_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferUpdateWithAdam <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void) updateWithAdam:(id<MTLBuffer>)dataBuffer
             gradBuffer:(id<MTLBuffer>)gradBuffer
             mBuffer:(id<MTLBuffer>)mBuffer
             vBuffer:(id<MTLBuffer>)vBuffer
             beta1:(float)beta1
             beta2:(float)beta2
             beta1powIterationLR:(float)beta1powIterationLR
             beta2powIteration:(float)beta2powIteration
             withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface KernelMTLBufferUpdateWithAdamImpl : NSObject <KernelMTLBufferUpdateWithAdam>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferUpdateWithAdam_h */
