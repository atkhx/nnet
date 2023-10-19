#ifndef KernelMTLBufferDropout_h
#define KernelMTLBufferDropout_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferDropout <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void) dropout:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        maskOutBuffer:(id<MTLBuffer>)maskOutBuffer
        probability:(float)probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface KernelMTLBufferDropoutImpl : NSObject <KernelMTLBufferDropout>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferDropout_h */
