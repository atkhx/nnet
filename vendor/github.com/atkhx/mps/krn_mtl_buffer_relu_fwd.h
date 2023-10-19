#ifndef KernelMTLBufferReluFwd_h
#define KernelMTLBufferReluFwd_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferReluFwd <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void) reluFwd:(id<MTLBuffer>)buffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface KernelMTLBufferReluFwdImpl : NSObject <KernelMTLBufferReluFwd>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferReluFwd_h */
