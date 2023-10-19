#ifndef KernelMTLBufferReluBwd_h
#define KernelMTLBufferReluBwd_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferReluBwd <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void)reluBwd:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        maskBuffer:(id<MTLBuffer>)maskBuffer
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface KernelMTLBufferReluBwdImpl : NSObject <KernelMTLBufferReluBwd>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferReluBwd_h */
