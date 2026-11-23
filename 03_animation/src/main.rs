//! # 03 — Animation: Spin the Triangle
//!
//! Same as 02, but the rotation angle in the argument buffer is updated
//! every frame based on elapsed time, producing a spinning triangle.
//!
//! Run with: `cargo run -p animation`

use shared::{
    App, Frame, MTLClearColor, MTLCompileOptions, MTLLoadAction, MTLPrimitiveType,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions, MTLStoreAction,
    MTLPixelFormat,
};
use metal_cpp_rs::objc_abi::Ref;
use metal_cpp_rs::prelude::*;

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VertexIn { float2 position; float4 color; };
struct VertexOut { float4 position [[position]]; float4 color; };
struct Args { float rotation; };
vertex VertexOut vs(device const VertexIn* v [[buffer(0)]],
                    constant Args& args [[buffer(1)]],
                    uint vid [[vertex_id]]) {
    float2 p = v[vid].position;
    float c = cos(args.rotation), s = sin(args.rotation);
    VertexOut o;
    o.position = float4(p.x*c - p.y*s, p.x*s + p.y*c, 0.0, 1.0);
    o.color = v[vid].color;
    return o;
}
fragment float4 fs(VertexOut i [[stage_in]]) { return i.color; }
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex { position: [f32; 2], color: [f32; 4] }

const VERTICES: [Vertex; 3] = [
    Vertex { position: [-0.5, -0.5], color: [1.0, 0.2, 0.2, 1.0] },
    Vertex { position: [ 0.5, -0.5], color: [0.2, 1.0, 0.2, 1.0] },
    Vertex { position: [ 0.0,  0.5], color: [0.2, 0.2, 1.0, 1.0] },
];

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct Args { rotation: f32 }

fn main() {
    let app = App::new("03 - Animation", 800.0, 600.0);
    let device = app.device().as_ref();
    let pipeline = unsafe { build_pipeline(device, SHADER_SOURCE, "vs", "fs") };
    let vertex_buffer = unsafe {
        let b = device.newBuffer(
            (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64,
            MTLResourceOptions::ResourceStorageModeShared,
        );
        std::ptr::copy_nonoverlapping(VERTICES.as_ptr(), b.contents() as *mut Vertex, VERTICES.len());
        b
    };
    let arg_buffer = unsafe {
        device.newBuffer(std::mem::size_of::<Args>() as u64, MTLResourceOptions::ResourceStorageModeShared)
    };

    app.run(|frame: &Frame| {
        unsafe {
            // Animate: rotate at ~1 radian per second.
            let args = Args { rotation: frame.time };
            *(arg_buffer.contents() as *mut Args) = args;
            render_frame(frame, pipeline.as_ref(), vertex_buffer.as_ref(), arg_buffer.as_ref())
        }
    });
}

unsafe fn build_pipeline(device: Ref<MTLDevice>, src: &str, vs: &str, fs: &str) -> Strong<MTLRenderPipelineState> {
    let library = device.newLibrary(NSString::from_str(src).as_ref(), MTLCompileOptions::new().as_ref(), &mut Strong::<MTLError>::nil());
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(library.newFunction(NSString::from_str(vs).as_ref()).as_ref());
    desc.setFragmentFunction(library.newFunction(NSString::from_str(fs).as_ref()).as_ref());
    desc.colorAttachments().objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::PixelFormatBGRA8Unorm);
    device.newRenderPipelineState(desc.as_ref(), &mut Strong::<MTLError>::nil())
}

unsafe fn render_frame(frame: &Frame, pipeline: Ref<MTLRenderPipelineState>, vb: Ref<MTLBuffer>, ub: Ref<MTLBuffer>) {
    let tex: Ref<MTLTexture> = msg_send!(frame.drawable, Sel::get("texture\0".as_ptr()).unwrap());
    let rp = MTLRenderPassDescriptor::renderPassDescriptor();
    let ca = rp.colorAttachments().objectAtIndexedSubscript(0);
    ca.as_attachment().setTexture(tex);
    ca.as_attachment().setLoadAction(MTLLoadAction::LoadActionClear);
    ca.as_attachment().setStoreAction(MTLStoreAction::StoreActionStore);
    ca.setClearColor(MTLClearColor::Make(0.05, 0.05, 0.08, 1.0));
    let cb = frame.queue.commandBuffer();
    let enc = cb.renderCommandEncoder(rp.as_ref());
    enc.setRenderPipelineState(pipeline);
    enc.setVertexBuffer(vb, 0, 0);
    enc.setVertexBuffer(ub, 0, 1);
    enc.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 3);
    enc.endEncoding();
    let _: () = msg_send!(cb, Sel::get("presentDrawable:\0".as_ptr()).unwrap(), frame.drawable);
    cb.commit();
}
