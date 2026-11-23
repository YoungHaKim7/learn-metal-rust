//! # 04 — Instancing: Draw Multiple Instances of the Triangle
//!
//! A single `drawPrimitives` call with `instanceCount > 1` lets the GPU
//! replicate the geometry. We pass per-instance offsets via a second
//! vertex buffer so each instance renders at a different position.
//!
//! Run with: `cargo run -p instancing`

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
struct Instance { float2 offset; };
struct VertexOut { float4 position [[position]]; float4 color; };
vertex VertexOut vs(device const VertexIn* v [[buffer(0)]],
                    device const Instance* inst [[buffer(1)]],
                    uint vid [[vertex_id]],
                    uint iid [[instance_id]]) {
    VertexOut o;
    o.position = float4(v[vid].position + inst[iid].offset, 0.0, 1.0);
    o.color = v[vid].color;
    return o;
}
fragment float4 fs(VertexOut i [[stage_in]]) { return i.color; }
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex { position: [f32; 2], color: [f32; 4] }
#[repr(C)]
#[derive(Copy, Clone)]
struct Instance { offset: [f32; 2] }

const VERTICES: [Vertex; 3] = [
    Vertex { position: [-0.3, -0.3], color: [1.0, 0.2, 0.2, 1.0] },
    Vertex { position: [ 0.3, -0.3], color: [0.2, 1.0, 0.2, 1.0] },
    Vertex { position: [ 0.0,  0.3], color: [0.2, 0.2, 1.0, 1.0] },
];

/// Four instances spread across the screen.
const INSTANCES: [Instance; 4] = [
    Instance { offset: [-0.5, -0.3] },
    Instance { offset: [ 0.5, -0.3] },
    Instance { offset: [-0.5,  0.3] },
    Instance { offset: [ 0.5,  0.3] },
];

fn main() {
    let app = App::new("04 - Instancing", 800.0, 600.0);
    let device = app.device().as_ref();
    let pipeline = unsafe { build_pipeline(device, SHADER_SOURCE, "vs", "fs") };
    let vb = unsafe { upload(device, &VERTICES) };
    let ib = unsafe { upload(device, &INSTANCES) };

    app.run(|frame: &Frame| unsafe {
        let tex: Ref<MTLTexture> = msg_send!(frame.drawable, Sel::get("texture\0".as_ptr()).unwrap());
        let rp = MTLRenderPassDescriptor::renderPassDescriptor();
        let ca = rp.colorAttachments().objectAtIndexedSubscript(0);
        ca.as_attachment().setTexture(tex);
        ca.as_attachment().setLoadAction(MTLLoadAction::LoadActionClear);
        ca.as_attachment().setStoreAction(MTLStoreAction::StoreActionStore);
        ca.setClearColor(MTLClearColor::Make(0.05, 0.05, 0.08, 1.0));
        let cb = frame.queue.commandBuffer();
        let enc = cb.renderCommandEncoder(rp.as_ref());
        enc.setRenderPipelineState(pipeline.as_ref());
        enc.setVertexBuffer(vb.as_ref(), 0, 0);
        enc.setVertexBuffer(ib.as_ref(), 0, 1);
        // Draw 3 vertices × 4 instances.
        enc.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 3, 4);
        enc.endEncoding();
        let _: () = msg_send!(cb, Sel::get("presentDrawable:\0".as_ptr()).unwrap(), frame.drawable);
        cb.commit();
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

unsafe fn upload<T: Copy>(device: Ref<MTLDevice>, data: &[T]) -> Strong<MTLBuffer> {
    let buf = device.newBuffer(
        (std::mem::size_of::<T>() * data.len()) as u64,
        MTLResourceOptions::ResourceStorageModeShared,
    );
    std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut T, data.len());
    buf
}
