//! # 05 — Perspective: 3D Rendering with Perspective Projection
//!
//! Adds depth via a perspective projection matrix. The triangle is moved
//! back in Z and projected, so it appears smaller and centered.
//!
//! Run with: `cargo run -p perspective`

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
struct VertexIn { float3 position; float4 color; };
struct Uniforms { float4x4 mvp; };
struct VertexOut { float4 position [[position]]; float4 color; };
vertex VertexOut vs(device const VertexIn* v [[buffer(0)]],
                    constant Uniforms& u [[buffer(1)]],
                    uint vid [[vertex_id]]) {
    VertexOut o;
    o.position = u.mvp * float4(v[vid].position, 1.0);
    o.color = v[vid].color;
    return o;
}
fragment float4 fs(VertexOut i [[stage_in]]) { return i.color; }
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex { position: [f32; 3], color: [f32; 4] }

const VERTICES: [Vertex; 3] = [
    Vertex { position: [-0.5, -0.5, 0.0], color: [1.0, 0.2, 0.2, 1.0] },
    Vertex { position: [ 0.5, -0.5, 0.0], color: [0.2, 1.0, 0.2, 1.0] },
    Vertex { position: [ 0.0,  0.5, 0.0], color: [0.2, 0.2, 1.0, 1.0] },
];

/// 4×4 matrix (column-major, matching Metal's float4x4).
#[repr(C)]
#[derive(Copy, Clone)]
struct Mat4([[f32; 4]; 4]);

/// Perspective projection: aspect = width/height, fov in radians, near/far.
fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov / 2.0).tan();
    Mat4([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, (near * far) / (near - far), 0.0],
    ])
}

/// Translate by (x, y, z).
fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    Mat4([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ])
}

fn main() {
    let app = App::new("05 - Perspective", 800.0, 600.0);
    let device = app.device().as_ref();
    let pipeline = unsafe { build_pipeline(device, SHADER_SOURCE, "vs", "fs") };
    let vb = unsafe { upload(device, &VERTICES) };
    // Uniform buffer holds one Mat4 (64 bytes).
    let ub = unsafe { device.newBuffer(64, MTLResourceOptions::ResourceStorageModeShared) };

    app.run(|frame: &Frame| unsafe {
        // Build the MVP: translate the triangle back by -2 in Z, then project.
        let aspect = frame.width as f32 / frame.height as f32;
        let proj = perspective(60.0f32.to_radians(), aspect, 0.1, 100.0);
        let model = translate(0.0, 0.0, -2.0);
        let mvp = mat_mul(&proj, &model);
        *(ub.contents() as *mut Mat4) = mvp;

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
        enc.setVertexBuffer(ub.as_ref(), 0, 1);
        enc.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 3);
        enc.endEncoding();
        let _: () = msg_send!(cb, Sel::get("presentDrawable:\0".as_ptr()).unwrap(), frame.drawable);
        cb.commit();
    });
}

fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                r[i][j] += a.0[k][j] * b.0[i][k];
            }
        }
    }
    Mat4(r)
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
    let buf = device.newBuffer((std::mem::size_of::<T>() * data.len()) as u64, MTLResourceOptions::ResourceStorageModeShared);
    std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut T, data.len());
    buf
}
