//! # 07 — Texturing: Apply a Texture to a Quad
//!
//! Creates a checkered texture on the CPU, uploads it to the GPU, and
//! samples it in the fragment shader using UV coordinates.
//!
//! Run with: `cargo run -p texturing`

use shared::{
    App, Frame, MTLClearColor, MTLCompileOptions, MTLLoadAction, MTLPrimitiveType,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions,
    MTLStoreAction, MTLPixelFormat, MTLStorageMode, MTLTextureUsage, MTLTextureDescriptor,
};
use metal_cpp_rs::objc_abi::Ref;
use metal_cpp_rs::prelude::*;

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VertexIn { float2 position; float2 uv; };
struct VertexOut { float4 position [[position]]; float2 uv; };
vertex VertexOut vs(device const VertexIn* v [[buffer(0)]], uint vid [[vertex_id]]) {
    VertexOut o;
    o.position = float4(v[vid].position, 0.0, 1.0);
    o.uv = v[vid].uv;
    return o;
}
fragment float4 fs(VertexOut i [[stage_in]], texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(address::repeat, filter::linear);
    return tex.sample(s, i.uv);
}
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex { position: [f32; 2], uv: [f32; 2] }

// A quad (two triangles) with UV coordinates.
const VERTICES: [Vertex; 6] = [
    Vertex { position: [-0.5, -0.5], uv: [0.0, 0.0] },
    Vertex { position: [ 0.5, -0.5], uv: [1.0, 0.0] },
    Vertex { position: [ 0.5,  0.5], uv: [1.0, 1.0] },
    Vertex { position: [-0.5, -0.5], uv: [0.0, 0.0] },
    Vertex { position: [ 0.5,  0.5], uv: [1.0, 1.0] },
    Vertex { position: [-0.5,  0.5], uv: [0.0, 1.0] },
];

const TEX_W: u64 = 64;
const TEX_H: u64 = 64;

fn main() {
    let app = App::new("07 - Texturing", 800.0, 600.0);
    let device = app.device().as_ref();
    let pipeline = unsafe { build_pipeline(device, SHADER_SOURCE, "vs", "fs") };
    let vb = unsafe { upload(device, &VERTICES) };

    // Create a checkerboard texture on the CPU.
    let mut pixels = Vec::with_capacity((TEX_W * TEX_H * 4) as usize);
    for y in 0..TEX_H {
        for x in 0..TEX_W {
            let checker = ((x / 8) + (y / 8)) % 2 == 0;
            let c: u8 = if checker { 255 } else { 64 };
            pixels.push(c);     // B
            pixels.push(c);     // G
            pixels.push(c);     // R
            pixels.push(255);   // A
        }
    }

    // Create the texture descriptor + texture.
    let tex_desc = MTLTextureDescriptor::texture2DDescriptor(
        MTLPixelFormat::PixelFormatBGRA8Unorm,
        TEX_W,
        TEX_H,
        false,
    );
    tex_desc.setUsage(MTLTextureUsage::TextureUsageShaderRead);
    tex_desc.setStorageMode(MTLStorageMode::StorageModeManaged);
    let texture = unsafe { device.newTexture(tex_desc.as_ref()) };

    // Upload pixel data into the texture.
    unsafe {
        let region = metal_cpp_rs::metal::types::MTLRegion {
            origin: metal_cpp_rs::metal::types::MTLOrigin { x: 0, y: 0, z: 0 },
            size: metal_cpp_rs::metal::types::MTLSize { width: TEX_W, height: TEX_H, depth: 1 },
        };
        texture.replaceRegion(
            region,
            0,
            pixels.as_ptr() as *const std::ffi::c_void,
            TEX_W * 4,
        );
    }

    app.run(|frame: &Frame| unsafe {
        let tex: Ref<MTLTexture> = msg_send!(frame.drawable, Sel::get("texture\0").as_ptr().unwrap());
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
        enc.setFragmentTexture(texture.as_ref(), 0);
        enc.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 6);
        enc.endEncoding();
        let _: () = msg_send!(cb, Sel::get("presentDrawable:\0").as_ptr().unwrap(), frame.drawable);
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
    let buf = device.newBuffer((std::mem::size_of::<T>() * data.len()) as u64, MTLResourceOptions::ResourceStorageModeShared);
    std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut T, data.len());
    buf
}
