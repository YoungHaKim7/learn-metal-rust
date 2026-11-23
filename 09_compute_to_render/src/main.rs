//! # 09 — Compute-to-Render: Render the Results of a Compute Kernel
//!
//! A compute kernel writes a procedurally generated pattern into a texture
//! each frame. A render pass then samples that texture and draws it as a
//! full-screen quad.
//!
//! This is the foundation for GPU particle systems, fluid simulations, and
//! other "compute then display" pipelines.
//!
//! Run with: `cargo run -p compute_to_render`

use shared::{
    App, Frame, MTLClearColor, MTLCompileOptions, MTLLoadAction, MTLPrimitiveType,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions,
    MTLStoreAction, MTLPixelFormat, MTLStorageMode, MTLTextureUsage, MTLTextureDescriptor,
};
use metal_cpp_rs::objc_abi::Ref;
use metal_cpp_rs::prelude::*;

const COMPUTE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void generate(texture2d<float, access::write> out [[texture(0)]],
                      constant float& time [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint2 dim = out.get_width();
    float2 uv = float2(gid) / float2(dim);
    float2 p = uv * 2.0 - 1.0;
    // Animated plasma effect.
    float v = sin(p.x * 10.0 + time * 2.0) +
              sin(p.y * 10.0 + time * 1.5) +
              sin((p.x + p.y) * 8.0 + time);
    v = v * 0.15 + 0.5;
    float3 col = float3(
        0.5 + 0.5 * sin(v * 3.14 + 0.0),
        0.5 + 0.5 * sin(v * 3.14 + 2.0),
        0.5 + 0.5 * sin(v * 3.14 + 4.0)
    );
    out.write(float4(col, 1.0), gid);
}
"#;

const RENDER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VsOut { float4 position [[position]]; float2 uv; };
vertex VsOut vs(uint vid [[vertex_id]]) {
    // Full-screen triangle (3 vertices, no vertex buffer needed).
    float2 p[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };
    float2 uv[3] = { float2(0,0), float2(2,0), float2(0,2) };
    VsOut o;
    o.position = float4(p[vid], 0, 1);
    o.uv = uv[vid];
    return o;
}
fragment float4 fs(VsOut i [[stage_in]], texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return tex.sample(s, i.uv);
}
"#;

const TEX_W: u64 = 512;
const TEX_H: u64 = 512;

fn main() {
    let app = App::new("09 - Compute-to-Render", 800.0, 600.0);
    let device = app.device().as_ref();

    // Compile both shaders.
    let compute_lib = unsafe {
        device.newLibrary(
            NSString::from_str(COMPUTE_SHADER).as_ref(),
            MTLCompileOptions::new().as_ref(),
            &mut Strong::<MTLError>::nil(),
        )
    };
    let render_lib = unsafe {
        device.newLibrary(
            NSString::from_str(RENDER_SHADER).as_ref(),
            MTLCompileOptions::new().as_ref(),
            &mut Strong::<MTLError>::nil(),
        )
    };
    if compute_lib.is_nil() || render_lib.is_nil() {
        panic!("shader compile failed");
    }

    // Compute pipeline.
    let compute_fn = unsafe { compute_lib.newFunction(NSString::from_str("generate").as_ref()) };
    let compute_pso = unsafe {
        device.newComputePipelineState(compute_fn.as_ref(), &mut Strong::<MTLError>::nil())
    };
    if compute_pso.is_nil() { panic!("compute pipeline failed"); }

    // Render pipeline.
    let render_desc = MTLRenderPipelineDescriptor::new();
    unsafe {
        render_desc.setVertexFunction(render_lib.newFunction(NSString::from_str("vs").as_ref()).as_ref());
        render_desc.setFragmentFunction(render_lib.newFunction(NSString::from_str("fs").as_ref()).as_ref());
        render_desc.colorAttachments().objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::PixelFormatBGRA8Unorm);
    }
    let render_pso = unsafe {
        device.newRenderPipelineState(render_desc.as_ref(), &mut Strong::<MTLError>::nil())
    };
    if render_pso.is_nil() { panic!("render pipeline failed"); }

    // Shared texture: compute writes, render reads.
    let tex_desc = MTLTextureDescriptor::texture2DDescriptor(
        MTLPixelFormat::PixelFormatRGBA32Float,
        TEX_W,
        TEX_H,
        false,
    );
    tex_desc.setUsage(MTLTextureUsage::TextureUsageShaderRead | MTLTextureUsage::TextureUsageShaderWrite);
    tex_desc.setStorageMode(MTLStorageMode::StorageModePrivate); // GPU-only for speed
    let texture = unsafe { device.newTexture(tex_desc.as_ref()) };

    // Uniform buffer for the time value.
    let time_buf = unsafe { device.newBuffer(4, MTLResourceOptions::ResourceStorageModeShared) };

    app.run(|frame: &Frame| unsafe {
        // Update time uniform.
        *(time_buf.contents() as *mut f32) = frame.time;

        let cb = frame.queue.commandBuffer();

        // --- Compute pass: fill the texture ---
        let comp_enc = cb.computeCommandEncoder();
        comp_enc.setComputePipelineState(compute_pso.as_ref());
        comp_enc.setTexture(texture.as_ref(), 0);
        comp_enc.setBuffer(time_buf.as_ref(), 0, 0);
        let grid = metal_cpp_rs::metal::types::MTLSize { width: TEX_W, height: TEX_H, depth: 1 };
        let tg = metal_cpp_rs::metal::types::MTLSize { width: 16, height: 16, depth: 1 };
        comp_enc.dispatchThreads(grid, tg);
        comp_enc.endEncoding();

        // --- Render pass: draw the texture to the screen ---
        let drawable_tex: Ref<MTLTexture> = msg_send!(frame.drawable, Sel::get("texture\0").as_ptr().unwrap());
        let rp = MTLRenderPassDescriptor::renderPassDescriptor();
        let ca = rp.colorAttachments().objectAtIndexedSubscript(0);
        ca.as_attachment().setTexture(drawable_tex);
        ca.as_attachment().setLoadAction(MTLLoadAction::LoadActionClear);
        ca.as_attachment().setStoreAction(MTLStoreAction::StoreActionStore);
        ca.setClearColor(MTLClearColor::Make(0.0, 0.0, 0.0, 1.0));

        let rend_enc = cb.renderCommandEncoder(rp.as_ref());
        rend_enc.setRenderPipelineState(render_pso.as_ref());
        rend_enc.setFragmentTexture(texture.as_ref(), 0);
        // Draw a full-screen triangle (3 vertices, no vertex buffer).
        rend_enc.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 3);
        rend_enc.endEncoding();

        // Present + commit.
        let _: () = msg_send!(cb, Sel::get("presentDrawable:\0").as_ptr().unwrap(), frame.drawable);
        cb.commit();
    });
}
