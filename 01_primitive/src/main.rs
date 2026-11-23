//! # 01 — Primitive: Render a Triangle
//!
//! Builds on `00_window` to draw a single triangle using a render pipeline
//! with vertex and fragment shaders.
//!
//! New concepts:
//!   * Runtime shader compilation (`device.newLibrary(source, options, &mut err)`)
//!   * Render pipeline state (`MTLRenderPipelineDescriptor` + `device.newRenderPipelineState`)
//!   * Vertex buffer (`device.newBuffer`)
//!   * Drawing (`encoder.drawPrimitives`)
//!
//! The triangle's three vertices have hardcoded positions and colors; the
//! fragment shader interpolates color across the surface.
//!
//! Run with: `cargo run -p primitive`

use shared::{
    App, Frame, MTLClearColor, MTLCompileOptions, MTLLoadAction, MTLPrimitiveType,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions, MTLStoreAction,
    MTLPixelFormat,
};
use metal_cpp_rs::objc_abi::Ref;
use metal_cpp_rs::prelude::*;

/// Metal Shading Language source for the triangle.
///
/// This is C++14 source compiled by the Metal driver at runtime. The vertex
/// function takes a vertex buffer (binding 0) and a vertex ID, outputs a
/// clip-space position plus an interpolated color. The fragment function
/// receives the interpolated color and writes it to the color attachment.
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position;
    float4 color;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut triangle_vertex(device const VertexIn* vertices [[buffer(0)]],
                                  uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    out.color = vertices[vid].color;
    return out;
}

fragment float4 triangle_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;

/// The three triangle vertices: position (x, y in NDC) + RGBA color.
/// Layout must match `VertexIn` in the shader.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

const VERTICES: [Vertex; 3] = [
    Vertex { position: [-0.5, -0.5], color: [1.0, 0.2, 0.2, 1.0] }, // bottom-left, red
    Vertex { position: [ 0.5, -0.5], color: [0.2, 1.0, 0.2, 1.0] }, // bottom-right, green
    Vertex { position: [ 0.0,  0.5], color: [0.2, 0.2, 1.0, 1.0] }, // top-center, blue
];

fn main() {
    let app = App::new("01 - Primitive", 800.0, 600.0);
    let device = app.device().as_ref();

    // 1. Compile the shader source at runtime.
    let pipeline = unsafe { build_pipeline(device, SHADER_SOURCE, "triangle_vertex", "triangle_fragment") };

    // 2. Upload the vertex data to a GPU buffer.
    let vertex_buffer = unsafe {
        device.newBuffer(
            (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64,
            MTLResourceOptions::ResourceStorageModeShared,
        )
    };
    unsafe {
        let dst = vertex_buffer.contents() as *mut Vertex;
        std::ptr::copy_nonoverlapping(VERTICES.as_ptr(), dst, VERTICES.len());
    }

    app.run(|frame: &Frame| {
        unsafe { render_frame(frame, pipeline.as_ref(), vertex_buffer.as_ref()) }
    });
}

/// Compile shader source and build a render pipeline state object.
unsafe fn build_pipeline(
    device: Ref<MTLDevice>,
    source: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
) -> Strong<MTLRenderPipelineState> {
    let src = NSString::from_str(source);
    let mut error = Strong::<MTLError>::nil();
    let options = MTLCompileOptions::new();
    let library = device.newLibrary(src.as_ref(), options.as_ref(), &mut error);
    if library.is_nil() {
        let msg = if error.is_nil() {
            "(unknown)".to_string()
        } else {
            error.localizedDescription().to_string()
        };
        panic!("shader compile failed: {msg}");
    }

    let vertex_fn = library.newFunction(NSString::from_str(vertex_fn_name).as_ref());
    let fragment_fn = library.newFunction(NSString::from_str(fragment_fn_name).as_ref());
    if vertex_fn.is_nil() || fragment_fn.is_nil() {
        panic!("function lookup failed");
    }

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(vertex_fn.as_ref());
    desc.setFragmentFunction(fragment_fn.as_ref());
    desc.colorAttachments()
        .objectAtIndexedSubscript(0)
        .setPixelFormat(MTLPixelFormat::PixelFormatBGRA8Unorm);

    let mut error2 = Strong::<MTLError>::nil();
    let pipeline = device.newRenderPipelineState(desc.as_ref(), &mut error2);
    if pipeline.is_nil() {
        let msg = if error2.is_nil() {
            "(unknown)".to_string()
        } else {
            error2.localizedDescription().to_string()
        };
        panic!("pipeline creation failed: {msg}");
    }
    pipeline
}

/// Per-frame: clear to dark blue, bind pipeline + vertex buffer, draw the triangle.
unsafe fn render_frame(
    frame: &Frame,
    pipeline: Ref<MTLRenderPipelineState>,
    vertex_buffer: Ref<MTLBuffer>,
) {
    // Get the drawable's texture.
    let texture_sel = metal_cpp_rs::objc_abi::Sel::get("texture\0".as_ptr()).unwrap();
    let texture: Ref<MTLTexture> = msg_send!(frame.drawable, texture_sel);

    // Render pass: clear to dark blue.
    let render_pass = MTLRenderPassDescriptor::renderPassDescriptor();
    let color0 = render_pass.colorAttachments().objectAtIndexedSubscript(0);
    let attachment = color0.as_attachment();
    attachment.setTexture(texture);
    attachment.setLoadAction(MTLLoadAction::LoadActionClear);
    attachment.setStoreAction(MTLStoreAction::StoreActionStore);
    color0.setClearColor(MTLClearColor::Make(0.05, 0.05, 0.08, 1.0));

    let command_buffer = frame.queue.commandBuffer();
    let encoder = command_buffer.renderCommandEncoder(render_pass.as_ref());

    encoder.setRenderPipelineState(pipeline);
    // Bind the vertex buffer at index 0 (matching `[[buffer(0)]]` in the shader).
    encoder.setVertexBuffer(vertex_buffer, 0, 0);
    // Draw 3 vertices as a triangle list, starting at vertex 0.
    encoder.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 3);

    encoder.endEncoding();

    // Present + commit.
    let present_sel = metal_cpp_rs::objc_abi::Sel::get("presentDrawable:\0".as_ptr()).unwrap();
    let _: () = msg_send!(command_buffer, present_sel, frame.drawable);
    command_buffer.commit();
}
