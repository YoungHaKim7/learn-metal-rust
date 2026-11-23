//! # 02 — Argument Buffers: Store Shader Arguments in a Buffer
//!
//! Instead of binding resources one by one to the vertex shader
//! (`setVertexBuffer:offset:atIndex:`), we pack them into an **argument
//! buffer** that the shader indexes indirectly.
//!
//! This is the foundation of Metal's "indirect argument" model — you encode
//! resource bindings into a buffer once, then bind that single buffer in
//! the encoder. Especially useful for MTL4 (Metal 4) where argument tables
//! are the primary binding mechanism.
//!
//! Run with: `cargo run -p argbuffers`

use metal_cpp_rs::objc_abi::Ref;
use metal_cpp_rs::prelude::*;
use shared::{
    App, Frame, MTLClearColor, MTLCompileOptions, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions, MTLStoreAction,
};

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

// The argument buffer contains a single uniform: a rotation angle.
// In a real app you'd pack many resources here (buffers, textures, samplers).
struct Args {
    float rotation;
};

vertex VertexOut triangle_vertex(device const VertexIn* vertices [[buffer(0)]],
                                  constant Args& args [[buffer(1)]],
                                  uint vid [[vertex_id]]) {
    float2 p = vertices[vid].position;
    // Rotate the position by args.rotation radians.
    float c = cos(args.rotation);
    float s = sin(args.rotation);
    float2 r = float2(p.x * c - p.y * s, p.x * s + p.y * c);
    VertexOut out;
    out.position = float4(r, 0.0, 1.0);
    out.color = vertices[vid].color;
    return out;
}

fragment float4 triangle_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

const VERTICES: [Vertex; 3] = [
    Vertex {
        position: [-0.5, -0.5],
        color: [1.0, 0.2, 0.2, 1.0],
    },
    Vertex {
        position: [0.5, -0.5],
        color: [0.2, 1.0, 0.2, 1.0],
    },
    Vertex {
        position: [0.0, 0.5],
        color: [0.2, 0.2, 1.0, 1.0],
    },
];

/// Uniform data matching `struct Args` in the shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct Args {
    rotation: f32,
}

fn main() {
    let app = App::new("02 - Argument Buffers", 800.0, 600.0);
    let device = app.device().as_ref();

    let pipeline = unsafe {
        build_pipeline(
            device,
            SHADER_SOURCE,
            "triangle_vertex",
            "triangle_fragment",
        )
    };

    // Upload vertex data.
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

    // Create the argument buffer. It holds one `Args` struct (4 bytes).
    // In Metal 4 you'd use MTL4::ArgumentTable; here we use the classic
    // MTL buffer approach which works on all Metal versions.
    let arg_buffer = unsafe {
        device.newBuffer(
            std::mem::size_of::<Args>() as u64,
            MTLResourceOptions::ResourceStorageModeShared,
        )
    };

    app.run(|frame: &Frame| {
        unsafe {
            // Update the argument buffer's contents each frame — a fixed
            // rotation angle (no animation yet; that's sample 03).
            let args = Args { rotation: 0.5 };
            let dst = arg_buffer.contents() as *mut Args;
            *dst = args;

            render_frame(
                frame,
                pipeline.as_ref(),
                vertex_buffer.as_ref(),
                arg_buffer.as_ref(),
            )
        }
    });
}

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
        panic!("shader compile failed");
    }
    let vertex_fn = library.newFunction(NSString::from_str(vertex_fn_name).as_ref());
    let fragment_fn = library.newFunction(NSString::from_str(fragment_fn_name).as_ref());
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(vertex_fn.as_ref());
    desc.setFragmentFunction(fragment_fn.as_ref());
    desc.colorAttachments()
        .objectAtIndexedSubscript(0)
        .setPixelFormat(MTLPixelFormat::PixelFormatBGRA8Unorm);
    let mut error2 = Strong::<MTLError>::nil();
    let pipeline = device.newRenderPipelineState(desc.as_ref(), &mut error2);
    if pipeline.is_nil() {
        panic!("pipeline creation failed");
    }
    pipeline
}

unsafe fn render_frame(
    frame: &Frame,
    pipeline: Ref<MTLRenderPipelineState>,
    vertex_buffer: Ref<MTLBuffer>,
    arg_buffer: Ref<MTLBuffer>,
) {
    let texture_sel = metal_cpp_rs::objc_abi::Sel::get("texture\0".as_ptr()).unwrap();
    let texture: Ref<MTLTexture> = msg_send!(frame.drawable, texture_sel);

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
    // buffer(0) = vertices, buffer(1) = argument buffer (the uniforms).
    encoder.setVertexBuffer(vertex_buffer, 0, 0);
    encoder.setVertexBuffer(arg_buffer, 0, 1);
    encoder.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 3);
    encoder.endEncoding();

    let present_sel = metal_cpp_rs::objc_abi::Sel::get("presentDrawable:\0".as_ptr()).unwrap();
    let _: () = msg_send!(command_buffer, present_sel, frame.drawable);
    command_buffer.commit();
}
