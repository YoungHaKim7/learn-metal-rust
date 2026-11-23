//! # 06 — Lighting: Light Geometry (Lambert Model)
//!
//! Adds per-vertex normals and a Lambert illumination model: each vertex
//! has a normal, and the fragment color is scaled by `max(dot(N, L), 0)`
//! where N is the surface normal and L is the light direction.
//!
//! Run with: `cargo run -p lighting`

use metal_cpp_rs::objc_abi::Ref;
use metal_cpp_rs::prelude::*;
use shared::{
    App, Frame, MTLClearColor, MTLCompileOptions, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions, MTLStoreAction,
};

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VertexIn { float3 position; float3 normal; float4 color; };
struct Uniforms { float4x4 mvp; float3 light_dir; };
struct VertexOut { float4 position [[position]]; float4 color; float3 normal; };
vertex VertexOut vs(device const VertexIn* v [[buffer(0)]],
                    constant Uniforms& u [[buffer(1)]],
                    uint vid [[vertex_id]]) {
    VertexOut o;
    o.position = u.mvp * float4(v[vid].position, 1.0);
    o.color = v[vid].color;
    o.normal = v[vid].normal;
    return o;
}
fragment float4 fs(VertexOut i [[stage_in]], constant Uniforms& u [[buffer(0)]]) {
    float3 N = normalize(i.normal);
    float3 L = normalize(u.light_dir);
    float diff = max(dot(N, L), 0.0);
    // Add a small ambient term so the dark side isn't pure black.
    float intensity = 0.2 + 0.8 * diff;
    return i.color * intensity;
}
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

// A small cube (12 triangles) with normals.
const VERTICES: [Vertex; 36] = [
    // Front face (+Z)
    Vertex {
        position: [-0.5, -0.5, 0.5],
        normal: [0, 0, 1],
        color: [1, 0.2, 0.2, 1],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        normal: [0, 0, 1],
        color: [1, 0.2, 0.2, 1],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        normal: [0, 0, 1],
        color: [1, 0.2, 0.2, 1],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        normal: [0, 0, 1],
        color: [1, 0.2, 0.2, 1],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        normal: [0, 0, 1],
        color: [1, 0.2, 0.2, 1],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        normal: [0, 0, 1],
        color: [1, 0.2, 0.2, 1],
    },
    // Back face (-Z)
    Vertex {
        position: [0.5, -0.5, -0.5],
        normal: [0, 0, -1],
        color: [0.2, 1, 0.2, 1],
    },
    Vertex {
        position: [-0.5, -0.5, -0.5],
        normal: [0, 0, -1],
        color: [0.2, 1, 0.2, 1],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        normal: [0, 0, -1],
        color: [0.2, 1, 0.2, 1],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        normal: [0, 0, -1],
        color: [0.2, 1, 0.2, 1],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        normal: [0, 0, -1],
        color: [0.2, 1, 0.2, 1],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        normal: [0, 0, -1],
        color: [0.2, 1, 0.2, 1],
    },
    // Top face (+Y)
    Vertex {
        position: [-0.5, 0.5, 0.5],
        normal: [0, 1, 0],
        color: [0.2, 0.2, 1, 1],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        normal: [0, 1, 0],
        color: [0.2, 0.2, 1, 1],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        normal: [0, 1, 0],
        color: [0.2, 0.2, 1, 1],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        normal: [0, 1, 0],
        color: [0.2, 0.2, 1, 1],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        normal: [0, 1, 0],
        color: [0.2, 0.2, 1, 1],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        normal: [0, 1, 0],
        color: [0.2, 0.2, 1, 1],
    },
    // Bottom face (-Y)
    Vertex {
        position: [-0.5, -0.5, -0.5],
        normal: [0, -1, 0],
        color: [1, 1, 0.2, 1],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        normal: [0, -1, 0],
        color: [1, 1, 0.2, 1],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        normal: [0, -1, 0],
        color: [1, 1, 0.2, 1],
    },
    Vertex {
        position: [-0.5, -0.5, -0.5],
        normal: [0, -1, 0],
        color: [1, 1, 0.2, 1],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        normal: [0, -1, 0],
        color: [1, 1, 0.2, 1],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        normal: [0, -1, 0],
        color: [1, 1, 0.2, 1],
    },
    // Right face (+X)
    Vertex {
        position: [0.5, -0.5, 0.5],
        normal: [1, 0, 0],
        color: [1, 0.5, 0.2, 1],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        normal: [1, 0, 0],
        color: [1, 0.5, 0.2, 1],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        normal: [1, 0, 0],
        color: [1, 0.5, 0.2, 1],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        normal: [1, 0, 0],
        color: [1, 0.5, 0.2, 1],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        normal: [1, 0, 0],
        color: [1, 0.5, 0.2, 1],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        normal: [1, 0, 0],
        color: [1, 0.5, 0.2, 1],
    },
    // Left face (-X)
    Vertex {
        position: [-0.5, -0.5, -0.5],
        normal: [-1, 0, 0],
        color: [0.5, 0.2, 1, 1],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        normal: [-1, 0, 0],
        color: [0.5, 0.2, 1, 1],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        normal: [-1, 0, 0],
        color: [0.5, 0.2, 1, 1],
    },
    Vertex {
        position: [-0.5, -0.5, -0.5],
        normal: [-1, 0, 0],
        color: [0.5, 0.2, 1, 1],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        normal: [-1, 0, 0],
        color: [0.5, 0.2, 1, 1],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        normal: [-1, 0, 0],
        color: [0.5, 0.2, 1, 1],
    },
];

#[repr(C)]
#[derive(Copy, Clone)]
struct Mat4([[f32; 4]; 4]);

#[repr(C)]
#[derive(Copy, Clone)]
struct Uniforms {
    mvp: Mat4,
    light_dir: [f32; 3],
}

fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov / 2.0).tan();
    Mat4([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ])
}

fn rotate_y(angle: f32) -> Mat4 {
    let c = angle.cos();
    let s = angle.sin();
    Mat4([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    Mat4([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ])
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

fn main() {
    let app = App::new("06 - Lighting", 800.0, 600.0);
    let device = app.device().as_ref();
    let pipeline = unsafe { build_pipeline(device, SHADER_SOURCE, "vs", "fs") };
    let vb = unsafe { upload(device, &VERTICES) };
    // Uniforms: Mat4 (64 bytes) + float3 (12 bytes, padded to 16) = 80 bytes.
    let ub = unsafe { device.newBuffer(80, MTLResourceOptions::ResourceStorageModeShared) };

    app.run(|frame: &Frame| unsafe {
        let aspect = frame.width as f32 / frame.height as f32;
        let proj = perspective(60.0f32.to_radians(), aspect, 0.1, 100.0);
        let rot = rotate_y(frame.time * 0.5);
        let trans = translate(0.0, 0.0, -3.0);
        let model = mat_mul(&trans, &rot);
        let mvp = mat_mul(&proj, &model);
        let uniforms = Uniforms {
            mvp,
            light_dir: [0.5, 0.8, -0.3],
        };
        let dst = ub.contents() as *mut Uniforms;
        *dst = uniforms;

        let tex: Ref<MTLTexture> =
            msg_send!(frame.drawable, Sel::get("texture\0").as_ptr().unwrap());
        let rp = MTLRenderPassDescriptor::renderPassDescriptor();
        let ca = rp.colorAttachments().objectAtIndexedSubscript(0);
        ca.as_attachment().setTexture(tex);
        ca.as_attachment()
            .setLoadAction(MTLLoadAction::LoadActionClear);
        ca.as_attachment()
            .setStoreAction(MTLStoreAction::StoreActionStore);
        ca.setClearColor(MTLClearColor::Make(0.05, 0.05, 0.08, 1.0));
        let cb = frame.queue.commandBuffer();
        let enc = cb.renderCommandEncoder(rp.as_ref());
        enc.setRenderPipelineState(pipeline.as_ref());
        enc.setVertexBuffer(vb.as_ref(), 0, 0);
        enc.setVertexBuffer(ub.as_ref(), 0, 1);
        // Fragment shader also reads uniforms from buffer(0).
        enc.setFragmentBuffer(ub.as_ref(), 0, 0);
        // Draw 12 triangles (36 vertices).
        enc.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 36);
        enc.endEncoding();
        let _: () = msg_send!(
            cb,
            Sel::get("presentDrawable:\0").as_ptr().unwrap(),
            frame.drawable
        );
        cb.commit();
    });
}

unsafe fn build_pipeline(
    device: Ref<MTLDevice>,
    src: &str,
    vs: &str,
    fs: &str,
) -> Strong<MTLRenderPipelineState> {
    let library = device.newLibrary(
        NSString::from_str(src).as_ref(),
        MTLCompileOptions::new().as_ref(),
        &mut Strong::<MTLError>::nil(),
    );
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(
        library
            .newFunction(NSString::from_str(vs).as_ref())
            .as_ref(),
    );
    desc.setFragmentFunction(
        library
            .newFunction(NSString::from_str(fs).as_ref())
            .as_ref(),
    );
    desc.colorAttachments()
        .objectAtIndexedSubscript(0)
        .setPixelFormat(MTLPixelFormat::PixelFormatBGRA8Unorm);
    // Enable depth testing (would need a depth attachment in the render pass).
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
