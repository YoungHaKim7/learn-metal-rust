//! # 08 — Compute: Use the GPU for General Purpose Computation
//!
//! Runs a compute kernel that fills a buffer with computed values, then
//! reads them back to the CPU and prints a few. This is the foundation
//! for image processing, particle simulation, and other GPU compute tasks.
//!
//! Run with: `cargo run -p compute`

use metal_cpp_rs::objc_abi::Ref;
use metal_cpp_rs::prelude::*;
use shared::{App, MTLCompileOptions, MTLComputePipelineDescriptor, MTLResourceOptions};

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void fill(device float* out [[buffer(0)]],
                  constant float& scale [[buffer(1)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = sin(float(id) * scale) * 0.5 + 0.5;
}
"#;

const COUNT: u64 = 256;

fn main() {
    let app = App::new("08 - Compute", 400.0, 200.0);
    let device = app.device().as_ref();

    // Compile the compute shader.
    let src = NSString::from_str(SHADER_SOURCE);
    let mut err = Strong::<MTLError>::nil();
    let library =
        unsafe { device.newLibrary(src.as_ref(), MTLCompileOptions::new().as_ref(), &mut err) };
    if library.is_nil() {
        panic!("shader compile failed");
    }

    let kernel_fn = unsafe { library.newFunction(NSString::from_str("fill").as_ref()) };
    if kernel_fn.is_nil() {
        panic!("function lookup failed");
    }

    // Build a compute pipeline.
    let mut err2 = Strong::<MTLError>::nil();
    let pipeline = unsafe { device.newComputePipelineState(kernel_fn.as_ref(), &mut err2) };
    if pipeline.is_nil() {
        panic!("compute pipeline creation failed");
    }

    // Allocate the output buffer + a small uniform buffer for `scale`.
    let out_buffer =
        unsafe { device.newBuffer(COUNT * 4, MTLResourceOptions::ResourceStorageModeShared) };
    let scale_buffer =
        unsafe { device.newBuffer(4, MTLResourceOptions::ResourceStorageModeShared) };

    // Run one compute dispatch when the app starts, print results, then
    // keep the window open (clearing to black) so the user can see it ran.
    let mut computed = false;

    app.run(|frame: &Frame| unsafe {
        if !computed {
            // Write the scale factor.
            *(scale_buffer.contents() as *mut f32) = 0.1;

            // Encode the compute command.
            let cb = frame.queue.commandBuffer();
            let enc = cb.computeCommandEncoder();
            enc.setComputePipelineState(pipeline.as_ref());
            enc.setBuffer(out_buffer.as_ref(), 0, 0);
            enc.setBuffer(scale_buffer.as_ref(), 0, 1);
            // Dispatch COUNT threads in a 1D grid.
            let tg = metal_cpp_rs::metal::types::MTLSize {
                width: COUNT,
                height: 1,
                depth: 1,
            };
            let tp = metal_cpp_rs::metal::types::MTLSize {
                width: pipeline.maxTotalThreadsPerThreadgroup(),
                height: 1,
                depth: 1,
            };
            enc.dispatchThreads(tg, tp);
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();

            // Read back and print.
            let ptr = out_buffer.contents() as *const f32;
            let vals: Vec<f32> = (0..COUNT as usize).map(|i| *ptr.add(i)).collect();
            println!("Compute kernel produced {} values.", vals.len());
            println!("First 8: {:?}", &vals[..8]);
            println!("Last 8:  {:?}", &vals[vals.len() - 8..]);
            computed = true;
        }

        // Just clear the window to black so it stays visible.
        let tex: Ref<MTLTexture> =
            msg_send!(frame.drawable, Sel::get("texture\0").as_ptr().unwrap());
        let rp = MTLRenderPassDescriptor::renderPassDescriptor();
        let ca = rp.colorAttachments().objectAtIndexedSubscript(0);
        ca.as_attachment().setTexture(tex);
        ca.as_attachment()
            .setLoadAction(MTLLoadAction::LoadActionClear);
        ca.as_attachment()
            .setStoreAction(MTLStoreAction::StoreActionStore);
        ca.setClearColor(MTLClearColor::Make(0.02, 0.02, 0.04, 1.0));
        let cb = frame.queue.commandBuffer();
        let enc = cb.renderCommandEncoder(rp.as_ref());
        enc.endEncoding();
        let _: () = msg_send!(
            cb,
            Sel::get("presentDrawable:\0").as_ptr().unwrap(),
            frame.drawable
        );
        cb.commit();
    });
}
