//! # 00 — Window: Create a Window for Metal Rendering
//!
//! The "hello world" of Metal: open a macOS window with a `CAMetalLayer`
//! swapchain and clear every frame to solid red.
//!
//! This is the Rust equivalent of the first sample in
//! `docs/LearnMetalCPP/`. Every subsequent sample builds on this pattern:
//!
//! ```text
//! device → command queue → command buffer → render pass descriptor →
//!   render command encoder → (draw commands) → endEncoding → commit → present
//! ```
//!
//! Run with: `cargo run -p window`

use shared::{App, Frame, MTLClearColor, MTLLoadAction, MTLRenderPassDescriptor, MTLStoreAction};
use metal_cpp_rs::objc_abi::Ref;
use metal_cpp_rs::prelude::*;

fn main() {
    let app = App::new("00 - Window", 800.0, 600.0);
    println!("Metal device: {}", app.device().name().to_string());
    app.run(render_frame);
}

/// Per-frame callback: build a render pass that clears to red, then commit.
fn render_frame(frame: &Frame) {
    unsafe {
        // 1. Get the drawable's texture — this is what we render into.
        //    The drawable is a `CAMetalDrawable`; its `texture` method
        //    returns the `MTLTexture` we attach to the render pass.
        let texture_sel = metal_cpp_rs::objc_abi::Sel::get("texture\0".as_ptr()).unwrap();
        let texture: Ref<MTLTexture> = msg_send!(frame.drawable, texture_sel);

        // 2. Create a render pass descriptor. This describes: what textures
        //    we render into, how to clear them, what to do with the result.
        let render_pass = MTLRenderPassDescriptor::renderPassDescriptor();
        let color0 = render_pass.colorAttachments().objectAtIndexedSubscript(0);
        let attachment = color0.as_attachment();
        attachment.setTexture(texture);
        attachment.setLoadAction(MTLLoadAction::LoadActionClear);
        attachment.setStoreAction(MTLStoreAction::StoreActionStore);
        // Solid red clear color: (r=1, g=0, b=0, a=1).
        color0.setClearColor(MTLClearColor::Make(1.0, 0.0, 0.0, 1.0));

        // 3. Create a command buffer from the queue. All GPU work is
        //    encoded into a command buffer, then submitted to the queue.
        let command_buffer = frame.queue.commandBuffer();

        // 4. Get a render command encoder from the buffer. The encoder
        //    records draw commands into the command buffer.
        let encoder = command_buffer.renderCommandEncoder(render_pass.as_ref());

        // No draw commands yet — we're just clearing. End encoding so the
        // command buffer can be submitted.
        encoder.endEncoding();

        // 5. Present the drawable to the screen when the command buffer
        //    finishes executing, then commit.
        let present_sel = metal_cpp_rs::objc_abi::Sel::get("presentDrawable:\0".as_ptr()).unwrap();
        let _: () = msg_send!(command_buffer, present_sel, frame.drawable);

        command_buffer.commit();
    }
}
