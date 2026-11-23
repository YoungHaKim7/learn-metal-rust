# Learn Metal with Rust (`metal-cpp-rs`)

A Rust port of the [LearnMetalCPP](../LearnMetalCPP) samples — each sample
incrementally introduces one Metal concept, building on the previous one.

These samples use [`metal-cpp-rs`](https://github.com/YoungHaKim7/metal-cpp-rs),
a Rust conversion of Apple's `metal-cpp` library that talks directly to the
Objective-C runtime with zero third-party dependencies.

## Samples

| # | Folder | Demonstrates |
|---|---|---|
| 00 | `00_window/` | Create a window for Metal rendering — clears to solid red |
| 01 | `01_primitive/` | Render a triangle using vertex + fragment shaders |
| 02 | `02_argbuffers/` | Store shader arguments in an argument buffer |
| 03 | `03_animation/` | Animate rendering — spin the triangle |
| 04 | `04_instancing/` | Draw multiple instances of an object |
| 05 | `05_perspective/` | Render 3D with perspective projection |
| 06 | `06_lighting/` | Light geometry using a Lambert illumination model |
| 07 | `07_texturing/` | Apply a texture to rendered geometry |
| 08 | `08_compute/` | Use the GPU for general-purpose computation |
| 09 | `09_compute_to_render/` | Render the results of a compute kernel |
| 10 | `10_frame_debugging/` | Capture GPU commands for debugging |

## Prerequisites

- **macOS** (Metal only exists on Apple platforms)
- **Rust** 1.71+ (`rustup.rs`)
- **Xcode Command Line Tools** (`xcode-select --install`)

## Running

From the `LearnMetalRust/` directory:

```bash
cargo run -p window              # 00
cargo run -p primitive           # 01
cargo run -p argbuffers          # 02
cargo run -p animation           # 03
cargo run -p instancing          # 04
cargo run -p perspective         # 05
cargo run -p lighting            # 06
cargo run -p texturing           # 07
cargo run -p compute             # 08
cargo run -p compute_to_render   # 09
cargo run -p frame_debugging     # 10
```

## Architecture

```
LearnMetalRust/
├── Cargo.toml              ← workspace root
├── README.md               ← this file
├── shared/                 ← shared library: window + CAMetalLayer + event loop
│   ├── Cargo.toml
│   └── src/lib.rs
├── 00_window/              ← each sample is a standalone binary crate
│   ├── Cargo.toml
│   └── src/main.rs
├── 01_primitive/
│   └── ...
└── 10_frame_debugging/
    └── ...
```

### The `shared` crate

Every sample reuses the `shared` crate for the ~150 lines of boilerplate
that every Metal app needs:

- Creating an `NSApplication` + `NSWindow`
- Attaching a `CAMetalLayer` as the window's content view
- Creating a `MTLDevice` + `MTLCommandQueue`
- Pumping the AppKit event loop (ESC to quit)
- Per-frame: acquiring the next drawable, handing it to your render closure

Each sample's `main.rs` then focuses on the **Metal-specific** logic:
compiling shaders, building pipeline states, encoding draw commands.

### The `metal-cpp-rs` crate

[metal-cpp-rs](https://github.com/YoungHaKim7/metal-cpp-rs) is an
ABI-faithful Rust conversion of Apple's `metal-cpp`. It wraps every Metal
class in a Rust type whose methods call `objc_msgSend` with the exact
Objective-C selectors — same technique as `metal-cpp` itself, but from Rust.

Key patterns you'll see in every sample:

```rust
use metal_cpp_rs::prelude::*;
use metal_cpp_rs::objc_abi::{autoreleasepool, Strong, Ref};

// Create a device:
let device = MTLCreateSystemDefaultDevice();

// Compile shader source at runtime:
let library = device.newLibrary(
    NSString::from_str(source).as_ref(),
    MTLCompileOptions::new().as_ref(),
    &mut Strong::<MTLError>::nil(),
);

// Build a render pipeline:
let desc = MTLRenderPipelineDescriptor::new();
desc.setVertexFunction(vs_fn.as_ref());
desc.setFragmentFunction(fs_fn.as_ref());
desc.colorAttachments().objectAtIndexedSubscript(0)
    .setPixelFormat(MTLPixelFormat::PixelFormatBGRA8Unorm);
let pipeline = device.newRenderPipelineState(desc.as_ref(), &mut error);

// Encode a draw:
let encoder = command_buffer.renderCommandEncoder(render_pass.as_ref());
encoder.setRenderPipelineState(pipeline.as_ref());
encoder.setVertexBuffer(vertex_buffer, 0, 0);
encoder.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 3);
encoder.endEncoding();
```

## Comparison with LearnMetalCPP

The C++ samples in `LearnMetalCPP/` use Apple's `metal-cpp` directly. These
Rust samples mirror them concept-for-concept:

| C++ (metal-cpp) | Rust (metal-cpp-rs) |
|---|---|
| `MTL::Device device = MTL::CreateSystemDefaultDevice();` | `let device = MTLCreateSystemDefaultDevice();` |
| `NS::String* s = NS::String::string("hi", NS::UTF8StringEncoding);` | `let s = NSString::from_str("hi");` |
| `device->newCommandQueue()` | `device.newCommandQueue()` |
| `encoder->setRenderPipelineState(pipeline);` | `encoder.setRenderPipelineState(pipeline.as_ref());` |
| `encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);` | `encoder.drawPrimitives(MTLPrimitiveType::PrimitiveTypeTriangle, 0, 3);` |
| `NS::RetainPtr<T>` / `NS::TransferPtr<T>` | `Strong<T>` / `Ref<T>` |

The ownership model is identical: methods that create objects return
`Strong<T>` (owning, releases on drop — like `NS::RetainPtr`); property
getters return `Ref<T>` (non-owning view — like the raw pointer metal-cpp
returns).

## Next steps

1. **Run each sample in order** — each builds on the previous one.
2. **Modify them.** Change the clear color in `00_window`. Add a second
   triangle in `01_primitive`. Change the rotation speed in `03_animation`.
3. **Read the shader source** — each sample's Metal Shading Language (MSL)
   is embedded as a `const &str` at the top of the file.
4. **Open the GPU capture** from `10_frame_debugging` in Xcode to inspect
   every draw call.

## License

MIT OR Apache-2.0 — same as the upstream `metal-cpp-rs` and `metal-cpp`.
