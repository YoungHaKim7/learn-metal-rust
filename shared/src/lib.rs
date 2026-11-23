//! # shared — window + Metal layer + event loop boilerplate
//!
//! Every sample in `LearnMetalRust/` reuses this crate so the per-sample
//! `main.rs` can focus on the Metal concept being introduced (clearing,
//! drawing a triangle, argument buffers, animation, etc.) instead of
//! duplicating ~150 lines of AppKit + CAMetalLayer setup each time.
//!
//! ## What `shared` provides
//!
//! * [`App`] — owns an `NSApplication`, an `NSWindow`, a `CAMetalLayer`
//!   swapchain, a `MTLDevice`, and a `MTLCommandQueue`.
//! * [`App::run`] — pumps the AppKit event loop by hand and calls your
//!   closure once per frame with a [`Frame`] that exposes the device,
//!   queue, current drawable, and dimensions.
//! * [`Frame`] — per-frame context handed to your render closure.
//!
//! ## Usage
//!
//! ```no_run
//! use shared::App;
//!
//! fn main() {
//!     let app = App::new("00 - Window", 800.0, 600.0);
//!     app.run(|frame| {
//!         // Build a render pass that clears to red, then commit.
//!         // (See any sample's main.rs for the full pattern.)
//!     });
//! }
//! ```
//!
//! ## Platform
//!
//! macOS only — depends on AppKit, QuartzCore, Metal, and the Objective-C
//! runtime. `cargo check` works on any host; running requires a Mac.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![allow(clippy::missing_safety_doc, clippy::new_without_default)]

use std::time::Instant;

use metal_cpp_rs::objc_abi::{NSUInteger, Obj, Raw, Ref, Sel, Strong, autoreleasepool};
use metal_cpp_rs::quartz_core::{CAMetalLayer, CGSize};

// ---------------------------------------------------------------------------
// AppKit constants and selectors (the metal-cpp-rs crate wraps Metal and
// Foundation but not AppKit, so we declare the few selectors we need by hand).
// ---------------------------------------------------------------------------

/// NSApplicationActivationPolicyRegular
const ACTIVATION_POLICY_REGULAR: i64 = 0;

/// NSEvent mask constants (subset).
const EVENT_KEY_DOWN: u64 = 10;
const EVENT_KEY_UP: u64 = 11;
const EVENT_FLAGS_CHANGED: u64 = 12;
const EVENT_LEFT_MOUSE_DOWN: u64 = 1;
const EVENT_LEFT_MOUSE_UP: u64 = 2;
const EVENT_MOUSE_MOVED: u64 = 5;
const EVENT_LEFT_MOUSE_DRAGGED: u64 = 6;

/// NSWindowStyleMask bits.
const WINDOW_STYLE_TITLED: u64 = 1;
const WINDOW_STYLE_CLOSABLE: u64 = 2;
const WINDOW_STYLE_MINIATURIZABLE: u64 = 4;
const WINDOW_STYLE_RESIZABLE: u64 = 8;
const WINDOW_STYLE_DEFAULT: u64 = WINDOW_STYLE_TITLED
    | WINDOW_STYLE_CLOSABLE
    | WINDOW_STYLE_MINIATURIZABLE
    | WINDOW_STYLE_RESIZABLE;

/// NSBackingStoreBuffered
const WINDOW_BACKING_BUFFERED: u64 = 2;

/// HID key code for Escape.
pub const KEY_ESC: u16 = 53;

// ---------------------------------------------------------------------------
// AppKit wrappers (minimal — just what we need).
// ---------------------------------------------------------------------------

/// Look up an Objective-C class by name. Returns a `&'static Class` handle.
fn cls(name: &str) -> &'static Obj {
    unsafe {
        let cstr = std::ffi::CString::new(name).unwrap();
        Obj::get(cstr.as_ptr()).expect(&format!("class {name} not found"))
    }
}

/// `NSApplication` — the macOS application object.
struct NSApplication;

impl NSApplication {
    /// `+[NSApplication sharedApplication]`
    fn shared() -> Strong<Obj> {
        unsafe {
            let sel = Sel::get("sharedApplication\0".as_ptr()).unwrap();
            msg_send!(cls("NSApplication"), sel)
        }
    }

    /// `-[NSApplication setActivationPolicy:]`
    fn set_activation_policy(&self, policy: i64) {
        unsafe {
            let sel = Sel::get("setActivationPolicy:\0".as_ptr()).unwrap();
            let _: () = msg_send!(cls("NSApplication"), sel, policy);
        }
    }

    /// `-[NSApplication activateIgnoringOtherApps:]`
    fn activate_ignoring_other_apps(&self, flag: bool) {
        unsafe {
            let sel = Sel::get("activateIgnoringOtherApps:\0".as_ptr()).unwrap();
            let _: () = msg_send!(cls("NSApplication"), sel, flag);
        }
    }
}

// Simpler: keep NSApplication as a Strong<Obj> in App and call msg_send! directly.
// (The struct above is illustrative; the real calls happen inline in App::new.)

/// `NSWindow` — a macOS window.
fn ns_window_alloc() -> Strong<Obj> {
    unsafe {
        let sel = Sel::get("alloc\0".as_ptr()).unwrap();
        msg_send!(cls("NSWindow"), sel)
    }
}

/// `-[NSWindow initWithContentRect:styleMask:backing:defer:]`
unsafe fn ns_window_init(
    win: Ref<Obj>,
    rect: NSRect,
    style: u64,
    backing: u64,
    defer: bool,
) -> Strong<Obj> {
    let sel = Sel::get("initWithContentRect:styleMask:backing:defer:\0".as_ptr()).unwrap();
    msg_send!(win, sel, rect, style, backing, defer)
}

/// `-[NSWindow setTitle:]`
unsafe fn ns_window_set_title(win: Ref<Obj>, title: Ref<Obj>) {
    let sel = Sel::get("setTitle:\0".as_ptr()).unwrap();
    let _: () = msg_send!(win, sel, title);
}

/// `-[NSWindow contentView]`
unsafe fn ns_window_content_view(win: Ref<Obj>) -> Strong<Obj> {
    let sel = Sel::get("contentView\0".as_ptr()).unwrap();
    msg_send!(win, sel)
}

/// `-[NSWindow makeKeyAndOrderFront:]`
unsafe fn ns_window_make_key_and_order_front(win: Ref<Obj>) {
    let sel = Sel::get("makeKeyAndOrderFront:\0".as_ptr()).unwrap();
    let _: () = msg_send!(win, sel, std::ptr::null::<Obj>());
}

/// `-[NSWindow backingScaleFactor]`
unsafe fn ns_window_backing_scale_factor(win: Ref<Obj>) -> f64 {
    let sel = Sel::get("backingScaleFactor\0".as_ptr()).unwrap();
    msg_send!(win, sel)
}

/// `-[NSView setWantsLayer:]`
unsafe fn ns_view_set_wants_layer(view: Ref<Obj>, wants: bool) {
    let sel = Sel::get("setWantsLayer:\0".as_ptr()).unwrap();
    let _: () = msg_send!(view, sel, wants);
}

/// `-[NSView setLayer:]`
unsafe fn ns_view_set_layer(view: Ref<Obj>, layer: Raw) {
    let sel = Sel::get("setLayer:\0".as_ptr()).unwrap();
    let _: () = msg_send!(view, sel, layer);
}

/// `-[NSView bounds]`
unsafe fn ns_view_bounds(view: Ref<Obj>) -> NSRect {
    let sel = Sel::get("bounds\0".as_ptr()).unwrap();
    msg_send!(view, sel)
}

/// `-[NSApplication nextEventMatchingMask:untilDate:inMode:dequeue:]`
unsafe fn ns_app_next_event(
    app: Ref<Obj>,
    mask: u64,
    until: Raw,
    mode: Raw,
    dequeue: bool,
) -> Strong<Obj> {
    let sel = Sel::get("nextEventMatchingMask:untilDate:inMode:dequeue:\0".as_ptr()).unwrap();
    msg_send!(app, sel, mask, until, mode, dequeue)
}

/// `-[NSApplication sendEvent:]`
unsafe fn ns_app_send_event(app: Ref<Obj>, event: Ref<Obj>) {
    let sel = Sel::get("sendEvent:\0".as_ptr()).unwrap();
    let _: () = msg_send!(app, sel, event);
}

/// `-[NSApplication terminate:]`
unsafe fn ns_app_terminate(app: Ref<Obj>) {
    let sel = Sel::get("terminate:\0".as_ptr()).unwrap();
    let _: () = msg_send!(app, sel, std::ptr::null::<Obj>());
}

/// `-[NSEvent type]`
unsafe fn ns_event_type(ev: Ref<Obj>) -> u64 {
    let sel = Sel::get("type\0".as_ptr()).unwrap();
    msg_send!(ev, sel)
}

/// `-[NSEvent keyCode]`
unsafe fn ns_event_key_code(ev: Ref<Obj>) -> u16 {
    let sel = Sel::get("keyCode\0".as_ptr()).unwrap();
    msg_send!(ev, sel)
}

/// `+[NSDate distantPast]`
unsafe fn ns_date_distant_past() -> Strong<Obj> {
    let sel = Sel::get("distantPast\0".as_ptr()).unwrap();
    msg_send!(cls("NSDate"), sel)
}

/// `+[NSDate dateWithTimeIntervalSinceNow:]`
unsafe fn ns_date_with_interval(secs: f64) -> Strong<Obj> {
    let sel = Sel::get("dateWithTimeIntervalSinceNow:\0".as_ptr()).unwrap();
    msg_send!(cls("NSDate"), sel, secs)
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A rectangle (matches NSRect / CGRect layout).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NSRect {
    pub origin: NSPoint,
    pub size: CGSize,
}

/// A 2D point (matches NSPoint / CGPoint layout).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NSPoint {
    pub x: f64,
    pub y: f64,
}

impl NSPoint {
    pub fn new(x: f64, y: f64) -> Self {
        NSPoint { x, y }
    }
}

/// Per-frame context handed to the render closure.
pub struct Frame {
    /// The Metal device.
    pub device: Ref<MTLDevice>,
    /// The command queue (call `.commandBuffer()` to get a fresh buffer).
    pub queue: Ref<MTLCommandQueue>,
    /// The current drawable to render into. Get its texture via
    /// `drawable.texture()`.
    pub drawable: Ref<Obj>,
    /// Drawable width in pixels (already multiplied by backing scale).
    pub width: u32,
    /// Drawable height in pixels.
    pub height: u32,
    /// Seconds since the app started — useful for animation.
    pub time: f32,
    /// Seconds since the previous frame (clamped to 0.1).
    pub dt: f32,
}

/// A macOS window with a Metal layer attached.
pub struct App {
    device: Strong<MTLDevice>,
    queue: Strong<MTLCommandQueue>,
    layer: Strong<CAMetalLayer>,
    window: Strong<Obj>,
    ns_app: Strong<Obj>,
    started: Instant,
    last_frame: Instant,
    running: bool,
}

impl App {
    /// Create a window titled `title` with a `width`×`height` point content
    /// area and a `CAMetalLayer` swapchain attached.
    ///
    /// # Panics
    /// Panics if no Metal device is available or AppKit fails to initialize
    /// (i.e. not running on macOS).
    pub fn new(title: &str, width: f64, height: f64) -> App {
        // Force-load the frameworks we link against. On macOS this is a no-op
        // because the frameworks are already loaded by the dynamic linker,
        // but the calls document the dependency.
        unsafe {
            let device = MTLCreateSystemDefaultDevice();
            if device.is_nil() {
                panic!("MTLCreateSystemDefaultDevice() returned nil — Metal is unavailable");
            }
            let queue = device.newCommandQueue();
            if queue.is_nil() {
                panic!("newCommandQueue() returned nil");
            }

            // NSApplication
            let ns_app = NSApplication::shared();
            ns_app.set_activation_policy(ACTIVATION_POLICY_REGULAR);

            // CAMetalLayer
            let layer = CAMetalLayer::layer();
            layer.setDevice(device.as_ref());
            layer.setPixelFormat(MTLPixelFormat::PixelFormatBGRA8Unorm);
            layer.setFramebufferOnly(true);
            layer.setMaximumDrawableCount(3);

            // NSWindow
            let win_alloc = ns_window_alloc();
            let rect = NSRect {
                origin: NSPoint::new(0.0, 0.0),
                size: CGSize::Make(width, height),
            };
            let window = ns_window_init(
                win_alloc.as_ref(),
                rect,
                WINDOW_STYLE_DEFAULT,
                WINDOW_BACKING_BUFFERED,
                false,
            );
            let title_ns = NSString::from_str(title);
            ns_window_set_title(window.as_ref(), title_ns.as_ref());
            let content = ns_window_content_view(window.as_ref());
            ns_view_set_wants_layer(content.as_ref(), true);
            ns_view_set_layer(content.as_ref(), layer.as_ref().as_raw());
            let bounds = ns_view_bounds(content.as_ref());
            let scale = ns_window_backing_scale_factor(window.as_ref()).max(1.0);
            layer.setDrawableSize(CGSize::Make(
                bounds.size.width * scale,
                bounds.size.height * scale,
            ));

            ns_app.activate_ignoring_other_apps(true);
            ns_window_make_key_and_order_front(window.as_ref());

            App {
                device,
                queue,
                layer,
                window,
                ns_app,
                started: Instant::now(),
                last_frame: Instant::now(),
                running: true,
            }
        }
    }

    /// The Metal device backing this app.
    pub fn device(&self) -> &Strong<MTLDevice> {
        &self.device
    }

    /// Run the event loop until ESC or the close button ends it. Calls `f`
    /// once per frame with a fresh [`Frame`].
    pub fn run<F: FnMut(&Frame)>(mut self, mut f: F) {
        while self.running {
            autoreleasepool(|| unsafe {
                self.pump_events();
                if !self.running {
                    ns_app_terminate(self.ns_app.as_ref());
                    return;
                }
                self.sync_size();
                let frame = self.make_frame();
                if !frame.drawable.is_nil() {
                    f(&frame);
                }
            });
        }
        unsafe { ns_app_terminate(self.ns_app.as_ref()) };
    }

    unsafe fn pump_events(&mut self) {
        let mode_ns = NSString::from_str("kCFRunLoopDefaultMode");
        // Block briefly for the next event, then drain everything pending.
        let mut until = ns_date_with_interval(1.0 / 120.0);
        loop {
            let event = ns_app_next_event(
                self.ns_app.as_ref(),
                NSUInteger::MAX,
                until.as_ref().as_raw(),
                mode_ns.as_ref().as_raw(),
                true,
            );
            if event.is_nil() {
                break;
            }
            let etype = ns_event_type(event.as_ref());
            if etype == EVENT_KEY_DOWN {
                let code = ns_event_key_code(event.as_ref());
                if code == KEY_ESC {
                    self.running = false;
                    break;
                }
            }
            ns_app_send_event(self.ns_app.as_ref(), event.as_ref());
            until = ns_date_distant_past();
        }
    }

    unsafe fn sync_size(&mut self) {
        let content = ns_window_content_view(self.window.as_ref());
        let bounds = ns_view_bounds(content.as_ref());
        let scale = ns_window_backing_scale_factor(self.window.as_ref()).max(1.0);
        self.layer.setDrawableSize(CGSize::Make(
            bounds.size.width * scale,
            bounds.size.height * scale,
        ));
    }

    unsafe fn make_frame(&mut self) -> Frame {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        let drawable = self.layer.nextDrawable();
        let (w, h) = self.drawable_size();
        Frame {
            device: self.device.as_ref(),
            queue: self.queue.as_ref(),
            drawable: drawable,
            width: w,
            height: h,
            time: self.started.elapsed().as_secs_f32(),
            dt,
        }
    }

    unsafe fn drawable_size(&self) -> (u32, u32) {
        let size = self.layer.drawableSize();
        ((size.width as u32).max(1), (size.height as u32).max(1))
    }
}

// Re-export commonly used items so samples can `use shared::*;`.
pub use metal_cpp_rs::metal::enums::*;
pub use metal_cpp_rs::objc_abi::{Ref, Strong};
