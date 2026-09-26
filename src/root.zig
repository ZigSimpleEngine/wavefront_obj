//! By convention, root.zig is the root source file when making a package.
/// Standard library import used for testing utilities and for the `Io.Writer` type referenced by `printAnotherMessage`.
const std = @import("std");
/// Alias to the new Zig IO interface type, used as the output channel type in `printAnotherMessage`.
const Io = std.Io;

/// OBJ asset descriptor module re-exported as the implementation backing `ObjBinaryDescriptor` and `DefaultObjBinaryDescriptor` for `assets_manager` integration.
pub const obj_descriptor = @import("obj_descriptor.zig");
/// Generic Wavefront OBJ binary descriptor factory that packs `.obj` files into a compact bundle and generates typed Zig accessors; re-exported from `obj_descriptor` for downstream `assets_manager` registration.
pub const ObjBinaryDescriptor = obj_descriptor.ObjBinaryDescriptor;
/// Default `f32` OBJ binary descriptor instantiation for consumers that do not need a custom scalar type; re-exported from `obj_descriptor`.
pub const DefaultObjBinaryDescriptor = obj_descriptor.DefaultObjBinaryDescriptor;

/// Prints a hint about running the package tests to the given writer, used as a small IO usage example for downstream consumers.
/// - `writer` - Destination `Io.Writer` receiving the hint text; flushed by the caller after this call.
pub fn printAnotherMessage(writer: *Io.Writer) Io.Writer.Error!void {
    try writer.print("Run `zig build test` to run the tests.\n", .{});
}

/// Adds two integers, exercised by the `basic add functionality` test to verify the test harness.
/// - `a` - First addend.
/// - `b` - Second addend.
///
/// Return: Arithmetic sum of `a` and `b`.
pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}

test {
    std.testing.refAllDecls(@This());
}
