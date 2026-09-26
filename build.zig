/// Standard library import providing the build system API used throughout this script to declare modules, executables, run steps and tests.
const std = @import("std");

/// Alias to this build script type, used in `Options.getModule` with `dependencyFromBuildZig` to reference `src/root.zig` from a caller package graph.
const ThisBuild = @This();

/// Raw OBJ descriptor source module re-exported so downstream builds can reach parser internals without importing the file path directly.
pub const obj_descriptor = @import("src/obj_descriptor.zig");

/// Generic OBJ binary descriptor factory re-exported from `obj_descriptor` and wired into the `wavefront_obj` module created by `Options.getModule` and `createModuleOwn`.
pub const ObjBinaryDescriptor = obj_descriptor.ObjBinaryDescriptor;

/// Default `f32` OBJ descriptor instantiation used when downstream code needs OBJ asset support without choosing a custom scalar type.
pub const DefaultObjBinaryDescriptor = obj_descriptor.DefaultObjBinaryDescriptor;

/// Build configuration for the `wavefront_obj` package, consumed by `initFromOptions`, `getModule` and `createModuleOwn` to create a correctly wired module with a single shared `assets_manager` instance.
pub const Options = struct {
    /// Target platform for the compiled module, resolved from standard build options in `initFromOptions` and applied in `getModule` and `createModuleOwn`.
    target: ?std.Build.ResolvedTarget = null,
    /// Optimization mode for the compiled module, resolved from standard build options in `initFromOptions` and applied in `getModule` and `createModuleOwn`.
    optimize: ?std.builtin.OptimizeMode = null,
    /// Shared `assets_manager` module instance. When `null`, it is resolved
    /// via `b.dependency("assets_manager", ...)`. Pass an explicit module
    /// from the final project to guarantee a single `assets_manager`
    /// instance across packages and avoid "repeated import" conflicts.
    dependency_assets_manager: ?*std.Build.Module = null,

    /// Creates default build options from the flags of the current build invocation.
    /// - `b` - Build graph providing `standardTargetOptions` and `standardOptimizeOption`; the returned `dependency_assets_manager` stays null and is resolved later in `getModule` and `createModuleOwn`.
    ///
    /// Return: Initialized `Options` value used by `build` and by downstream `getModule` callers.
    pub fn initFromOptions(b: *std.Build) Options {
        return .{
            .target = b.standardTargetOptions(.{}),
            .optimize = b.standardOptimizeOption(.{}),
        };
    }

    /// Create the `wavefront_obj` module in the caller's build graph.
    ///
    /// ```zig
    /// const assets_mod = (@import("assets_manager").Options{
    ///     .target = target,
    ///     .optimize = optimize,
    /// }).getModule(b);
    /// const obj_mod = (@import("wavefront_obj").Options{
    ///     .target = target,
    ///     .optimize = optimize,
    ///     .dependency_assets_manager = assets_mod,
    /// }).getModule(b);
    /// ```
    /// - `self` - Configuration carrying target, optimize mode and an optional shared `assets_manager` module; when the shared module is null a private dependency is resolved internally.
    /// - `b` - Caller build graph that receives the new module with its `assets_manager` import attached.
    ///
    /// Return: New `wavefront_obj` module rooted at `src/root.zig`, ready to be added to executables or tests.
    pub fn getModule(self: Options, b: *std.Build) *std.Build.Module {
        const target = self.target orelse b.standardTargetOptions(.{});
        const optimize = self.optimize orelse b.standardOptimizeOption(.{});
        const self_dep = b.dependencyFromBuildZig(ThisBuild, .{
            .target = target,
            .optimize = optimize,
        });
        const assets_mod = self.dependency_assets_manager orelse self_dep.builder.dependency("assets_manager", .{
            .target = target,
            .optimize = optimize,
        }).module("assets_manager");
        const mod = b.createModule(.{
            .root_source_file = self_dep.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        });
        mod.addImport("assets_manager", assets_mod);
        return mod;
    }
};

/// Creates the `wavefront_obj` module inside the own package graph, mirroring `Options.getModule` but using local paths.
/// - `b` - Own package build graph that owns `src/root.zig` and registers the `wavefront_obj` module for the module tests.
/// - `options` - Configuration carrying target, optimize mode and an optional shared `assets_manager` module; falls back to `b.dependency` when null.
///
/// Return: Registered `wavefront_obj` module with the `assets_manager` import attached, reused by `build` for the module tests.
fn createModuleOwn(b: *std.Build, options: Options) *std.Build.Module {
    const target = options.target orelse b.standardTargetOptions(.{});
    const optimize = options.optimize orelse b.standardOptimizeOption(.{});
    const assets_mod = options.dependency_assets_manager orelse b.dependency("assets_manager", .{
        .target = target,
        .optimize = optimize,
    }).module("assets_manager");
    const mod = b.addModule("wavefront_obj", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("assets_manager", assets_mod);
    return mod;
}

/// Main package build entry point invoked by the Zig build system to produce the module test suite.
/// - `b` - Build graph used to create the `wavefront_obj` module via `createModuleOwn` plus the `test` step.
pub fn build(b: *std.Build) void {
    const options = Options.initFromOptions(b);

    const mod = createModuleOwn(b, options);

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
}
