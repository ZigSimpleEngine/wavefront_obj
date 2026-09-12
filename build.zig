const std = @import("std");

const ThisBuild = @This();

pub const obj_descriptor = @import("src/obj_descriptor.zig");
pub const ObjBinaryDescriptor = obj_descriptor.ObjBinaryDescriptor;
pub const DefaultObjBinaryDescriptor = obj_descriptor.DefaultObjBinaryDescriptor;

pub const Options = struct {
    /// The target architecture for which the module will be built.
    target: ?std.Build.ResolvedTarget = null,
    /// The optimization mode used to compile the module.
    optimize: ?std.builtin.OptimizeMode = null,
    /// Shared `assets_manager` module instance. When `null`, it is resolved
    /// via `b.dependency("assets_manager", ...)`. Pass an explicit module
    /// from the final project to guarantee a single `assets_manager`
    /// instance across packages and avoid "repeated import" conflicts.
    dependency_assets_manager: ?*std.Build.Module = null,

    pub fn initFromOptions(b: *std.Build) Options {
        return .{
            .target = b.standardTargetOptions(.{}),
            .optimize = b.standardOptimizeOption(.{}),
            // `dependency_*` cannot come from `-D` flags; they stay null
            // here and are resolved via `b.dependency` in the helpers below.
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

/// Create the `wavefront_obj` module in the *own* package graph.
/// Same wiring as `Options.getModule` but uses `b.path`/`b.dependency`.
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

pub fn build(b: *std.Build) void {
    const options = Options.initFromOptions(b);
    const target = options.target orelse b.standardTargetOptions(.{});
    const optimize = options.optimize orelse b.standardOptimizeOption(.{});

    const mod = createModuleOwn(b, options);

    const exe = b.addExecutable(.{
        .name = "wavefront_obj",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "wavefront_obj", .module = mod },
            },
        }),
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
