const std = @import("std");
const Node = @import("assets_manager").assets_tree.Node;
const text_utils = @import("assets_manager").text_utils;
const binary_descriptors = @import("assets_manager").binary_descriptors;
const BinaryDescriptor = binary_descriptors.BinaryDescriptor;
const MappingDescriptor = binary_descriptors.MappingDescriptor;

// Parsed object representation for internal use.
// An object splits into independent primitive sub-buffers (Mesh/Line/Point),
// each with its own deduplicated vertex set so points/lines never drag the
// full face attribute set with them.
const SubKind = enum { mesh, line, point };

const SubData = struct {
    kind: SubKind,
    vertices: std.ArrayList(Vertex),
    indices: std.ArrayList(u32),
    vertexMap: std.AutoHashMap(VertexKey, u32),
    hasUV: bool = false,
    hasNormal: bool = false,
    hasColor: bool = false,

    fn deinit(self: *SubData, gpa: std.mem.Allocator) void {
        self.vertices.deinit(gpa);
        self.indices.deinit(gpa);
        self.vertexMap.deinit();
    }
};

const ParsedObject = struct {
    name: []u8, // owned
    mesh: ?SubData = null,
    line: ?SubData = null,
    point: ?SubData = null,
};

const Vertex = struct {
    pos: [3]f64,
    uv: [2]f64,
    normal: [3]f64,
    color: [4]f64,
};

const VertexKey = struct {
    p: usize,
    vt: isize, // -1 means missing
    vn: isize,
};

// Helper to free ParsedObject
fn deinitParsedObject(obj: *ParsedObject, gpa: std.mem.Allocator) void {
    gpa.free(obj.name);
    if (obj.mesh) |*m| m.deinit(gpa);
    if (obj.line) |*l| l.deinit(gpa);
    if (obj.point) |*pt| pt.deinit(gpa);
}

fn newSubData(gpa: std.mem.Allocator, kind: SubKind) SubData {
    return .{
        .kind = kind,
        .vertices = .empty,
        .indices = .empty,
        .vertexMap = .init(gpa),
    };
}

fn readFileContent(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const size: usize = @intCast(stat.size);
    if (size == 0) return try gpa.alloc(u8, 0);
    const buf = try gpa.alloc(u8, size);
    errdefer gpa.free(buf);
    var total: usize = 0;
    while (total < size) {
        const n = try file.readStreaming(io, &.{buf[total..]});
        if (n == 0) break;
        total += n;
    }
    if (total != size) {
        const trimmed = try gpa.realloc(buf, total);
        return trimmed;
    }
    return buf;
}

fn parseFloatSafe(s: []const u8) !f64 {
    return std.fmt.parseFloat(f64, s);
}

fn parseIntSafe(s: []const u8) !i32 {
    return std.fmt.parseInt(i32, s, 10);
}

fn resolveIndex(raw: i32, count: usize) !usize {
    if (raw == 0) return error.InvalidIndex;
    if (raw > 0) {
        const idx = @as(usize, @intCast(raw - 1));
        if (idx >= count) return error.IndexOutOfBounds;
        return idx;
    } else {
        // negative: relative to end, -1 = last
        const idx_isize = @as(isize, @intCast(count)) + raw;
        if (idx_isize < 0) return error.IndexOutOfBounds;
        const idx = @as(usize, @intCast(idx_isize));
        if (idx >= count) return error.IndexOutOfBounds;
        return idx;
    }
}

// Parser-wide vertex data: positions, texcoords and normals live in the whole
// file (indexed by the face/line/point corner references), while color is
// derived from the same `v` line and indexed like position.
const FileData = struct {
    positions: std.ArrayList([3]f64),
    colors: std.ArrayList([4]f64),
    texcoords: std.ArrayList([2]f64),
    normals: std.ArrayList([3]f64),
    colors_present: bool = false,
};

fn cornerIdxOf(raw: ?i32, count: usize) !?usize {
    const rv = raw orelse return null;
    return try resolveIndex(rv, count);
}

fn buildVertex(data: *const FileData, p_idx: usize, vt_idx: ?usize, vn_idx: ?usize) Vertex {
    const pos = data.positions.items[p_idx];
    const color = data.colors.items[p_idx];
    var uv: [2]f64 = .{ 0, 0 };
    if (vt_idx) |v| uv = data.texcoords.items[v];
    var normal: [3]f64 = .{ 0, 0, 0 };
    if (vn_idx) |v| normal = data.normals.items[v];
    return .{ .pos = pos, .uv = uv, .normal = normal, .color = color };
}

fn isSameVector3(a: [3]f64, b: [3]f64) bool {
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2];
}

fn addVector3(a: [3]f64, b: [3]f64) [3]f64 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

fn cross(a: [3]f64, b: [3]f64) [3]f64 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn normalize3(v: [3]f64) [3]f64 {
    const len = std.math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len == 0) return .{ 0, 0, 0 };
    return .{ v[0] / len, v[1] / len, v[2] / len };
}

fn parseObjContent(gpa: std.mem.Allocator, content: []const u8, objects_out: *std.ArrayList(ParsedObject), data: *FileData) !void {
    var lines = std.mem.splitScalar(u8, content, '\n');
    var current_obj_idx: ?usize = null;
    // Current smoothing group for face normals when vn is absent (-1 = flat/off).
    var current_group: i32 = -1;
    // Track per-object vertex start indices so pure-v objects can be turned
    // into Point clouds using only their own slice of `data.positions`.
    var objVertexStarts: std.ArrayList(usize) = .empty;
    defer objVertexStarts.deinit(gpa);
    while (lines.next()) |raw_line| {
        var line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
        // handle BOM
        if (line.len >= 3 and line[0] == 0xEF and line[1] == 0xBB and line[2] == 0xBF) {
            line = line[3..];
            line = std.mem.trim(u8, line, " \t\r");
            if (line.len == 0) continue;
        }
        if (std.mem.startsWith(u8, line, "o ")) {
            const name_raw = std.mem.trim(u8, line[2..], " \t");
            const name = if (name_raw.len == 0) "default" else name_raw;
            const name_copy = try gpa.dupe(u8, name);
            try objects_out.append(gpa, .{ .name = name_copy });
            try objVertexStarts.append(gpa, data.positions.items.len);
            current_obj_idx = objects_out.items.len - 1;
        } else if (std.mem.startsWith(u8, line, "g ")) {
            if (current_obj_idx == null) {
                const name_raw = std.mem.trim(u8, line[2..], " \t");
                const name = if (name_raw.len == 0) "default" else name_raw;
                const name_copy = try gpa.dupe(u8, name);
                try objects_out.append(gpa, .{ .name = name_copy });
                try objVertexStarts.append(gpa, data.positions.items.len);
                current_obj_idx = objects_out.items.len - 1;
            }
        } else if (std.mem.startsWith(u8, line, "v ")) {
            const rest = std.mem.trim(u8, line[2..], " \t");
            var tok_it = std.mem.tokenizeAny(u8, rest, " \t");
            var vals: [8]f64 = undefined;
            var count: usize = 0;
            while (tok_it.next()) |tok| {
                if (count >= 8) break;
                vals[count] = try parseFloatSafe(tok);
                count += 1;
            }
            if (count < 3) continue;
            const v: [3]f64 = .{ vals[0], vals[1], vals[2] };
            try data.positions.append(gpa, v);
            // Optional colour: after xyz (an optional 4th "w" is skipped) the next
            // 3 (or 4) values are r g b [a]. When absent, colour is (0,0,0,0).
            // 6 values  -> "v x y z r g b"   (color at index 3)
            // 7 values  -> "v x y z w r g b" (color at index 4)
            var color: [4]f64 = .{ 0, 0, 0, 0 };
            if (count == 6) {
                color = .{ vals[3], vals[4], vals[5], 1.0 };
            } else if (count == 7) {
                color = .{ vals[4], vals[5], vals[6], 1.0 };
            } else if (count == 8) {
                color = .{ vals[4], vals[5], vals[6], vals[7] };
            }
            if (count == 6 or count == 7 or count == 8) data.colors_present = true;
            try data.colors.append(gpa, color);
        } else if (std.mem.startsWith(u8, line, "vt ")) {
            const rest = std.mem.trim(u8, line[3..], " \t");
            var tok_it = std.mem.tokenizeAny(u8, rest, " \t");
            var vals: [3]f64 = undefined;
            var count: usize = 0;
            while (tok_it.next()) |tok| {
                if (count >= 3) break;
                vals[count] = try parseFloatSafe(tok);
                count += 1;
            }
            if (count < 2) continue;
            const tc: [2]f64 = .{ vals[0], vals[1] };
            try data.texcoords.append(gpa, tc);
        } else if (std.mem.startsWith(u8, line, "vn ")) {
            const rest = std.mem.trim(u8, line[3..], " \t");
            var tok_it = std.mem.tokenizeAny(u8, rest, " \t");
            var vals: [3]f64 = undefined;
            var count: usize = 0;
            while (tok_it.next()) |tok| {
                if (count >= 3) break;
                vals[count] = try parseFloatSafe(tok);
                count += 1;
            }
            if (count < 3) continue;
            const n: [3]f64 = .{ vals[0], vals[1], vals[2] };
            try data.normals.append(gpa, n);
        } else if (std.mem.startsWith(u8, line, "s ")) {
            const rest = std.mem.trim(u8, line[2..], " \t");
            if (std.mem.eql(u8, rest, "off")) {
                current_group = -1;
            } else {
                current_group = std.fmt.parseInt(i32, rest, 10) catch -1;
            }
        } else if (std.mem.startsWith(u8, line, "f ")) {
            // ensure current object exists
            if (current_obj_idx == null) {
                const name_copy = try gpa.dupe(u8, "default");
                try objects_out.append(gpa, .{ .name = name_copy });
                try objVertexStarts.append(gpa, data.positions.items.len);
                current_obj_idx = objects_out.items.len - 1;
            }
            const obj = &objects_out.items[current_obj_idx.?];
            if (obj.mesh == null) obj.mesh = newSubData(gpa, .mesh);
            const sub = &obj.mesh.?;
            const rest = std.mem.trim(u8, line[2..], " \t");
            // collect corners for this face
            var corners: std.ArrayList(struct { p: usize, vt: ?usize, vn: ?usize }) = .empty;
            defer corners.deinit(gpa);
            var tok_it = std.mem.tokenizeAny(u8, rest, " \t");
            while (tok_it.next()) |tok| {
                var parts_it = std.mem.splitScalar(u8, tok, '/');
                var v_raw: ?i32 = null;
                var vt_raw: ?i32 = null;
                var vn_raw: ?i32 = null;
                var part_idx: usize = 0;
                while (parts_it.next()) |part| : (part_idx += 1) {
                    if (part_idx == 0) {
                        if (part.len > 0) v_raw = try parseIntSafe(part);
                    } else if (part_idx == 1) {
                        if (part.len > 0) vt_raw = try parseIntSafe(part);
                    } else if (part_idx == 2) {
                        if (part.len > 0) vn_raw = try parseIntSafe(part);
                    }
                }
                const v_val = v_raw orelse continue;
                const p_idx = try resolveIndex(v_val, data.positions.items.len);
                const vt_idx = try cornerIdxOf(vt_raw, data.texcoords.items.len);
                const vn_idx = try cornerIdxOf(vn_raw, data.normals.items.len);
                if (vt_idx != null) sub.hasUV = true;
                if (vn_idx != null) sub.hasNormal = true;
                try corners.append(gpa, .{ .p = p_idx, .vt = vt_idx, .vn = vn_idx });
            }
            if (corners.items.len < 3) continue;
            // triangulate fan from 0
            const v0 = corners.items[0];
            // record per-face normal (computed) so we can average for smooth groups
            var face_normal: [3]f64 = .{ 0, 0, 0 };
            if (data.colors_present) sub.hasColor = true;
            const need_computed_normal = !sub.hasNormal;
            if (need_computed_normal) {
                const a = data.positions.items[corners.items[0].p];
                const b = data.positions.items[corners.items[1].p];
                const cpos = data.positions.items[corners.items[2].p];
                const e1 = [_]f64{ b[0] - a[0], b[1] - a[1], b[2] - a[2] };
                const e2 = [_]f64{ cpos[0] - a[0], cpos[1] - a[1], cpos[2] - a[2] };
                face_normal = normalize3(cross(e1, e2));
            }
            for (1..corners.items.len - 1) |i| {
                const tri = [_]@TypeOf(v0){ v0, corners.items[i], corners.items[i + 1] };
                for (tri) |corner| {
                    const key = VertexKey{
                        .p = corner.p,
                        .vt = if (corner.vt) |idx| @as(isize, @intCast(idx)) else -1,
                        .vn = if (corner.vn) |idx| @as(isize, @intCast(idx)) else -1,
                    };
                    const maybe = sub.vertexMap.get(key);
                    var idx: u32 = undefined;
                    if (maybe) |existing| {
                        idx = existing;
                    } else {
                        const new_idx: u32 = @intCast(sub.vertices.items.len);
                        var vtx = buildVertex(data, corner.p, corner.vt, corner.vn);
                        if (need_computed_normal) {
                            // The average is accumulated below via pending normal list.
                            vtx.normal = face_normal;
                        }
                        try sub.vertices.append(gpa, vtx);
                        try sub.vertexMap.put(key, new_idx);
                        idx = new_idx;
                    }
                    try sub.indices.append(gpa, idx);
                }
            }
        } else if (std.mem.startsWith(u8, line, "l ")) {
            if (current_obj_idx == null) {
                const name_copy = try gpa.dupe(u8, "default");
                try objects_out.append(gpa, .{ .name = name_copy });
                try objVertexStarts.append(gpa, data.positions.items.len);
                current_obj_idx = objects_out.items.len - 1;
            }
            const obj = &objects_out.items[current_obj_idx.?];
            if (obj.line == null) obj.line = newSubData(gpa, .line);
            const sub = &obj.line.?;
            if (data.colors_present) sub.hasColor = true;
            const rest = std.mem.trim(u8, line[2..], " \t");
            var parts = try parseCornerLine(gpa, rest, data);
            defer parts.deinit(gpa);
            for (parts.items) |corner| {
                const key = VertexKey{
                    .p = corner.p,
                    .vt = if (corner.vt) |idx| @as(isize, @intCast(idx)) else -1,
                    .vn = if (corner.vn) |idx| @as(isize, @intCast(idx)) else -1,
                };
                const maybe = sub.vertexMap.get(key);
                var idx: u32 = undefined;
                if (maybe) |existing| {
                    idx = existing;
                } else {
                    const new_idx: u32 = @intCast(sub.vertices.items.len);
                    const vtx = buildVertex(data, corner.p, corner.vt, corner.vn);
                    try sub.vertices.append(gpa, vtx);
                    try sub.vertexMap.put(key, new_idx);
                    idx = new_idx;
                }
                try sub.indices.append(gpa, idx);
            }
        } else if (std.mem.startsWith(u8, line, "p ")) {
            if (current_obj_idx == null) {
                const name_copy = try gpa.dupe(u8, "default");
                try objects_out.append(gpa, .{ .name = name_copy });
                try objVertexStarts.append(gpa, data.positions.items.len);
                current_obj_idx = objects_out.items.len - 1;
            }
            const obj = &objects_out.items[current_obj_idx.?];
            if (obj.point == null) obj.point = newSubData(gpa, .point);
            const sub = &obj.point.?;
            if (data.colors_present) sub.hasColor = true;
            const rest = std.mem.trim(u8, line[2..], " \t");
            var parts = try parseCornerLine(gpa, rest, data);
            defer parts.deinit(gpa);
            for (parts.items) |corner| {
                const key = VertexKey{
                    .p = corner.p,
                    .vt = if (corner.vt) |idx| @as(isize, @intCast(idx)) else -1,
                    .vn = if (corner.vn) |idx| @as(isize, @intCast(idx)) else -1,
                };
                const maybe = sub.vertexMap.get(key);
                var idx: u32 = undefined;
                if (maybe) |existing| {
                    idx = existing;
                } else {
                    const new_idx: u32 = @intCast(sub.vertices.items.len);
                    const vtx = buildVertex(data, corner.p, corner.vt, corner.vn);
                    try sub.vertices.append(gpa, vtx);
                    try sub.vertexMap.put(key, new_idx);
                    idx = new_idx;
                }
                try sub.indices.append(gpa, idx);
            }
        }
    }
    // --- Fallback: .obj without any `f` (no indices) becomes Point cloud ---
    // Any object that stayed without mesh/line/point but has vertices assigned
    // to it is interpreted as a Point primitive: each `v` becomes a point.
    // This covers `src/models/points/sphere.obj` which contains only `v` lines.
    if (objects_out.items.len == 0 and data.positions.items.len > 0) {
        // No `o` at all -> create a single default point object covering all.
        const name_copy = try gpa.dupe(u8, "default");
        try objects_out.append(gpa, .{ .name = name_copy });
        try objVertexStarts.append(gpa, 0);
    }
    for (objects_out.items, 0..) |*obj, idx| {
        const has_mesh = obj.mesh != null and (obj.mesh.?.vertices.items.len > 0 or obj.mesh.?.indices.items.len > 0);
        const has_line = obj.line != null and (obj.line.?.vertices.items.len > 0 or obj.line.?.indices.items.len > 0);
        const has_point = obj.point != null and (obj.point.?.vertices.items.len > 0 or obj.point.?.indices.items.len > 0);
        if (has_mesh or has_line or has_point) continue;
        const start = if (idx < objVertexStarts.items.len) objVertexStarts.items[idx] else 0;
        const end = if (idx + 1 < objVertexStarts.items.len) objVertexStarts.items[idx + 1] else data.positions.items.len;
        var actualStart = start;
        var actualEnd = end;
        if (actualEnd <= actualStart) {
            // No slice (e.g. `o` after vertices, or empty file). For a single
            // object, fall back to the whole vertex buffer; otherwise skip.
            if (objects_out.items.len == 1 and data.positions.items.len > 0) {
                actualStart = 0;
                actualEnd = data.positions.items.len;
            } else {
                continue;
            }
        }
        if (actualEnd <= actualStart) continue;
        var pt = newSubData(gpa, .point);
        errdefer pt.deinit(gpa);
        if (data.colors_present) pt.hasColor = true;
        // Pure-v clouds have no UV/normal indices; keep hasUV/hasNormal false
        // so only positions (and optional colors) are packed.
        for (actualStart..actualEnd) |p_idx| {
            const key = VertexKey{ .p = p_idx, .vt = -1, .vn = -1 };
            const maybe = pt.vertexMap.get(key);
            var out_idx: u32 = undefined;
            if (maybe) |existing| {
                out_idx = existing;
            } else {
                const new_idx: u32 = @intCast(pt.vertices.items.len);
                const vtx = buildVertex(data, p_idx, null, null);
                try pt.vertices.append(gpa, vtx);
                try pt.vertexMap.put(key, new_idx);
                out_idx = new_idx;
            }
            try pt.indices.append(gpa, out_idx);
        }
        if (pt.vertices.items.len > 0) {
            obj.point = pt;
        } else {
            pt.deinit(gpa);
        }
    }
}

const Corner = struct { p: usize, vt: ?usize, vn: ?usize };

const CornerList = std.ArrayList(Corner);
fn parseCornerLine(gpa: std.mem.Allocator, rest: []const u8, data: *const FileData) !CornerList {
    var corners: CornerList = .empty;
    errdefer corners.deinit(gpa);
    var tok_it = std.mem.tokenizeAny(u8, rest, " \t");
    while (tok_it.next()) |tok| {
        var parts_it = std.mem.splitScalar(u8, tok, '/');
        var v_raw: ?i32 = null;
        var vt_raw: ?i32 = null;
        var vn_raw: ?i32 = null;
        var part_idx: usize = 0;
        while (parts_it.next()) |part| : (part_idx += 1) {
            if (part_idx == 0) {
                if (part.len > 0) v_raw = try parseIntSafe(part);
            } else if (part_idx == 1) {
                if (part.len > 0) vt_raw = try parseIntSafe(part);
            } else if (part_idx == 2) {
                if (part.len > 0) vn_raw = try parseIntSafe(part);
            }
        }
        const v_val = v_raw orelse continue;
        const p_idx = try resolveIndex(v_val, data.positions.items.len);
        const vt_idx = try cornerIdxOf(vt_raw, data.texcoords.items.len);
        const vn_idx = try cornerIdxOf(vn_raw, data.normals.items.len);
        try corners.append(gpa, .{ .p = p_idx, .vt = vt_idx, .vn = vn_idx });
    }
    return corners;
}

// Descriptor that handles .obj files: packs into compact binary
pub fn ObjBinaryDescriptor(comptime Scalar: type) type {
    return struct {
        const Self = @This();
        const ScalarType = Scalar;
        spaces_per_depth: usize = 4,

        pub fn isSuitableData(ptr: *anyopaque, init: std.process.Init, node: *Node) anyerror!bool {
            _ = ptr;
            _ = init;
            if (node.kind != .file) return false;
            if (node.name) |n| return std.mem.endsWith(u8, n, ".obj");
            return false;
        }

        pub fn getData(ptr: *anyopaque, init: std.process.Init, node_path: []const u8) anyerror![]u8 {
            const self: *Self = @ptrCast(@alignCast(ptr));
            _ = self;
            const gpa = init.gpa;
            const io = init.io;
            const content = try readFileContent(gpa, io, node_path);
            defer gpa.free(content);
            var objects: std.ArrayList(ParsedObject) = .empty;
            defer {
                for (objects.items) |*obj| deinitParsedObject(obj, gpa);
                objects.deinit(gpa);
            }
            var data = FileData{
                .positions = .empty,
                .colors = .empty,
                .texcoords = .empty,
                .normals = .empty,
            };
            defer {
                data.positions.deinit(gpa);
                data.colors.deinit(gpa);
                data.texcoords.deinit(gpa);
                data.normals.deinit(gpa);
            }
            try parseObjContent(gpa, content, &objects, &data);

            var bin = std.ArrayList(u8).empty;
            errdefer bin.deinit(gpa);

            for (objects.items) |*obj| {
                if (obj.mesh) |*m| try packSubData(gpa, m, &bin);
                if (obj.line) |*l| try packSubData(gpa, l, &bin);
                if (obj.point) |*pt| try packSubData(gpa, pt, &bin);
            }
            return bin.toOwnedSlice(gpa);
        }

        fn packSubData(gpa: std.mem.Allocator, sub: *const SubData, bin: *std.ArrayList(u8)) !void {
            if (sub.vertices.items.len == 0 and sub.indices.items.len == 0) return;
            const vertexCount = sub.vertices.items.len;
            // positions
            for (sub.vertices.items) |v| {
                const comps = [_]f64{ v.pos[0], v.pos[1], v.pos[2] };
                for (comps) |c| {
                    var val: Scalar = @floatCast(c);
                    try bin.appendSlice(gpa, std.mem.asBytes(&val));
                }
            }
            if (sub.hasColor) {
                for (sub.vertices.items) |v| {
                    const comps = [_]f64{ v.color[0], v.color[1], v.color[2], v.color[3] };
                    for (comps) |c| {
                        var val: Scalar = @floatCast(c);
                        try bin.appendSlice(gpa, std.mem.asBytes(&val));
                    }
                }
            }
            if (sub.hasUV) {
                for (sub.vertices.items) |v| {
                    const comps = [_]f64{ v.uv[0], v.uv[1] };
                    for (comps) |c| {
                        var val: Scalar = @floatCast(c);
                        try bin.appendSlice(gpa, std.mem.asBytes(&val));
                    }
                }
            }
            if (sub.hasNormal) {
                for (sub.vertices.items) |v| {
                    const comps = [_]f64{ v.normal[0], v.normal[1], v.normal[2] };
                    for (comps) |c| {
                        var val: Scalar = @floatCast(c);
                        try bin.appendSlice(gpa, std.mem.asBytes(&val));
                    }
                }
            }
            // pack indices as minimal type
            const vertexCountInt = vertexCount;
            if (vertexCountInt <= 256) {
                for (sub.indices.items) |idx| {
                    var v: u8 = @intCast(idx);
                    try bin.appendSlice(gpa, std.mem.asBytes(&v));
                }
            } else if (vertexCountInt <= 65536) {
                for (sub.indices.items) |idx| {
                    var v: u16 = @intCast(idx);
                    try bin.appendSlice(gpa, std.mem.asBytes(&v));
                }
            } else {
                for (sub.indices.items) |idx| {
                    var v: u32 = @intCast(idx);
                    try bin.appendSlice(gpa, std.mem.asBytes(&v));
                }
            }
        }

        pub fn deinitData(ptr: *anyopaque, init: std.process.Init, data: []u8) void {
            _ = ptr;
            init.gpa.free(data);
        }

        pub fn getMappingCode(ptr: *anyopaque, init: std.process.Init, data: MappingDescriptor.Data) anyerror![]u8 {
            const self: *Self = @ptrCast(@alignCast(ptr));
            const gpa = init.gpa;
            const io = init.io;
            const depth = data.depth;
            const node = data.node;

            const node_path_raw = try node.createPath(gpa);
            defer gpa.free(node_path_raw);
            const full_path = if (data.path_to_root_node.len == 0) try gpa.dupe(u8, node_path_raw) else try std.fs.path.join(gpa, &.{ data.path_to_root_node, node_path_raw });
            defer gpa.free(full_path);

            const content = try readFileContent(gpa, io, full_path);
            defer gpa.free(content);
            var objects: std.ArrayList(ParsedObject) = .empty;
            defer {
                for (objects.items) |*obj| deinitParsedObject(obj, gpa);
                objects.deinit(gpa);
            }
            var fdata = FileData{
                .positions = .empty,
                .colors = .empty,
                .texcoords = .empty,
                .normals = .empty,
            };
            defer {
                fdata.positions.deinit(gpa);
                fdata.colors.deinit(gpa);
                fdata.texcoords.deinit(gpa);
                fdata.normals.deinit(gpa);
            }
            try parseObjContent(gpa, content, &objects, &fdata);

            // Prepare prefixes
            const prefix = try text_utils.repeat(gpa, " ", depth * self.spaces_per_depth);
            defer if (prefix) |p| gpa.free(p);
            const obj_prefix = try text_utils.repeat(gpa, " ", (depth + 1) * self.spaces_per_depth);
            defer if (obj_prefix) |p| gpa.free(p);
            const sub_prefix = try text_utils.repeat(gpa, " ", (depth + 2) * self.spaces_per_depth);
            defer if (sub_prefix) |p| gpa.free(p);
            const attr_prefix = try text_utils.repeat(gpa, " ", (depth + 3) * self.spaces_per_depth);
            defer if (attr_prefix) |p| gpa.free(p);

            const file_name = node.name orelse return error.MissingName;
            const file_ident = try text_utils.filenameToIdentifier(gpa, file_name);
            defer gpa.free(file_ident);

            var escaped = std.ArrayList(u8).empty;
            defer escaped.deinit(gpa);
            for (data.bundle_path) |ch| {
                switch (ch) {
                    '\\' => try escaped.appendSlice(gpa, "\\\\"),
                    '"' => try escaped.appendSlice(gpa, "\\\""),
                    '\n' => try escaped.appendSlice(gpa, "\\n"),
                    '\r' => {},
                    else => try escaped.append(gpa, ch),
                }
            }

            const scalarName = @typeName(Scalar);
            const posType = try std.fmt.allocPrint(gpa, "@import(\"math\").Vec(3, {s})", .{scalarName});
            defer gpa.free(posType);
            const colorType = try std.fmt.allocPrint(gpa, "@import(\"math\").Vec(4, {s})", .{scalarName});
            defer gpa.free(colorType);
            const uvType = try std.fmt.allocPrint(gpa, "@import(\"math\").Vec(2, {s})", .{scalarName});
            defer gpa.free(uvType);
            const normalType = try std.fmt.allocPrint(gpa, "@import(\"math\").Vec(3, {s})", .{scalarName});
            defer gpa.free(normalType);

            var inner = std.ArrayList(u8).empty;
            defer inner.deinit(gpa);

            var seen = std.StringHashMap(usize).init(gpa);
            defer {
                var it = seen.iterator();
                while (it.next()) |e| gpa.free(e.key_ptr.*);
                seen.deinit();
            }

            var cumulative: usize = 0;
            var any_object = false;
            for (objects.items) |*obj| {
                const has_any = (obj.mesh != null and (obj.mesh.?.vertices.items.len > 0 or obj.mesh.?.indices.items.len > 0)) or
                    (obj.line != null and (obj.line.?.vertices.items.len > 0 or obj.line.?.indices.items.len > 0)) or
                    (obj.point != null and (obj.point.?.vertices.items.len > 0 or obj.point.?.indices.items.len > 0));
                if (!has_any) continue;
                any_object = true;

                const base_ident = try text_utils.filenameToIdentifier(gpa, obj.name);
                var final_ident: []u8 = undefined;
                if (seen.getPtr(base_ident)) |entry| {
                    final_ident = try std.fmt.allocPrint(gpa, "{s}_{d}", .{ base_ident, entry.* });
                    entry.* += 1;
                    gpa.free(base_ident);
                } else {
                    final_ident = try gpa.dupe(u8, base_ident);
                    const key_copy = try gpa.dupe(u8, base_ident);
                    try seen.put(key_copy, 1);
                    gpa.free(base_ident);
                }

                try inner.appendSlice(gpa, obj_prefix orelse "");
                try inner.appendSlice(gpa, "pub const ");
                try inner.appendSlice(gpa, final_ident);
                try inner.appendSlice(gpa, " = struct {\n");

                var any_emitted = false;

                if (obj.mesh) |*m| {
                    const sz = try emitSubStruct(gpa, m, &inner, sub_prefix, attr_prefix, &escaped, &cumulative, data.offset, posType, colorType, uvType, normalType, depth, self.spaces_per_depth);
                    if (sz) |_| any_emitted = true;
                }
                if (obj.line) |*l| {
                    const sz = try emitSubStruct(gpa, l, &inner, sub_prefix, attr_prefix, &escaped, &cumulative, data.offset, posType, colorType, uvType, normalType, depth, self.spaces_per_depth);
                    if (sz) |_| any_emitted = true;
                }
                if (obj.point) |*pt| {
                    const sz = try emitSubStruct(gpa, pt, &inner, sub_prefix, attr_prefix, &escaped, &cumulative, data.offset, posType, colorType, uvType, normalType, depth, self.spaces_per_depth);
                    if (sz) |_| any_emitted = true;
                }

                if (any_emitted) {
                    try inner.appendSlice(gpa, obj_prefix orelse "");
                    try inner.appendSlice(gpa, "};\n");
                }
                gpa.free(final_ident);
            }

            const inner_slice = try inner.toOwnedSlice(gpa);
            defer gpa.free(inner_slice);

            if (!any_object) {
                return std.fmt.allocPrint(gpa, "{s}pub const {s} = struct {{}};\n", .{ prefix orelse "", file_ident });
            }

            return std.fmt.allocPrint(gpa, "{s}pub const {s} = struct {{\n{s}{s}{s}}};\n", .{
                prefix orelse "",
                file_ident,
                inner_slice,
                if (inner_slice.len > 0 and inner_slice[inner_slice.len - 1] == '\n') "" else "\n",
                prefix orelse "",
            });
        }

        fn emitSubStruct(
            gpa: std.mem.Allocator,
            sub: *const SubData,
            inner: *std.ArrayList(u8),
            sub_prefix: ?[]const u8,
            attr_prefix: ?[]const u8,
            escaped: *std.ArrayList(u8),
            cumulative: *usize,
            baseOffset: usize,
            posType: []const u8,
            colorType: []const u8,
            uvType: []const u8,
            normalType: []const u8,
            depth: u32,
            spaces_per_depth: usize,
        ) !?usize {
            const vertexCount = sub.vertices.items.len;
            const indexCount = sub.indices.items.len;
            if (vertexCount == 0 and indexCount == 0) return null;

            const indexTypeStr = if (vertexCount <= 256) "u8" else if (vertexCount <= 65536) "u16" else "u32";
            const indexBytesPer = if (vertexCount <= 256) @as(usize, 1) else if (vertexCount <= 65536) @as(usize, 2) else @as(usize, 4);

            const posSize = vertexCount * 3 * @sizeOf(Scalar);
            const colorSize = if (sub.hasColor) vertexCount * 4 * @sizeOf(Scalar) else 0;
            const uvSize = if (sub.hasUV) vertexCount * 2 * @sizeOf(Scalar) else 0;
            const normalSize = if (sub.hasNormal) vertexCount * 3 * @sizeOf(Scalar) else 0;
            const indexSize = indexCount * indexBytesPer;

            const posOffset = baseOffset + cumulative.*;
            cumulative.* += posSize;
            const colorOffset = if (sub.hasColor) baseOffset + cumulative.* else 0;
            if (sub.hasColor) cumulative.* += colorSize;
            const uvOffset = if (sub.hasUV) baseOffset + cumulative.* else 0;
            if (sub.hasUV) cumulative.* += uvSize;
            const normalOffset = if (sub.hasNormal) baseOffset + cumulative.* else 0;
            if (sub.hasNormal) cumulative.* += normalSize;
            const indicesOffset = baseOffset + cumulative.*;
            cumulative.* += indexSize;

            const kind_name = switch (sub.kind) {
                .mesh => "Mesh",
                .line => "Line",
                .point => "Point",
            };

            try inner.appendSlice(gpa, sub_prefix orelse "");
            try inner.appendSlice(gpa, "pub const ");
            try inner.appendSlice(gpa, kind_name);
            try inner.appendSlice(gpa, " = struct {\n");

            {
                const line = try std.fmt.allocPrint(gpa, "{s}pub const position = Asset({s}, \"{s}\", {d}, {d});\n", .{ attr_prefix orelse "", posType, escaped.items, posOffset, posSize });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasColor) {
                const line = try std.fmt.allocPrint(gpa, "{s}pub const color = Asset({s}, \"{s}\", {d}, {d});\n", .{ attr_prefix orelse "", colorType, escaped.items, colorOffset, colorSize });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasUV) {
                const line = try std.fmt.allocPrint(gpa, "{s}pub const uv = Asset({s}, \"{s}\", {d}, {d});\n", .{ attr_prefix orelse "", uvType, escaped.items, uvOffset, uvSize });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasNormal) {
                const line = try std.fmt.allocPrint(gpa, "{s}pub const normal = Asset({s}, \"{s}\", {d}, {d});\n", .{ attr_prefix orelse "", normalType, escaped.items, normalOffset, normalSize });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            {
                const line = try std.fmt.allocPrint(gpa, "{s}pub const indices = Asset({s}, \"{s}\", {d}, {d});\n", .{ attr_prefix orelse "", indexTypeStr, escaped.items, indicesOffset, indexSize });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }

            // SOA nested struct mirroring parent layout but with slices
            {
                const soa_fun_prefix = try text_utils.repeat(gpa, " ", ((depth + 4) * spaces_per_depth));
                defer if (soa_fun_prefix) |p| gpa.free(p);
                const soa_return_prefix = try text_utils.repeat(gpa, " ", ((depth + 5) * spaces_per_depth));
                defer if (soa_return_prefix) |p| gpa.free(p);
                try inner.appendSlice(gpa, attr_prefix orelse "");
                try inner.appendSlice(gpa, "pub const SOA = struct {\n");
                {
                    const line = try std.fmt.allocPrint(gpa, "{s}position: []const {s},\n", .{ soa_fun_prefix orelse "", posType });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasColor) {
                    const line = try std.fmt.allocPrint(gpa, "{s}color: []const {s},\n", .{ soa_fun_prefix orelse "", colorType });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasUV) {
                    const line = try std.fmt.allocPrint(gpa, "{s}uv: []const {s},\n", .{ soa_fun_prefix orelse "", uvType });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasNormal) {
                    const line = try std.fmt.allocPrint(gpa, "{s}normal: []const {s},\n", .{ soa_fun_prefix orelse "", normalType });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                {
                    const line = try std.fmt.allocPrint(gpa, "{s}indices: []const {s},\n", .{ soa_fun_prefix orelse "", indexTypeStr });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                try inner.appendSlice(gpa, attr_prefix orelse "");
                try inner.appendSlice(gpa, "};\n");

                try inner.appendSlice(gpa, attr_prefix orelse "");
                try inner.appendSlice(gpa, "pub fn instanceSOA(allocator: std.mem.Allocator) !SOA {\n");
                try inner.appendSlice(gpa, soa_fun_prefix orelse "");
                try inner.appendSlice(gpa, "return .{\n");
                {
                    const line = try std.fmt.allocPrint(gpa, "{s}.position = try position.instance(allocator),\n", .{ soa_return_prefix orelse "" });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasColor) {
                    const line = try std.fmt.allocPrint(gpa, "{s}.color = try color.instance(allocator),\n", .{ soa_return_prefix orelse "" });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasUV) {
                    const line = try std.fmt.allocPrint(gpa, "{s}.uv = try uv.instance(allocator),\n", .{ soa_return_prefix orelse "" });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasNormal) {
                    const line = try std.fmt.allocPrint(gpa, "{s}.normal = try normal.instance(allocator),\n", .{ soa_return_prefix orelse "" });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                {
                    const line = try std.fmt.allocPrint(gpa, "{s}.indices = try indices.instance(allocator),\n", .{ soa_return_prefix orelse "" });
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                try inner.appendSlice(gpa, soa_fun_prefix orelse "");
                try inner.appendSlice(gpa, "};\n");
                try inner.appendSlice(gpa, attr_prefix orelse "");
                try inner.appendSlice(gpa, "}\n");
            }

            // unloadAll
            try inner.appendSlice(gpa, attr_prefix orelse "");
            try inner.appendSlice(gpa, "pub fn unloadAll(allocator: std.mem.Allocator) void {\n");
            const fun_prefix = try text_utils.repeat(gpa, " ", ((depth + 4) * spaces_per_depth));
            defer if (fun_prefix) |p| gpa.free(p);
            {
                const line = try std.fmt.allocPrint(gpa, "{s}position.unload(allocator);\n", .{ fun_prefix orelse "" });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasColor) {
                const line = try std.fmt.allocPrint(gpa, "{s}color.unload(allocator);\n", .{ fun_prefix orelse "" });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasUV) {
                const line = try std.fmt.allocPrint(gpa, "{s}uv.unload(allocator);\n", .{ fun_prefix orelse "" });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasNormal) {
                const line = try std.fmt.allocPrint(gpa, "{s}normal.unload(allocator);\n", .{ fun_prefix orelse "" });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            {
                const line = try std.fmt.allocPrint(gpa, "{s}indices.unload(allocator);\n", .{ fun_prefix orelse "" });
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            try inner.appendSlice(gpa, attr_prefix orelse "");
            try inner.appendSlice(gpa, "}\n");

            try inner.appendSlice(gpa, sub_prefix orelse "");
            try inner.appendSlice(gpa, "};\n");
            return posSize;
        }

        pub fn mapping(self: *Self) MappingDescriptor {
            return .{
                .ptr = self,
                .vtable = .{ .get_code = getMappingCode },
            };
        }

        pub fn descriptor(self: *Self) BinaryDescriptor {
            return .{
                .ptr = self,
                .mapping = self.mapping(),
                .vtable = .{
                    .get_data = getData,
                    .is_suitable_data = isSuitableData,
                    .deinit_data = deinitData,
                },
            };
        }
    };
}

// Also provide non-generic wrapper for convenience with default f32
pub const DefaultObjBinaryDescriptor = ObjBinaryDescriptor(f32);

test "parse obj: color, face vertex indices and line split" {
    const gpa = std.testing.allocator;
    const content =
        \\o TriColored
        \\v 0 0 0 1 0 0
        \\v 1 0 0 0 1 0
        \\v 0 1 0 0 0 1
        \\f 1 2 3
        \\o Empty
        \\l 1 2 3
    ;
    var objects: std.ArrayList(ParsedObject) = .empty;
    defer {
        for (objects.items) |*obj| deinitParsedObject(obj, gpa);
        objects.deinit(gpa);
    }
    var data = FileData{
        .positions = .empty,
        .colors = .empty,
        .texcoords = .empty,
        .normals = .empty,
    };
    defer {
        data.positions.deinit(gpa);
        data.colors.deinit(gpa);
        data.texcoords.deinit(gpa);
        data.normals.deinit(gpa);
    }
    try parseObjContent(gpa, content, &objects, &data);

    try std.testing.expectEqual(@as(usize, 2), objects.items.len);

    // First object: mesh with colors.
    const tri = &objects.items[0];
    try std.testing.expect(tri.mesh != null);
    try std.testing.expect(tri.line == null);
    const m = &tri.mesh.?;
    try std.testing.expectEqual(@as(usize, 3), m.vertices.items.len);
    try std.testing.expectEqual(@as(usize, 3), m.indices.items.len);
    try std.testing.expect(m.hasColor);
    // Red first vertex
    try std.testing.expectEqual(@as(f64, 1), m.vertices.items[0].color[0]);
    try std.testing.expectEqual(@as(f64, 0), m.vertices.items[0].color[1]);

    // Second object: line primitive only.
    const line = &objects.items[1];
    try std.testing.expect(line.line != null);
    try std.testing.expect(line.mesh == null);
    const l = &line.line.?;
    try std.testing.expectEqual(@as(usize, 3), l.vertices.items.len);
    try std.testing.expectEqual(@as(usize, 3), l.indices.items.len);
    try std.testing.expectEqual(@as(f64, 0), l.vertices.items[0].pos[0]);
    // Colors are global to the file, so the line sub-buffer carries them too.
    try std.testing.expect(l.hasColor);
    try std.testing.expectEqual(@as(f64, 1), l.vertices.items[0].color[0]);
}
