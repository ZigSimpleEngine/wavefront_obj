/// Standard library import providing allocation, IO, math, formatting and container types used across the parser, binary packer and mapping generator.
const std = @import("std");
/// Asset tree node type from `assets_manager`, used in `isSuitableData` to filter `.obj` files and in `getMappingCode` to resolve file paths and generated identifiers.
const Node = @import("assets_manager").assets_tree.Node;
/// Text helpers from `assets_manager`, used in `getMappingCode` and `emitSubStruct` for indentation repeats and for converting file and object names into valid Zig identifiers.
const text_utils = @import("assets_manager").text_utils;
/// Binary descriptor definitions from `assets_manager`, source of the `BinaryDescriptor` and `MappingDescriptor` types implemented by the struct returned from `ObjBinaryDescriptor`.
const binary_descriptors = @import("assets_manager").binary_descriptors;
/// Binary bundling interface type implemented by `ObjBinaryDescriptor.descriptor`; drives `getData` packing and `deinitData` cleanup during asset builds.
const BinaryDescriptor = binary_descriptors.BinaryDescriptor;
/// Mapping interface type implemented by `ObjBinaryDescriptor.mapping`; drives `getMappingCode` generation of typed `Asset` accessors consumed by downstream Zig code.
const MappingDescriptor = binary_descriptors.MappingDescriptor;

/// Primitive class of a per-object sub-buffer, keeps mesh, line and point data in independent deduplicated buffers so points and lines never inherit face attributes.
const SubKind = enum {
    /// Triangle mesh sub-buffer fed by `f` face records, triangulated as a fan in `parseObjContent` and packed and emitted as `Mesh` by `packSubData` and `emitSubStruct`.
    mesh,
    /// Polyline sub-buffer fed by `l` records through `parseCornerLine`, packed and emitted as `Line` by `packSubData` and `emitSubStruct`.
    line,
    /// Point cloud sub-buffer fed by `p` records plus the pure `v` fallback in `parseObjContent`, packed and emitted as `Point` by `packSubData` and `emitSubStruct`.
    point,
};

/// Deduplicated primitive buffer for one `SubKind` inside a `ParsedObject`, accumulated in `parseObjContent` and consumed by `packSubData` and `emitSubStruct`.
const SubData = struct {
    /// Primitive class of this buffer, selects the generated struct name in `emitSubStruct` and is set once in `newSubData`.
    kind: SubKind,
    /// Deduplicated vertex list keyed by `vertexMap`, appended in `parseObjContent` face, line and point branches and serialized attribute by attribute in `packSubData`.
    vertices: std.ArrayList(Vertex),
    /// Index list referencing `vertices`, appended alongside vertex deduplication and packed with the smallest fitting integer width in `packSubData`.
    indices: std.ArrayList(u32),
    /// Deduplication map from `VertexKey` to vertex index, consulted on every corner in `parseObjContent` to reuse identical position and attribute combinations.
    vertexMap: std.AutoHashMap(VertexKey, u32),
    /// Presence flag for texture coordinates, set when any corner resolves a `vt` index and controls whether UV blocks are packed and emitted.
    hasUV: bool = false,
    /// Presence flag for normals, set when any corner resolves a `vn` index; when false `parseObjContent` substitutes a computed face normal.
    hasNormal: bool = false,
    /// Presence flag for vertex colors, set when file-wide `colors_present` is true and controls whether color blocks are packed and emitted.
    hasColor: bool = false,

    /// Releases vertex, index and map storage owned by this sub-buffer, called from `deinitParsedObject` and from the point-cloud fallback error path.
    /// - `self` - Sub-buffer to deinitialize; its lists and map become unusable after this call.
    /// - `gpa` - Allocator that owns `vertices`, `indices` and the internal `vertexMap` storage.
    fn deinit(self: *SubData, gpa: std.mem.Allocator) void {
        self.vertices.deinit(gpa);
        self.indices.deinit(gpa);
        self.vertexMap.deinit();
    }
};

/// Single named object assembled from `o` and `g` directives, owns up to three optional `SubData` buffers later packed in mesh, line, point order by `getData`.
const ParsedObject = struct {
    /// Owned object name duplicated from the `o` or `g` line, used as the basis for the generated Zig constant in `getMappingCode` and freed in `deinitParsedObject`.
    name: []u8,
    /// Optional mesh buffer built from `f` faces, created on demand in `parseObjContent` and packed first in `getData`.
    mesh: ?SubData = null,
    /// Optional line buffer built from `l` records, created on demand in `parseObjContent` and packed after the mesh in `getData`.
    line: ?SubData = null,
    /// Optional point buffer built from `p` records or the pure `v` fallback, created on demand in `parseObjContent` and packed last in `getData`.
    point: ?SubData = null,
};

/// Fully resolved intermediate vertex in `f64` precision, built by `buildVertex` from global `FileData` tables and stored in `SubData.vertices` before scalar conversion in `packSubData`.
const Vertex = struct {
    /// Position sampled from `FileData.positions`, always present and serialized first in `packSubData`.
    pos: [3]f64,
    /// Texture coordinate sampled from `FileData.texcoords` or zero when the corner has no `vt` index; serialized only when the owning buffer sets `hasUV`.
    uv: [2]f64,
    /// Normal sampled from `FileData.normals`, substituted with a computed face normal or zero when missing; serialized only when the owning buffer sets `hasNormal`.
    normal: [3]f64,
    /// Color sampled from `FileData.colors` aligned with the position index, zero with zero alpha when the `v` line carries no color; serialized only when the owning buffer sets `hasColor`.
    color: [4]f64,
};

/// Hashable corner identity used as `SubData.vertexMap` key to deduplicate vertices sharing the same position, texcoord and normal indices.
const VertexKey = struct {
    /// Resolved position index into `FileData.positions`, always present and derived via `resolveIndex` in face, line and point parsing.
    p: usize,
    /// Resolved texcoord index as a signed value with `-1` for missing, produced from optional `vt` corners and compared during deduplication.
    vt: isize,
    /// Resolved normal index as a signed value with `-1` for missing, produced from optional `vn` corners and compared during deduplication.
    vn: isize,
};

/// Frees an object name and all owned sub-buffers, used to clean the `objects` list in `getData`, `getMappingCode` and the unit test.
/// - `obj` - Parsed object whose `name` and optional `mesh`, `line` and `point` buffers are released.
/// - `gpa` - Allocator that owns the name copy and all sub-buffer storage.
fn deinitParsedObject(obj: *ParsedObject, gpa: std.mem.Allocator) void {
    gpa.free(obj.name);
    if (obj.mesh) |*m| m.deinit(gpa);
    if (obj.line) |*l| l.deinit(gpa);
    if (obj.point) |*pt| pt.deinit(gpa);
}

/// Creates an empty sub-buffer of the given kind, used whenever `parseObjContent` first encounters an `f`, `l` or `p` record for an object and in the point-cloud fallback.
/// - `gpa` - Allocator used to initialize the internal `vertexMap`; vertex and index lists start empty.
/// - `kind` - Primitive class stored in the new `SubData.kind` field and later used to name the generated struct.
///
/// Return: Initialized empty `SubData` ready to receive deduplicated vertices and indices.
fn newSubData(gpa: std.mem.Allocator, kind: SubKind) SubData {
    return .{
        .kind = kind,
        .vertices = .empty,
        .indices = .empty,
        .vertexMap = .init(gpa),
    };
}

/// Loads a complete file into memory for `getData` and `getMappingCode` before OBJ parsing.
/// - `gpa` - Allocator used for the returned content buffer.
/// - `io` - IO context used to open the current directory file and stream its bytes.
/// - `path` - File system path resolved by the caller from an asset node path.
///
/// Return: Owned file bytes, possibly truncated on short reads and empty for zero-size files.
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

/// Parses a decimal token from `v`, `vt` and `vn` lines in `parseObjContent`.
/// - `s` - Single whitespace-separated number token.
///
/// Return: Parsed `f64` value or a float parse error.
fn parseFloatSafe(s: []const u8) !f64 {
    return std.fmt.parseFloat(f64, s);
}

/// Parses a decimal index token from `f`, `l` and `p` corner references in `parseObjContent` and `parseCornerLine`.
/// - `s` - Single index token without the slash separators.
///
/// Return: Parsed `i32` index, still in OBJ 1-based or negative-relative form before `resolveIndex`.
fn parseIntSafe(s: []const u8) !i32 {
    return std.fmt.parseInt(i32, s, 10);
}

/// Converts OBJ 1-based and negative relative indices to 0-based storage indices with bounds checking, used for every position, texcoord and normal reference.
/// - `raw` - Raw OBJ index where positive values are 1-based, negative values count back from the end, and zero is invalid.
/// - `count` - Current element count of the referenced table used for bounds checks and negative resolution.
///
/// Return: Validated 0-based index into the referenced table.
fn resolveIndex(raw: i32, count: usize) !usize {
    if (raw == 0) return error.InvalidIndex;
    if (raw > 0) {
        const idx = @as(usize, @intCast(raw - 1));
        if (idx >= count) return error.IndexOutOfBounds;
        return idx;
    } else {
        const idx_isize = @as(isize, @intCast(count)) + raw;
        if (idx_isize < 0) return error.IndexOutOfBounds;
        const idx = @as(usize, @intCast(idx_isize));
        if (idx >= count) return error.IndexOutOfBounds;
        return idx;
    }
}

/// File-wide vertex tables shared by all objects, filled from `v`, `vt` and `vn` lines in `parseObjContent` and read by `buildVertex`.
const FileData = struct {
    /// All positions in file order, indexed by resolved `p` corner values and parallel to `colors`.
    positions: std.ArrayList([3]f64),
    /// Per-position colors parsed from extended `v` lines, parallel to `positions` and sampled together in `buildVertex`.
    colors: std.ArrayList([4]f64),
    /// All texture coordinates in file order, indexed by resolved `vt` corner values.
    texcoords: std.ArrayList([2]f64),
    /// All normals in file order, indexed by resolved `vn` corner values.
    normals: std.ArrayList([3]f64),
    /// Global color presence latch set when any `v` line carries color values; copied into each active `SubData.hasColor` flag.
    colors_present: bool = false,
};

/// Resolves an optional `vt` or `vn` raw index via `resolveIndex`, preserving absence as null for `buildVertex` defaults.
/// - `raw` - Optional raw OBJ index, null when the corner omits the corresponding slash-separated part.
/// - `count` - Current element count of the referenced texcoord or normal table.
///
/// Return: Resolved 0-based index or null when the corner has no such component.
fn cornerIdxOf(raw: ?i32, count: usize) !?usize {
    const rv = raw orelse return null;
    return try resolveIndex(rv, count);
}

/// Assembles a `Vertex` from global tables, used for every deduplicated face, line and point corner including the pure `v` fallback.
/// - `data` - File-wide tables supplying position, color, texcoord and normal values.
/// - `p_idx` - Resolved position index selecting both `positions` and parallel `colors` entries.
/// - `vt_idx` - Optional resolved texcoord index, null selects a zero UV.
/// - `vn_idx` - Optional resolved normal index, null selects a zero normal later possibly replaced by a computed face normal.
///
/// Return: Fully populated intermediate vertex in `f64` precision.
fn buildVertex(data: *const FileData, p_idx: usize, vt_idx: ?usize, vn_idx: ?usize) Vertex {
    const pos = data.positions.items[p_idx];
    const color = data.colors.items[p_idx];
    var uv: [2]f64 = .{ 0, 0 };
    if (vt_idx) |v| uv = data.texcoords.items[v];
    var normal: [3]f64 = .{ 0, 0, 0 };
    if (vn_idx) |v| normal = data.normals.items[v];
    return .{ .pos = pos, .uv = uv, .normal = normal, .color = color };
}

/// Compares two 3-component vectors for exact equality, currently unused and reserved for future position or normal comparison helpers.
/// - `a` - First vector.
/// - `b` - Second vector.
///
/// Return: True when all three components are exactly equal.
fn isSameVector3(a: [3]f64, b: [3]f64) bool {
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2];
}

/// Adds two 3-component vectors component-wise, currently unused and reserved for future normal averaging or position math.
/// - `a` - First addend.
/// - `b` - Second addend.
///
/// Return: Component-wise sum of the two input vectors.
fn addVector3(a: [3]f64, b: [3]f64) [3]f64 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

/// Computes the cross product of two edge vectors, used in `parseObjContent` together with `normalize3` to derive a face normal when `vn` data is absent.
/// - `a` - First edge vector.
/// - `b` - Second edge vector.
///
/// Return: Cross product vector perpendicular to both inputs.
fn cross(a: [3]f64, b: [3]f64) [3]f64 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

/// Normalizes a 3-component vector, returning zero on zero length; used in `parseObjContent` with `cross` to produce unit face normals for buffers without `hasNormal`.
/// - `v` - Input vector, typically the cross product of two face edges.
///
/// Return: Unit-length vector in the same direction, or zero when the input has zero length.
fn normalize3(v: [3]f64) [3]f64 {
    const len = std.math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len == 0) return .{ 0, 0, 0 };
    return .{ v[0] / len, v[1] / len, v[2] / len };
}

/// Parses complete Wavefront OBJ text into per-object primitive buffers and global vertex tables, handling `o`, `g`, `v`, `vt`, `vn`, `s`, `f`, `l` and `p` lines plus BOM removal, comment skipping and pure `v` point-cloud fallback. Consumed by `getData`, `getMappingCode` and the unit test.
/// - `gpa` - Allocator used for all parsed lists, duplicated object names and fallback point clouds.
/// - `content` - Complete OBJ file text previously loaded by `readFileContent` and split line by line.
/// - `objects_out` - Output list receiving one `ParsedObject` per `o` or `g` block with deduplicated mesh, line and point buffers.
/// - `data` - Global `FileData` tables filled from `v`, `vt` and `vn` lines alongside `objects_out`.
fn parseObjContent(gpa: std.mem.Allocator, content: []const u8, objects_out: *std.ArrayList(ParsedObject), data: *FileData) !void {
    var lines = std.mem.splitScalar(u8, content, '\n');
    var current_obj_idx: ?usize = null;
    var current_group: i32 = -1;
    var objVertexStarts: std.ArrayList(usize) = .empty;
    defer objVertexStarts.deinit(gpa);
    while (lines.next()) |raw_line| {
        var line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
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
            const v0 = corners.items[0];
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
    if (objects_out.items.len == 0 and data.positions.items.len > 0) {
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

/// Resolved `l` and `p` corner produced by `parseCornerLine` and consumed in the line and point branches of `parseObjContent`.
const Corner = struct {
    /// Resolved position index into `FileData.positions`, validated by `resolveIndex` against the current position count.
    p: usize,
    /// Optional resolved texcoord index into `FileData.texcoords`, null when the token omits the `vt` part.
    vt: ?usize,
    /// Optional resolved normal index into `FileData.normals`, null when the token omits the `vn` part.
    vn: ?usize,
};

/// Growable list of `Corner` values returned by `parseCornerLine` for one `l` or `p` line and iterated by the line and point branches of `parseObjContent`.
const CornerList = std.ArrayList(Corner);

/// Parses whitespace-separated `v` and optional `vt` and `vn` tokens of one `l` or `p` line, shared by both line and point branches to avoid duplicated corner logic.
/// - `gpa` - Allocator used for the returned corner list.
/// - `rest` - Trimmed remainder of the `l` or `p` line after its prefix, containing slash-separated corner tokens.
/// - `data` - File-wide tables used to validate every resolved `p`, `vt` and `vn` index.
///
/// Return: Owned corner list with all valid entries; invalid empty `v` tokens are skipped.
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

/// Generic factory creating the `assets_manager` binary and mapping implementation for Wavefront OBJ files, instantiated as `DefaultObjBinaryDescriptor` for `f32` and re-exported through `root.zig` and `build.zig`.
/// - `Scalar` - Floating-point type used for packed positions, colors, UVs and normals; determines `@sizeOf(Scalar)` block sizes and `@floatCast` conversions in `packSubData` and `emitSubStruct`.
///
/// Return: Descriptor struct type implementing `BinaryDescriptor` and `MappingDescriptor` for `.obj` assets.
pub fn ObjBinaryDescriptor(comptime Scalar: type) type {
    return struct {
        /// Alias to the concrete instantiated descriptor struct, used to cast the opaque `ptr` in `getData`, `getMappingCode`, `mapping` and `descriptor`.
        const Self = @This();
        /// Preserved scalar type alias documenting the configured float precision; the implementation itself uses `Scalar` directly for packing and type name generation.
        const ScalarType = Scalar;
        /// Spaces per nesting depth for generated Zig code, multiplied by `depth` in `getMappingCode` and forwarded to `emitSubStruct` for aligned output.
        spaces_per_depth: usize = 4,

        /// Reports whether an asset node holds Wavefront OBJ data, called by `assets_manager` during descriptor selection before `getData` or `getMappingCode`.
        /// - `ptr` - Opaque descriptor instance pointer, unused because suitability depends only on the node.
        /// - `init` - Process context, unused here but kept for the shared descriptor vtable signature.
        /// - `node` - Asset tree node to inspect; only file nodes ending in `.obj` are accepted.
        ///
        /// Return: True for `.obj` file nodes, false otherwise.
        pub fn isSuitableData(ptr: *anyopaque, init: std.process.Init, node: *Node) anyerror!bool {
            _ = ptr;
            _ = init;
            if (node.kind != .file) return false;
            if (node.name) |n| return std.mem.endsWith(u8, n, ".obj");
            return false;
        }

        /// Packs one `.obj` file into a compact binary blob in mesh, line, point order, called by `assets_manager` when building the asset bundle.
        /// - `ptr` - Opaque descriptor instance pointer, currently unused because packing depends only on file content and `Scalar`.
        /// - `init` - Process context supplying the allocator and IO used by `readFileContent` and the parser.
        /// - `node_path` - File system path of the `.obj` file to load, parse with `parseObjContent` and serialize with `packSubData`.
        ///
        /// Return: Owned binary bytes with per-vertex `Scalar` attributes followed by minimal-width indices.
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

        /// Appends one sub-buffer attribute blocks and minimal-width indices to the binary bundle in the exact layout mirrored by `emitSubStruct` offsets, called from `getData` for mesh, line and point buffers.
        /// - `gpa` - Allocator used to grow the output byte list.
        /// - `sub` - Source deduplicated buffer whose positions, optional colors, UVs, normals and indices are serialized.
        /// - `bin` - Destination byte buffer receiving raw `Scalar` attribute bytes followed by `u8`, `u16` or `u32` indices.
        fn packSubData(gpa: std.mem.Allocator, sub: *const SubData, bin: *std.ArrayList(u8)) !void {
            if (sub.vertices.items.len == 0 and sub.indices.items.len == 0) return;
            const vertexCount = sub.vertices.items.len;
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

        /// Releases a binary blob previously returned by `getData`, called by `assets_manager` after the blob has been copied into the bundle.
        /// - `ptr` - Opaque descriptor instance pointer, unused because deallocation needs only the allocator.
        /// - `init` - Process context supplying the allocator that owns `data`.
        /// - `data` - Binary bytes to free.
        pub fn deinitData(ptr: *anyopaque, init: std.process.Init, data: []u8) void {
            _ = ptr;
            init.gpa.free(data);
        }

        /// Generates typed Zig accessor code for one `.obj` file with per-object `Mesh`, `Line` and `Point` structs plus `SOA`, `instanceSOA` and `unloadAll`, called by `assets_manager` through the mapping vtable.
        /// - `ptr` - Opaque descriptor instance pointer providing `spaces_per_depth` for indentation.
        /// - `init` - Process context supplying the allocator and IO used to reload and reparse the file.
        /// - `data` - Mapping inputs carrying the asset node, bundle-relative paths, depth, base offset and previously escaped bundle path context.
        ///
        /// Return: Owned generated Zig source for the file constant, or an empty struct when no drawable primitive exists.
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

        /// Emits one `Mesh`, `Line` or `Point` accessor struct with `Asset` constants plus nested `SOA`, `instanceSOA` and `unloadAll`, advancing the shared bundle offset cursor exactly as `packSubData` lays out bytes. Called from `getMappingCode` for every non-empty sub-buffer.
        /// - `gpa` - Allocator used for temporary indentation strings and formatted `Asset` lines appended to `inner`.
        /// - `sub` - Source sub-buffer providing vertex counts, attribute presence flags and primitive kind for naming.
        /// - `inner` - Destination source buffer receiving the generated struct text.
        /// - `sub_prefix` - Indentation for the `pub const Mesh` style struct header.
        /// - `attr_prefix` - Indentation for attribute `Asset` constants and helper definitions inside the struct.
        /// - `escaped` - Escaped bundle path text reused verbatim in every generated `Asset` path argument.
        /// - `cumulative` - Running byte cursor advanced by each attribute and index block; combined with `baseOffset` to compute absolute offsets.
        /// - `baseOffset` - Bundle base offset of this `.obj` file supplied in `MappingDescriptor.Data.offset`.
        /// - `posType` - Formatted position vector type name derived from `Scalar`.
        /// - `colorType` - Formatted color vector type name derived from `Scalar`.
        /// - `uvType` - Formatted UV vector type name derived from `Scalar`.
        /// - `normalType` - Formatted normal vector type name derived from `Scalar`.
        /// - `depth` - Nesting depth used to derive `SOA` helper indentation.
        /// - `spaces_per_depth` - Spaces per depth level copied from the descriptor instance for consistent formatting.
        ///
        /// Return: Position block size when a struct was emitted, null when the sub-buffer is empty and skipped.
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
                    const line = try std.fmt.allocPrint(gpa, "{s}.position = try position.instance(allocator),\n", .{soa_return_prefix orelse ""});
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasColor) {
                    const line = try std.fmt.allocPrint(gpa, "{s}.color = try color.instance(allocator),\n", .{soa_return_prefix orelse ""});
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasUV) {
                    const line = try std.fmt.allocPrint(gpa, "{s}.uv = try uv.instance(allocator),\n", .{soa_return_prefix orelse ""});
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                if (sub.hasNormal) {
                    const line = try std.fmt.allocPrint(gpa, "{s}.normal = try normal.instance(allocator),\n", .{soa_return_prefix orelse ""});
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                {
                    const line = try std.fmt.allocPrint(gpa, "{s}.indices = try indices.instance(allocator),\n", .{soa_return_prefix orelse ""});
                    defer gpa.free(line);
                    try inner.appendSlice(gpa, line);
                }
                try inner.appendSlice(gpa, soa_fun_prefix orelse "");
                try inner.appendSlice(gpa, "};\n");
                try inner.appendSlice(gpa, attr_prefix orelse "");
                try inner.appendSlice(gpa, "}\n");
            }

            try inner.appendSlice(gpa, attr_prefix orelse "");
            try inner.appendSlice(gpa, "pub fn unloadAll(allocator: std.mem.Allocator) void {\n");
            const fun_prefix = try text_utils.repeat(gpa, " ", ((depth + 4) * spaces_per_depth));
            defer if (fun_prefix) |p| gpa.free(p);
            {
                const line = try std.fmt.allocPrint(gpa, "{s}position.unload(allocator);\n", .{fun_prefix orelse ""});
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasColor) {
                const line = try std.fmt.allocPrint(gpa, "{s}color.unload(allocator);\n", .{fun_prefix orelse ""});
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasUV) {
                const line = try std.fmt.allocPrint(gpa, "{s}uv.unload(allocator);\n", .{fun_prefix orelse ""});
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            if (sub.hasNormal) {
                const line = try std.fmt.allocPrint(gpa, "{s}normal.unload(allocator);\n", .{fun_prefix orelse ""});
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            {
                const line = try std.fmt.allocPrint(gpa, "{s}indices.unload(allocator);\n", .{fun_prefix orelse ""});
                defer gpa.free(line);
                try inner.appendSlice(gpa, line);
            }
            try inner.appendSlice(gpa, attr_prefix orelse "");
            try inner.appendSlice(gpa, "}\n");

            try inner.appendSlice(gpa, sub_prefix orelse "");
            try inner.appendSlice(gpa, "};\n");
            return posSize;
        }

        /// Builds the mapping half of the `assets_manager` contract, wiring `getMappingCode` into a `MappingDescriptor` consumed by `descriptor` and the asset builder.
        /// - `self` - Descriptor instance whose pointer is stored in the returned mapping for later `getMappingCode` calls.
        ///
        /// Return: Mapping descriptor referencing this instance and its code generator.
        pub fn mapping(self: *Self) MappingDescriptor {
            return .{
                .ptr = self,
                .vtable = .{ .get_code = getMappingCode },
            };
        }

        /// Builds the full binary descriptor consumed by `assets_manager` asset registration, combining `mapping` with the data vtable.
        /// - `self` - Descriptor instance whose pointer backs both the mapping and binary vtable entries.
        ///
        /// Return: Binary descriptor capable of packing `.obj` files and generating their Zig accessors.
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

/// Convenience `f32` instantiation of `ObjBinaryDescriptor` for consumers that do not need configurable precision; registered with `assets_manager` exactly like the generic version and re-exported through `root.zig` and `build.zig`.
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

    const tri = &objects.items[0];
    try std.testing.expect(tri.mesh != null);
    try std.testing.expect(tri.line == null);
    const m = &tri.mesh.?;
    try std.testing.expectEqual(@as(usize, 3), m.vertices.items.len);
    try std.testing.expectEqual(@as(usize, 3), m.indices.items.len);
    try std.testing.expect(m.hasColor);
    try std.testing.expectEqual(@as(f64, 1), m.vertices.items[0].color[0]);
    try std.testing.expectEqual(@as(f64, 0), m.vertices.items[0].color[1]);

    const line = &objects.items[1];
    try std.testing.expect(line.line != null);
    try std.testing.expect(line.mesh == null);
    const l = &line.line.?;
    try std.testing.expectEqual(@as(usize, 3), l.vertices.items.len);
    try std.testing.expectEqual(@as(usize, 3), l.indices.items.len);
    try std.testing.expectEqual(@as(f64, 0), l.vertices.items[0].pos[0]);
    try std.testing.expect(l.hasColor);
    try std.testing.expectEqual(@as(f64, 1), l.vertices.items[0].color[0]);
}
