#include "../config.inc.hlsl"
#include "../frame_bindings.hlsl"
#include "../scene_bindings.hlsl"

#include "geometry_ray.inc.hlsl"

#define SKIP_POSITION_LOAD 0
#define TEX_LOD_BIAS 2

// ----------------
// Geometry loading

float2 load_float2(uint attr_offset, uint vert_id)
{
    uint2 words = uint2(
        vertex_buffer[attr_offset + vert_id * 2 + 0],
        vertex_buffer[attr_offset + vert_id * 2 + 1]);
    return asfloat(words);
}

float3 load_float3(uint attr_offset, uint vert_id)
{
    uint3 words = uint3(
        vertex_buffer[attr_offset + vert_id * 3 + 0],
        vertex_buffer[attr_offset + vert_id * 3 + 1],
        vertex_buffer[attr_offset + vert_id * 3 + 2]);
    return asfloat(words);
}

float4 load_float4(uint attr_offset, uint vert_id)
{
    uint4 words = uint4(
        vertex_buffer[attr_offset + vert_id * 4 + 0],
        vertex_buffer[attr_offset + vert_id * 4 + 1],
        vertex_buffer[attr_offset + vert_id * 4 + 2],
        vertex_buffer[attr_offset + vert_id * 4 + 3]);
    return asfloat(words);
}

// NOTE: bary.x is weight for vertex_1, bary.y is weight for vertex_2
#define INTERPOLATE_MESH_ATTR(type, attr_offset, indicies, bary) \
    (load_##type(attr_offset, indicies.x) * (1.0f - bary.x - bary.y) + load_##type(attr_offset, indicies.y) * bary.x + load_##type(attr_offset, indicies.z) * bary.y)

struct Attribute
{
    float2 bary;
};

[shader("closesthit")]
void main(inout GeometryRayPayload payload, in Attribute attr)
{
    // GeometryGroup <-> BLAS
    uint geometry_group_index = InstanceID(); // BLAS index (via instance_custom_index)
    uint geometry_index_offset = geometry_group_params[geometry_group_index].geometry_index_offset;

    uint local_geometry_index = GeometryIndex(); // within the BLAS
    uint mesh_index = geometry_index_offset + local_geometry_index;

    uint triangle_index = PrimitiveIndex();

    MeshParams mesh = mesh_params[mesh_index];
    uint3 indicies = uint3(
        index_buffer[mesh.index_offset + triangle_index * 3 + 0],
        index_buffer[mesh.index_offset + triangle_index * 3 + 1],
        index_buffer[mesh.index_offset + triangle_index * 3 + 2]);

    float2 uv = INTERPOLATE_MESH_ATTR(float2, mesh.texcoords_offset, indicies, attr.bary);
    float3 pos_ls = INTERPOLATE_MESH_ATTR(float3, mesh.positions_offset, indicies, attr.bary);
    float3 normal_ls = INTERPOLATE_MESH_ATTR(float3, mesh.normals_offset, indicies, attr.bary);
    float4 tangent_ls = INTERPOLATE_MESH_ATTR(float4, mesh.tangents_offset, indicies, attr.bary);

    // transform mesh data
    float4x3 normal_xform = WorldToObject4x3(); // transpose(inverse(ObjectToWorld()))
    float3 normal = normalize(mul(normal_xform, normal_ls).xyz);
    float3 tangent = normalize(mul(normal_xform, tangent_ls.xyz).xyz);

    // TODO calculate before interpolation?
    float3 bitangent = normalize(tangent_ls.w * cross(normal, tangent.xyz));

    // TODO inlining the material parameters?
    MaterialParams mat = material_params[mesh.material_index];
    float4 base_color = bindless_textures[mat.base_color_index].SampleLevel(sampler_linear_wrap, uv, TEX_LOD_BIAS);
    float4 metal_rough = bindless_textures[mat.metallic_roughness_index].SampleLevel(sampler_linear_wrap, uv, TEX_LOD_BIAS);
    float4 normal_map = bindless_textures[mat.normal_index].SampleLevel(sampler_linear_wrap, uv, TEX_LOD_BIAS);

    #if HACK_MAKE_EVERYTHING_GLOSSY 
    metal_rough.g *= HACK_ROUGHNESS_MULTIPLIER;
    #endif

    // normal mapping
    float3 normal_ts = normal_map.xyz * 2.0f - 1.0f;
    float3 normal_ws = normalize(normal_ts.x * tangent.xyz + normal_ts.y * bitangent + normal_ts.z * normal);

    // World position
    float3 position_ws = mul(ObjectToWorld3x4(), float4(pos_ls, 1.0f));
#if SKIP_POSITION_LOAD
    position_ws = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
#endif

    // Geometry normal
    float3 pos0 = load_float3(mesh.positions_offset, indicies.x);
    float3 pos1 = load_float3(mesh.positions_offset, indicies.y);
    float3 pos2 = load_float3(mesh.positions_offset, indicies.z);
    float3 normal_geo = cross(pos1 - pos0, pos2 - pos0);
    normal_geo = normalize(mul(normal_xform, normal_geo).xyz);
#if SKIP_POSITION_LOAD
    normal_geo = normal_ws;
#endif

    // Two-side geometry: flip the geometry attribute if ray hit from back face
    if (dot(normal_geo, WorldRayDirection()) > 0.0f)
    {
        normal_geo = -normal_geo;
        normal_ws = -normal_ws;
    }

#if SHRINK_PAYLOAD

    payload.hit_t = RayTCurrent();
    payload.base_color_enc = GeometryRayPayload::enc_color_rough_metal(base_color.rgb, metal_rough.g, metal_rough.r);
    payload.normal_ws_enc = normal_encode_oct_u32(normal_ws);
    payload.normal_geo_ws_enc = normal_encode_oct_u32(normal_geo);

#else

    payload.missed = false;
    payload.hit_t = RayTCurrent();
    payload.normal_ws = normal_ws;
    payload.normal_geo_ws = normal_geo;
#if !GEOMETRY_RAY_PAYLOAD_NO_POSITION
    payload.position_ws = position_ws;
#endif
    payload.base_color = base_color.xyz;
    payload.metallic = metal_rough.r;
    payload.perceptual_roughness = metal_rough.g;
    payload.mesh_index = mesh_index;
    payload.triangle_index = triangle_index;

#endif
}
