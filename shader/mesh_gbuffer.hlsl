// Scene Bindings (Set #1)
#include "frame_bindings.hlsl"
#include "gbuffer.hlsl"
#include "scene_bindings.hlsl"

// Per Material bindings?

struct PushConstants
{
    float4 model_xform_r0;
    float4 model_xform_r1;
    float4 model_xform_r2;
    float4 normal_xform_r0;
    float4 normal_xform_r1;
    float4 normal_xform_r2;
    uint mesh_index;
    uint material_index;
};
[[vk::push_constant]]
PushConstants pc;

void vs_main(uint vert_id: SV_VertexID, out float4 hpos: SV_Position, out float2 uv: TEXCOORD0, out float3 normal: TEXCOORD1, out float4 tangent: TEXCOORD2, out float2 screen_pos: TEXCOORD3, out float3 bitangent: TEXCOORD4)
{
    MeshParams mesh = mesh_params[pc.mesh_index];

    float3 pos = float3(
        asfloat(vertex_buffer[mesh.positions_offset + vert_id * 3 + 0]),
        asfloat(vertex_buffer[mesh.positions_offset + vert_id * 3 + 1]),
        asfloat(vertex_buffer[mesh.positions_offset + vert_id * 3 + 2]));
    uv = float2(
        asfloat(vertex_buffer[mesh.texcoords_offset + vert_id * 2 + 0]),
        asfloat(vertex_buffer[mesh.texcoords_offset + vert_id * 2 + 1]));
    normal = float3(
        asfloat(vertex_buffer[mesh.normals_offset + vert_id * 3 + 0]),
        asfloat(vertex_buffer[mesh.normals_offset + vert_id * 3 + 1]),
        asfloat(vertex_buffer[mesh.normals_offset + vert_id * 3 + 2]));
    tangent = float4(
        asfloat(vertex_buffer[mesh.tangents_offset + vert_id * 4 + 0]),
        asfloat(vertex_buffer[mesh.tangents_offset + vert_id * 4 + 1]),
        asfloat(vertex_buffer[mesh.tangents_offset + vert_id * 4 + 2]),
        asfloat(vertex_buffer[mesh.tangents_offset + vert_id * 4 + 3]));

    normal = normalize(float3(
        dot(pc.normal_xform_r0.xyz, normal),
        dot(pc.normal_xform_r1.xyz, normal),
        dot(pc.normal_xform_r2.xyz, normal)));
    tangent.xyz = normalize(float3(
        dot(pc.normal_xform_r0.xyz, tangent.xyz),
        dot(pc.normal_xform_r1.xyz, tangent.xyz),
        dot(pc.normal_xform_r2.xyz, tangent.xyz)));

    bitangent = normalize(tangent.w * cross(normal, tangent.xyz));

    float3 wpos = float3(
        dot(pc.model_xform_r0, float4(pos, 1.0)),
        dot(pc.model_xform_r1, float4(pos, 1.0)),
        dot(pc.model_xform_r2, float4(pos, 1.0)));
    float4x4 view_proj = view_params().view_proj;
    hpos = mul(view_proj, float4(wpos, 1.0f));
    screen_pos = hpos.xy / hpos.w * 0.5f + 0.5f;
}

void ps_main(
    // Input
    float4 hpos: SV_Position,
    float2 uv: TEXCOORD0,
    float3 normal: TEXCOORD1,
    float4 tangent: TEXCOORD2,
    noperspective float2 screen_pos: TEXCOORD3,
    float3 bitangent: TEXCOORD4,
    bool if_front_face: SV_IsFrontFace,
    // Output
    out uint output0: SV_Target0,
    out uint output1: SV_Target1,
    out uint output2: SV_Target2,
    out uint output3: SV_Target3)
{
    MaterialParams mat = material_params[pc.material_index];

    // TODO sampler based on model definition
    float4 base_color = bindless_textures[mat.base_color_index].Sample(sampler_linear_wrap, uv);
    float4 normal_map = bindless_textures[mat.normal_index].Sample(sampler_linear_wrap, uv);
    float4 metal_rough = bindless_textures[mat.metallic_roughness_index].Sample(sampler_linear_wrap, uv);

    // normal mapping
    float3 normal_ts = normal_map.xyz * 2.0f - 1.0f;
    float3 normal_ws = normalize(normal_ts.x * tangent.xyz + normal_ts.y * bitangent + normal_ts.z * normal);
    normal_ws *= if_front_face ? 1.0f : -1.0f;

    GBuffer gbuffer;
    gbuffer.color = base_color.rgb;
    gbuffer.metallic = metal_rough.r;
    gbuffer.perceptual_roughness = metal_rough.g;
    gbuffer.normal = normal_ws;
    gbuffer.shading_path = 1;
    uint4 o4 = encode_gbuffer(gbuffer);
    output0 = o4.x;
    output1 = o4.y;
    output2 = o4.z;
    output3 = o4.w;
}

