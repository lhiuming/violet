// Scene Bindings (Set #1)
#include "scene_bindings.hlsl"
#include "gbuffer.hlsl"

// Per Material bindings?

struct PushConstants
{
	float4x4 model_transform;
	// Geometry
	uint positions_offset;
	uint texcoords_offset;
	uint normals_offset;
	uint tangnets_offset;
	uint material_index;
};
[[vk::push_constant]]
PushConstants pc;

void vs_main(uint vert_id : SV_VertexID, out float4 hpos : SV_Position, out float2 uv : TEXCOORD0, out float3 normal : TEXCOORD1, out float4 tangent : TEXCOORD2, out float2 screen_pos : TEXCOORD3, out float3 bitangent : TEXCOORD4)
{
	float3 pos = float3( 
		asfloat(vertex_buffer[pc.positions_offset + vert_id * 3 + 0]),
		asfloat(vertex_buffer[pc.positions_offset + vert_id * 3 + 1]),
		asfloat(vertex_buffer[pc.positions_offset + vert_id * 3 + 2]));
	uv = float2(
		asfloat(vertex_buffer[pc.texcoords_offset + vert_id * 2 + 0]),
		asfloat(vertex_buffer[pc.texcoords_offset + vert_id * 2 + 1]));
	normal = float3(
		asfloat(vertex_buffer[pc.normals_offset + vert_id * 3 + 0]),
		asfloat(vertex_buffer[pc.normals_offset + vert_id * 3 + 1]),
		asfloat(vertex_buffer[pc.normals_offset + vert_id * 3 + 2]));
	tangent = float4(
		asfloat(vertex_buffer[pc.tangnets_offset + vert_id * 4 + 0]),
		asfloat(vertex_buffer[pc.tangnets_offset + vert_id * 4 + 1]),
		asfloat(vertex_buffer[pc.tangnets_offset + vert_id * 4 + 2]),
		asfloat(vertex_buffer[pc.tangnets_offset + vert_id * 4 + 3]));

	bitangent = normalize(tangent.w * cross(normal, tangent.xyz));

	float4x4 model_transform = pc.model_transform;
	float3 wpos = mul(model_transform, float4(pos, 1.0f)).xyz;
	float4x4 view_proj = view_params.view_proj;
	hpos =  mul(view_proj, float4(wpos, 1.0f));
	screen_pos = hpos.xy / hpos.w * 0.5f + 0.5f;
}

void ps_main(
	// Input
	float4 hpos : SV_Position, 
	float2 uv : TEXCOORD0,
	float3 normal : TEXCOORD1,
	float4 tangent : TEXCOORD2,
	noperspective float2 screen_pos : TEXCOORD3,
	float3 bitangent : TEXCOORD4,
	// Output
	out uint4 output : SV_Target0)
{
	MaterialParams mat = material_params[pc.material_index];

	float4 base_color = bindless_textures[mat.base_color_index].Sample(sampler_linear_clamp, uv);
	float4 normal_map = bindless_textures[mat.normal_index].Sample(sampler_linear_clamp, uv);
	float4 metal_rough = bindless_textures[mat.metallic_roughness_index].Sample(sampler_linear_clamp, uv);

	// normal mapping
	float3 normal_ts = normal_map.xyz * 2.0f - 1.0f;
	float3 normal_ws = normalize( normal_ts.x * tangent.xyz + normal_ts.y * bitangent + normal_ts.z * normal );

    GBuffer gbuffer;
    gbuffer.color = base_color.rgb;
    gbuffer.metallic = metal_rough.b;
    gbuffer.perceptual_roughness = metal_rough.g;
    gbuffer.normal = normal_ws;
    gbuffer.shading_path = 1;
    output = encode_gbuffer(gbuffer);
} 


