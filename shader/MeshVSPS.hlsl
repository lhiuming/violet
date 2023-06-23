/* NOTE for Debugging: 
Conceptually (and as reflected in the way matrix components are accessed), SPIRV use columns to represent matrix, while HLSL use rows. HLSL-to-SPIRV codegen must respect this relationship in order to handle matrix compoenent access properly. 

The result is that, matrix representation is 'transposed' between HLSL srouce code and generated SPIRV, e.g. a float4x3 matrix in HLSL is represented as a 3x4 matrix in generated SPIRV, such that m[0] in HLSL and m[0] in SPIRV refer to the same float3 data.

This also affects the generated packing rule. When HLSL code specify a row-major layout, the generated SPIRV must use column-major layout, and vice versa. Therefore, the below HLSL column_major packing specification is translated to a (transposed) RowMajor packing in SPIRV, in order to make the packing behave as expected from user's point of view. 

These fact is important when you are investigating the generated SPIRV code or raw buffer content from a debugger, e.g. RenderDoc.

See: https://github.com/Microsoft/DirectXShaderCompiler/blob/master/docs/SPIR-V.rst#vectors-and-matrices
*/
// Use column-major as GLAM
#pragma pack_matrix(column_major)


//// BRDF

#define PI 3.14159265359

// Fresnel with f90
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float3 F_Schlick_with_f90(float u, float3 f0, float f90) {
    return f0 + (f90.xxx - f0) * pow(1.0 - u, 5.0);
}

// Fresnel 
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float3 F_Schlick(float u, float3 f0) {
	// Replace f90 with 1.0 in F_Schlick, and some refactoring
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

// Fresnel with single channel
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float F_Schlick_single(float u, float f0, float f90) {
	return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float D_GGX(float NoH, float roughness) {
    float a = NoH * roughness;
    float k = roughness / (1.0 - NoH * NoH + a * a);
    return k * k * (1.0 / PI);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
    float a2 = roughness * roughness;
    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float Fd_Lambert() {
    return 1.0 / PI;
}

// Disney BRDF
// https://google.github.io/filament/Filament.md.html#materialsystem/brdf
float Fd_Burley(float NoV, float NoL, float LoH, float roughness) {
    float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    float lightScatter = F_Schlick_single(NoL, 1.0, f90);
    float viewScatter = F_Schlick_single(NoV, 1.0, f90);
    return lightScatter * viewScatter * (1.0 / PI);
}

// End BRDF

// Scene Bindings (Set #1)
#include "scene_bindings.hlsl"

// Per Pass Bindings (Set #2)
[[vk::binding(0, 2)]] TextureCube<float3> skycube;

// Per Material bindings?

struct PushConstants
{
	float4x4 model_transform;
	// Geometry
	uint positions_offset;
	uint texcoords_offset;
	uint normals_offset;
	uint tangnets_offset;
	// Bindless Materials
	uint color_texture;
	uint normal_texture;
	uint metal_rough_texture;
};
[[vk::push_constant]]
PushConstants pc;

void vs_main(uint vert_id : SV_VertexID, out float4 hpos : SV_Position, out float2 uv : TEXCOORD0, out float3 normal : TEXCOORD1, out float4 tangent : TEXCOORD2, out float2 screen_pos : TEXCOORD3, out float3 bitangent : TEXCOORD4)
{
#if 0
	const float2 vbuffer[] = {
		float2(0, 1.0f),
		float2(0.5, -0.5f),
		float2(-0.5, -0.5f),
	};
	float2 v = vbuffer[vert_id % 3];
	hpos = float4(v, 0.5f, 1.0f);
#else
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
#endif
}

float4 unpack_unorm(uint packed_value)
{
	return float4(
		((packed_value >> 0) & 0xFF) / 255.0f,
		((packed_value >> 8) & 0xFF) / 255.0f,
		((packed_value >> 16) & 0xFF) / 255.0f,
		((packed_value >> 24) & 0xFF) / 255.0f
	);
}

float3 cal_lighting(float3 v /*view*/, float3 l /*light*/, float3 n /*normal*/, float perceptualRoughness, float3 diffuseColor, float3 specularColor)
{
	float3 h = normalize(v + l);

    float NoV = abs(dot(n, v)) + 1e-5;
    float NoL = clamp(dot(n, l), 0.0, 1.0);
    float NoH = clamp(dot(n, h), 0.0, 1.0);
    float LoH = clamp(dot(l, h), 0.0, 1.0);

    // perceptually linear roughness to roughness (see parameterization)
	// Clamp for cheap specular aliasing under punctual light
	perceptualRoughness = max(perceptualRoughness, 0.045f);
    float roughness = perceptualRoughness * perceptualRoughness;

    float D = D_GGX(NoH, roughness);
    float3 F = F_Schlick(LoH, specularColor);
    float V = V_SmithGGXCorrelated(NoV, NoL, roughness);

    // specular BRDF
    float3 Fr = (D * V) * F;

    // diffuse BRDF
    float3 Fd = diffuseColor * Fd_Lambert();

	return (Fd + Fr) * NoL;
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
	out float4 output : SV_Target0)
{
	float4 base_color = bindless_textures[pc.color_texture].Sample(sampler_linear_clamp, uv);
	float4 normal_map = bindless_textures[pc.normal_texture].Sample(sampler_linear_clamp, uv);
	float4 metal_rough = bindless_textures[pc.metal_rough_texture].Sample(sampler_linear_clamp, uv);

	// normal mapping
	float3 normal_ts = normal_map.xyz * 2.0f - 1.0f;
	float3 normal_ws = normalize( normal_ts.x * tangent.xyz + normal_ts.y * bitangent + normal_ts.z * normal );

	// world position reconstruction
	float depth = hpos.z / hpos.w;
	float3 position_ws = mul(view_params.inv_view_proj, float4(screen_pos * 2.0f - 1.0f, depth, 1.0f)).xyz;

	float3 view = normalize(view_params.view_pos - position_ws);
	float3 light = float3(0.0f, 0.0f, 1.0f);
	float metallic = metal_rough.b;
	float roughness = metal_rough.g;
	float3 diffuseColor = base_color.rgb * (1.0f - metallic);
	float3 specularColor = lerp(float3(0.04f, 0.04f, 0.04f), base_color.rgb, metallic);
	float3 light_inten = float3(1.0f, 1.0f, 1.0f) * PI;
	float3 lighting = cal_lighting(view, light, normal_ws, roughness, diffuseColor, specularColor) * light_inten;

	// IBL fake
	{
		float3 refl_dir = reflect(-view, normal_ws);
		lighting += skycube.SampleLevel(sampler_linear_clamp, refl_dir, 0).rgb * 0.5f;
	}

	float3 ambient = 0.1f;
	lighting += diffuseColor * ambient;

	// Material inspect rect
	if ( (screen_pos.x > 0.25f) && (screen_pos.x < 0.3f)
		&& (screen_pos.y > 0.25f) && (screen_pos.y < 0.3f) )
	{
		float u = frac(screen_pos.x / 0.05f);
		if (u < 0.33f)
			output = base_color;
		else if (u < 0.66f)
			output = normal_map;
		else
			output = metal_rough;
		return;
	}

#if 0
//	output = float4(normal.xyz * .5f + .5f, 1.0f);
	output = float4(normal_ws * .5f + .5f, 1.0f);
//	output = float4(base_color.rgb, 1.0f);
//	output = float4(normal_map.rgb * 0.5f + 0.5f, 1.0f);
//	output = float4(metal_rough.rgb, 1.0f);
	return;
#endif
	output = float4(lighting, 1.0f);
} 


