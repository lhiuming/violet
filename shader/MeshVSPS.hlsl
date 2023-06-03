/* NOTE for Debugging: 
Conceptually (and as reflected in the way matrix components are accessed), SPIRV use columns to represent matrix, while HLSL use rows. HLSL-to-SPIRV codegen must respect this relationship in order to handle matrix compoenent access properly. 

The result is that, matrix representation is 'transposed' between HLSL srouce code and generated SPIRV, e.g. a float4x3 matrix in HLSL is represented as a 3x4 matrix in generated SPIRV, such that m[0] in HLSL and m[0] in SPIRV refer to the same float3 data.

This also affects the generated packing rule. When HLSL code specify a row-major layout, the generated SPIRV must use column-major layout, and vice versa. Therefore, the below HLSL row_major packing specification is translated to a ColMajor packing in SPIRV, in order to make the packing behave as expected from user's point of view. 

These fact is important when you are investigating the generated SPIRV code or raw buffer content from a debugger, e.g. RenderDoc.

See: https://github.com/Microsoft/DirectXShaderCompiler/blob/master/docs/SPIR-V.rst#vectors-and-matrices
*/
#pragma pack_matrix(row_major)

struct ViewParams 
{
	float4x4 view_proj;
};
[[vk::binding(0, 1)]]
ConstantBuffer<ViewParams> view_params;

Buffer<uint> vertex_buffer;
Texture2DArray material_texture;
SamplerState material_texture_sampler;

struct PushConstants
{
	float3x4 model_transform;
	// Geometry
	uint pos_offset;
	uint uv_offset;
	// Materials
	uint2 base_color;
	uint2 normal;
	uint2 metal_rough;
};
[[vk::push_constant]]
PushConstants pc;

void vs_main(uint vert_id : SV_VertexID, out float4 hpos : SV_Position, out float2 uv : TEXCOORD0, out float2 screen_pos : TEXCOORD1) 
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
		asfloat(vertex_buffer[pc.pos_offset + vert_id * 3 + 0]),
		asfloat(vertex_buffer[pc.pos_offset + vert_id * 3 + 1]),
		asfloat(vertex_buffer[pc.pos_offset + vert_id * 3 + 2]));
	uv = float2(
		asfloat(vertex_buffer[pc.uv_offset + vert_id * 2 + 0]),
		asfloat(vertex_buffer[pc.uv_offset + vert_id * 2 + 1]));

	float3x4 model_transform = pc.model_transform;
	float3 wpos = mul(model_transform, float4(pos, 1.0f));
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

float4 sample_material_texture(float2 local_uv, uint2 packed_params)
{
	float4 scale_offset = unpack_unorm(packed_params.x);
	float2 uv = local_uv * scale_offset.xy + scale_offset.zw;
	return material_texture.Sample(material_texture_sampler, float3(uv, packed_params.y));
}

void ps_main(float4 hpos : SV_Position, float2 uv : TEXCOORD0, noperspective float2 screen_pos : TEXCOORD1, out float4 output : SV_Target0)
{
	float4 base_color = sample_material_texture(uv, pc.base_color);
	float4 normal = sample_material_texture(uv, pc.normal);
	float4 metal_rough = sample_material_texture(uv, pc.metal_rough);

	//output = float4(1.0f, 1.0f, 1.0f, 1.0f);
	//output = float4(uv, hpos.z / hpos.w, 1.0f);
	float u = frac(screen_pos.x * 3);
	if (u < 0.33f)
		output = base_color;
	else if (u < 0.66f)
		output = normal;
	else
		output = metal_rough;
} 