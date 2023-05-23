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

struct PushConstants
{
	float3x4 model_transform;
	uint pos_offset;
	uint uv_offset;
};
[[vk::push_constant]]
PushConstants pc;

void vs_main(uint vert_id : SV_VertexID, out float4 hpos : SV_Position, out float2 uv : TEXCOORD0) 
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
#endif
}

void ps_main(float2 uv: TEXCOORD0, out float4 output : SV_Target0)
{
	//output = float4(1.0f, 1.0f, 1.0f, 1.0f);
	output = float4(uv, 0.0f, 1.0f);
} 