// NOTE: SPIRV use columns to represent matrix, while HLSL use rows. HLSL-to-SPIRV codegen does not translate this for us (or translate partially, for some reason, which is confusing).
// The result is, the actual memory layout is always the opposite of what is in HLSL code. That is, if you tag the matrix as row_major, the data layout in buffer is actually column major, and vice versa.
// But if you load a matrix from buffer for, e.g., multiplication, the matrix is automatically got transposed and the multiplication order is flipped for you, so the result would 'seems' respecting the intent matrix layout. 
// You only need to be careful when you are investigating the generated code or raw buffer content.
// See: https://github.com/Microsoft/DirectXShaderCompiler/blob/master/docs/SPIR-V.rst#vectors-and-matrices
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
};
[[vk::push_constant]]
PushConstants push_constant;

void vs_main(uint vert_id : SV_VertexID, out float4 hpos : SV_Position) 
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
	float3 v = float3( 
		asfloat(vertex_buffer[vert_id * 3 + 0]),
		asfloat(vertex_buffer[vert_id * 3 + 1]),
		asfloat(vertex_buffer[vert_id * 3 + 2]));
	float3x4 model_transform = push_constant.model_transform;
	float3 wpos = mul(model_transform, float4(v, 1.0f));
	float4x4 view_proj = view_params.view_proj;
	hpos =  mul(view_proj, float4(wpos, 1.0f));
#endif
}

void ps_main(out float4 output : SV_Target0)
{
	output = float4(1.0f, 1.0f, 1.0f, 1.0f);
} 