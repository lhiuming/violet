#pragma once

/* NOTE for Debugging: 
Conceptually (and as reflected in the way matrix components are accessed), SPIRV use columns to represent matrix, while HLSL use rows. HLSL-to-SPIRV codegen must respect this relationship in order to handle matrix compoenent access properly. 

The result is that, matrix representation is 'transposed' between HLSL srouce code and generated SPIRV, e.g. a float4x3 matrix in HLSL is represented as a 3x4 matrix in generated SPIRV, such that m[0] in HLSL and m[0] in SPIRV refer to the same float3 data.

This also affects the generated packing rule. When HLSL code specify a row-major layout, the generated SPIRV must use column-major layout, and vice versa. Therefore, the below HLSL column_major packing specification is translated to a (transposed) RowMajor packing in SPIRV, in order to make the packing behave as expected from user's point of view. 

These fact is important when you are investigating the generated SPIRV code or raw buffer content from a debugger, e.g. RenderDoc.

See: https://github.com/Microsoft/DirectXShaderCompiler/blob/master/docs/SPIR-V.rst#vectors-and-matrices
*/
// Use column-major as GLAM
#pragma pack_matrix(column_major)

#define SCENE_DESCRIPTOR_SET_INDEX 1

// Scene/Persistent Bindings
[[vk::binding(0, SCENE_DESCRIPTOR_SET_INDEX)]] Buffer<uint> vertex_buffer;
struct MaterialParams {
    uint base_color_index;
    uint metallic_roughness_index;
    uint normal_index;
    uint pad;
};
[[vk::binding(1, SCENE_DESCRIPTOR_SET_INDEX)]] StructuredBuffer<MaterialParams> material_params;
[[vk::binding(2, SCENE_DESCRIPTOR_SET_INDEX)]] Texture2D bindless_textures[];
struct ViewParams 
{
	float4x4 view_proj;
	float4x4 inv_view_proj;
	float3 view_pos;
	float3 view_ray_top_left;
	float3 view_ray_right_shift;
	float3 view_ray_down_shift;
	float3 sun_dir;
};
[[vk::binding(3, SCENE_DESCRIPTOR_SET_INDEX)]] ConstantBuffer<ViewParams> view_params;
[[vk::binding(4, SCENE_DESCRIPTOR_SET_INDEX)]] SamplerState sampler_linear_clamp;
