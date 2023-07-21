#pragma once 
#pragma pack_matrix(column_major)

#define FRAME_DESCRIPTOR_SET_INDEX 2

struct FrameParams
{
    // TODO may be in seperate set
    float4x4 view_proj;
	float4x4 inv_view_proj;
	float4 view_pos;
	float4 view_ray_top_left;
	float4 view_ray_right_shift;
	float4 view_ray_down_shift;

	float4 sun_dir;
    float4 sun_inten;
};
[[vk::binding(0, FRAME_DESCRIPTOR_SET_INDEX)]] ConstantBuffer<FrameParams> frame_params;

struct ViewParams {
    // TODO may be in seperate set
    float4x4 view_proj;
	float4x4 inv_view_proj;
	float3 view_pos;
	float3 view_ray_top_left;
	float3 view_ray_right_shift;
	float3 view_ray_down_shift;
};

ViewParams view_params() {
    ViewParams ret;
    ret.view_proj = frame_params.view_proj;
    ret.inv_view_proj = frame_params.inv_view_proj;
    ret.view_pos = frame_params.view_pos.xyz;
    ret.view_ray_top_left = frame_params.view_ray_top_left.xyz;
    ret.view_ray_right_shift = frame_params.view_ray_right_shift.xyz;
    ret.view_ray_down_shift = frame_params.view_ray_down_shift.xyz;
    return ret;
}