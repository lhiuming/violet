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

    float4x4 prev_view_proj;
	float4x4 prev_inv_view_proj;
    float4 prev_view_pos;

    float4 jitter; // in clip space; xy: this frame; zw: prev frame

	float4 sun_dir;
    float4 sun_inten;
};
[[vk::binding(0, FRAME_DESCRIPTOR_SET_INDEX)]] ConstantBuffer<FrameParams> frame_params;
#define SAMPLER_BINDING_INDEX_BEGIN 1
[[vk::binding(SAMPLER_BINDING_INDEX_BEGIN + 0, FRAME_DESCRIPTOR_SET_INDEX)]] SamplerState sampler_linear_clamp;
[[vk::binding(SAMPLER_BINDING_INDEX_BEGIN + 1, FRAME_DESCRIPTOR_SET_INDEX)]] SamplerState sampler_linear_wrap;
[[vk::binding(SAMPLER_BINDING_INDEX_BEGIN + 2, FRAME_DESCRIPTOR_SET_INDEX)]] SamplerState sampler_nearest_clamp;

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

// world position reconstruction from depth buffer
float3 cs_depth_to_position(uint2 pix_coord, uint2 buffer_size, float depth_buffer_value) {
    float2 screen_uv = (float2(pix_coord) + 0.5f) / float2(buffer_size);
	float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_uv * 2.0f - 1.0f, depth_buffer_value, 1.0f));
	return position_ws_h.xyz / position_ws_h.w;
}

// ray direction by pixel position, without depth buffer
float3 cs_view_ray_direction(uint2 pix_coord, uint2 buffer_size, float2 rand_jitter) {
    float2 pix_coord_f = float2(pix_coord) + 0.5f;
    pix_coord_f += lerp(-0.5f.xx, 0.5f.xx, rand_jitter);
    float2 ndc = pix_coord_f / float2(buffer_size) * 2.0f - 1.0f;
    float4 view_dir_end_h = mul(view_params().inv_view_proj, float4(ndc, 1.0f, 1.0f));
    float3 view_dir_end = view_dir_end_h.xyz / view_dir_end_h.w;
    float3 view_dir = normalize(view_dir_end - view_params().view_pos);
    return view_dir;
}