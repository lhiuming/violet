#include "../frame_bindings.hlsl"
#include "../util.hlsl"

Texture2D<float> depth_buffer;
Texture2D<float3> prev_ind_diff_texture;
Texture2D<float3> curr_ind_diff_texture;
RWTexture2D<float3> rw_filtered_ind_diff_texture;

[[vk::push_constant]]
struct PushConstant {
    uint has_prev_frame;
} pc;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) 
{
    uint2 buffer_size;
    depth_buffer.GetDimensions(buffer_size.x, buffer_size.y);

    // early out
    float depth = depth_buffer[dispatch_id];
    if (has_no_geometry_via_depth(depth))
    {
        rw_filtered_ind_diff_texture[dispatch_id] = float3(0, 0, 0);
        return;
    }

    float3 src_color = curr_ind_diff_texture[dispatch_id];

    // History
    float3 prev_color;
    if (bool(pc.has_prev_frame))
    {
        // TODO save motion vector in a buffer so that we can reuse thi calculation in different passes
        float3 position = cs_depth_to_position(dispatch_id, buffer_size, depth);
        float4 hpos = mul(frame_params.view_proj, float4(position, 1));
        float2 ndc = hpos.xy / hpos.w - frame_params.jitter.xy;
        float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(position, 1));
        float2 ndc_reproj = hpos_reproj.xy / hpos_reproj.w - frame_params.jitter.zw;
        float2 motion_vector = (ndc - ndc_reproj) * 0.5;

        float2 src_uv = (float2(dispatch_id) + 0.5f) / float2(buffer_size); 
        float2 history_uv = src_uv - motion_vector;

        prev_color = prev_ind_diff_texture.SampleLevel(sampler_linear_clamp, history_uv, 0.0);
    }
    else
    {
        prev_color = src_color;
    }

    // Blend
    // TODO Box clamping to reduce ghosting
    const float BLEND_FACTOR = 1.0 / 16.0f;
    float3 filtered_color = lerp(prev_color, src_color, BLEND_FACTOR);

    rw_filtered_ind_diff_texture[dispatch_id] = filtered_color;
}