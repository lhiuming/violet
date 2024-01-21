#include "../frame_bindings.hlsl"
#include "../util.hlsl"
#include "reservoir.hlsl"

Texture2D<uint2> hit_pos_normal_texture;
Texture2D<float> depth_buffer;
Texture2D<float4> prev_ao_texture;
RWTexture2D<float4> rw_ao_texture;

[[vk::push_constant]]
struct PushConstants
{
    uint has_new_sample;
    uint has_prev_frame;
    float radius_ws;
} pc;

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    uint2 buffer_size;
    depth_buffer.GetDimensions(buffer_size.x, buffer_size.y);

    float depth = depth_buffer[dispatch_id.xy];

    // early out
    if (has_no_geometry_via_depth(depth))
    {
        rw_ao_texture[dispatch_id] = 0.0f;
        return;
    }

    // TODO we are not generating movtion vector buffers (not having moving objects); so we reproject directly, asumming everying geometry is static.
    float2 motion_vector;
    {
        float3 position = cs_depth_to_position(dispatch_id, buffer_size, depth);
        float4 hpos = mul(frame_params.view_proj, float4(position, 1));
        float2 ndc = hpos.xy / hpos.w - frame_params.jitter.xy;
        float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(position, 1));
        float2 ndc_reproj = hpos_reproj.xy / hpos_reproj.w - frame_params.jitter.zw;
        motion_vector = (ndc - ndc_reproj) * 0.5;
    }
    float2 src_uv = (float2(dispatch_id) + 0.5f) / float2(buffer_size); 
    float2 history_uv = src_uv - motion_vector;

    // Read history
    float4 history;
    bool is_in_last_view = all(history_uv == saturate(history_uv));
    if (is_in_last_view && bool(pc.has_prev_frame))
    {
        history = prev_ao_texture.SampleLevel(sampler_linear_clamp, history_uv, 0);
    }
    else
    {
        history = 0.0f;
    }

    // Calculate AO
    if (bool(pc.has_new_sample))
    {
        uint2 hit_encoded = hit_pos_normal_texture[dispatch_id.xy];
        HitPosNormal hit = HitPosNormal::decode(hit_encoded);

        float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);
        float3 sample_offset = hit.pos - position_ws;
        float sample_distance = length(sample_offset);
        float ao = select(sample_distance < pc.radius_ws, 0.0, 1.0);

        // Temporal filter
        history.x = lerp(ao, history.x, ((32.0 - 1.0) / 32.0) * history.a);
        history.a = 1.0f;
    }
    else
    {
        // Use ao 1.0 if we just dont have any sample in this pixel
        history.x = select(history.a > 0.0, history.x, 1.0f);
    }

    // Done
    rw_ao_texture[dispatch_id] = history;
}
