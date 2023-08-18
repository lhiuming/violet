#include "frame_bindings.hlsl"

Texture2D<float> gbuffer_depth;
Texture2D<float3> source;
Texture2D<float3> history;
RWTexture2D<float3> rw_target;

struct PushConstants
{
    uint has_history;
};
[[vk::push_constant]]
PushConstants pc;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id : SV_DispatchThreadID) {
    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    // Reproject
    float depth = gbuffer_depth[dispatch_id];
    float3 position = cs_depth_to_position(dispatch_id, buffer_size, depth);
    float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(position, 1));
    float2 ndc_reproj = hpos_reproj.xy / hpos_reproj.w;
    bool in_last_view = all(abs(ndc_reproj) < 1.0f);

    float3 src = source[dispatch_id];

    // Blend
    // TODO rejection
    float3 blended;
    if (pc.has_history && in_last_view)
    {
        // TODO blend in display linear space?
        float2 uv = ndc_reproj.xy * 0.5 + 0.5;
        float3 hsty = history.SampleLevel(sampler_linear_clamp, uv, 0);
        blended = lerp(hsty, src, 1.0 / 8);
    }
    else
    {
        blended = src;
    }

#if 0
    // NaN Stopping
    if (any(isnan(blended))) {
        blended = float3(0.9, 0.1, 0.9) * 2;
    }
#endif

    rw_target[dispatch_id] = blended;
}