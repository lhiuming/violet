#include "../brdf.hlsl"
#include "../enc.inc.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"
#include "../util.hlsl"

#include "raytrace.inc.hlsl"
#include "reservoir.hlsl"

RWTexture2D<uint2> rw_prev_reservoir_texture;
RWTexture2D<float4> rw_prev_hit_pos_texture;
RWTexture2D<float3> rw_prev_hit_radiance_texture;

[shader("raygeneration")]
void main()
{
    uint2 dispatch_id = DispatchRaysIndex().xy;

    float prev_depth = prev_depth_texture[dispatch_id.xy];
    if (has_no_geometry_via_depth(prev_depth))
    {
        return;
    }

    uint2 buffer_size;
    prev_depth_texture.GetDimensions(buffer_size.x, buffer_size.y);

    // world position reconstruction from depth buffer
    float depth_error = 1.f / 16777216.f; // 24 bit unorm depth
    float2 screen_pos = (dispatch_id.xy + 0.5f) / float2(buffer_size);
    float4 position_ws_h = mul(frame_params.prev_inv_view_proj, float4(screen_pos * 2.0f - 1.0f, prev_depth + depth_error, 1.0f));
    float3 position_ws = position_ws_h.xyz / position_ws_h.w;

    ReservoirSimple reservoir = reservoir_decode_u32(rw_prev_reservoir_texture[dispatch_id.xy]);
    float4 prev_hit_pos_xyzw = rw_prev_hit_pos_texture[dispatch_id.xy];
    float3 prev_hit_pos = prev_hit_pos_xyzw.xyz;
    float3 prev_hit_radiance = rw_prev_hit_radiance_texture[dispatch_id.xy];
    float prev_reservoir_target_function = prev_hit_pos_xyzw.w;

    // Raytrace
    float3 sample_dir = normalize(prev_hit_pos - position_ws);
    RadianceTraceResult trace_result = trace_radiance(position_ws, sample_dir, true);

    // Validate
    float max_radiance = component_max(trace_result.radiance);
    max_radiance = max(max_radiance, component_max(prev_hit_radiance));
    bool changed = any(abs(trace_result.radiance - prev_hit_radiance) > 0.5f * max_radiance);
    if (changed)
    {
        // TODO just discard?
        reservoir.M = 1;
        //reservoir.W;

        // Store
        // TODO what to do with prev_reservoir_target_function?
        rw_prev_reservoir_texture[dispatch_id.xy] = reservoir_encode_u32(reservoir);
        rw_prev_hit_pos_texture[dispatch_id.xy] = float4(trace_result.position_ws, prev_reservoir_target_function);
        rw_prev_hit_radiance_texture[dispatch_id.xy] = trace_result.radiance;
    }

}