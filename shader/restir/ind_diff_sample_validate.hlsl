#include "../constants.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"
#include "../util.hlsl"

#include "raytrace.inc.hlsl"
#include "reservoir.hlsl"

#define DO_VALIDATION 1

// Depth buffer matching with the reservoir
Texture2D<float> depth_texture;
// Reservoir (with selected sample) to be validated
RWTexture2D<uint> rw_reservoir_texture;
RWTexture2D<uint2> rw_hit_pos_normal_texture;
RWTexture2D<float3> rw_hit_radiance_texture;

struct PushConstants
{
    // If validating reservoir from last frame
    //uint prev_frame;
};
[[vk::push_constant]]
PushConstants pc;

[shader("raygeneration")]
void main()
{
#if !DO_VALIDATION
    return;
#endif

    const uint2 dispatch_id = DispatchRaysIndex().xy;

    uint2 buffer_size;
    depth_texture.GetDimensions(buffer_size.x, buffer_size.y);
    
    const float depth = depth_texture[dispatch_id.xy];    
    if (has_no_geometry_via_depth(depth))
    {
        return;
    }

    // Read sample to validate
    uint2 hit_pos_normal_enc = rw_hit_pos_normal_texture[dispatch_id];
    HitPosNormal hit = HitPosNormal::decode(hit_pos_normal_enc);
    float3 hit_radiance = rw_hit_radiance_texture[dispatch_id];

    float3 sample_origin_ws;
    //if (bool(pc.prev_frame))
    if (true)
    {
        sample_origin_ws = cs_prev_depth_to_position(dispatch_id, buffer_size, depth);
    }
    float3 sample_dir_ws = normalize(hit.pos - sample_origin_ws);

    // Trace for up-to-date radiance
    RadianceTraceResult trace_result = trace_radiance(sample_origin_ws, sample_dir_ws, true);

    // Just replace the sample if too different
    // TODO should blend in natually
    float3 prev_radiance = hit_radiance;
    float max_radiance = component_max(hit_radiance);
    bool changed = any(abs(prev_radiance - trace_result.radiance) > max_radiance * 0.5f);
    if (changed)
    {
        ReservoirSimple reservior;
        reservior.M = 1;
        reservior.W = 1.0 / TWO_PI;
        rw_reservoir_texture[dispatch_id] = reservior.encode_32b();
    }

    // Update hit sample
    hit.pos = trace_result.position_ws;
    hit.normal = trace_result.normal_ws;
    rw_hit_pos_normal_texture[dispatch_id] = hit.encode();
    rw_hit_radiance_texture[dispatch_id] = trace_result.radiance;
}
