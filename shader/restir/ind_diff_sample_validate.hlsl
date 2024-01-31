#include "../constants.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"
#include "../util.hlsl"

#include "raytrace.inc.hlsl"
#include "reservoir.hlsl"

#define DO_VALIDATION 1

// Exagerate the radiance difference when scaling down prev M,  to increase responsiveness
#define RADIANCE_DIFF_MULTIPLIER 4.0

// Depth buffer matching with the reservoir
Texture2D<float> depth_texture;
// Reservoir (with selected sample) to be validated
RWTexture2D<uint> rw_reservoir_texture;
RWTexture2D<uint2> rw_hit_pos_normal_texture;
RWTexture2D<float3> rw_hit_radiance_texture;

/*
struct PushConstants
{
};
[[vk::push_constant]]
PushConstants pc;
*/

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
    // TODO reproject and only validate samples that are visible in this frame 
    // TODO trace normal diffuse ray for that pixels. ref: [Bouma 2023]
    uint2 hit_pos_normal_enc = rw_hit_pos_normal_texture[dispatch_id];
    HitPosNormal hit = HitPosNormal::decode(hit_pos_normal_enc);
    float3 prev_hit_radiance = rw_hit_radiance_texture[dispatch_id];

    float3 sample_origin_ws = cs_prev_depth_to_position(dispatch_id, buffer_size, depth);
    float3 sample_dir_ws = normalize(hit.pos - sample_origin_ws);

    // Trace for up-to-date radiance
    RadianceTraceResult trace_result = trace_radiance(sample_origin_ws, sample_dir_ws, true);

    // Scale down history confidance if lighting condition changes
    float lumi_prev = luminance(prev_hit_radiance);
    float lumi_curr = luminance(trace_result.radiance);
    float3 radiance_diff = trace_result.radiance - prev_hit_radiance;
    float lumi_diff = luminance(abs(radiance_diff));
    bool changed = lumi_diff > (lumi_prev / 256.0); // a 1/256 change is mearly noticable
    float prev_ray_len = length(hit.pos - sample_origin_ws);
    bool pos_changed = component_max(abs(hit.pos - trace_result.position_ws)) > (prev_ray_len / 128.0);
    //if (changed || pos_changed)
    {
        // Update reservoir
        ReservoirSimple reservior = ReservoirSimple::decode_32b(rw_reservoir_texture[dispatch_id]);
        // scale down previous M by relative radiance diff
        float m_scale = saturate(1.0 - RADIANCE_DIFF_MULTIPLIER * lumi_diff / lumi_prev);
        float M_f32 = select((lumi_prev > 0) && !pos_changed, reservior.M * m_scale , 0.0f);
        reservior.M = uint(floor(M_f32));
        rw_reservoir_texture[dispatch_id] = reservior.encode_32b();
    }

    // Update hit sample
    hit.pos = trace_result.position_ws;
    hit.normal = trace_result.normal_ws;
    rw_hit_pos_normal_texture[dispatch_id] = hit.encode();
    rw_hit_radiance_texture[dispatch_id] = trace_result.radiance;
}
