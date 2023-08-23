#pragma once 

#include "../frame_bindings.hlsl"
#include "../raytrace/geometry_ray.hlsli"
#include "../sampling.hlsl"

RaytracingAccelerationStructure scene_tlas;
TextureCube<float4> skycube;
Texture2D<float3> prev_color;
Texture2D<float> prev_depth;

struct TraceResult {
    float3 position_ws;
    float3 normal_ws;
    float3 radiance;
};

// The Trace function, used by sample generation pass and sample validation pass.
TraceResult trace(float3 ray_origin, float3 ray_dir, uint has_prev_frame) 
{
    GeometryRayPayload payload;
    payload.missed = false;

    RayDesc ray;
    ray.Origin = ray_origin; // position_ws;
    ray.Direction = ray_dir; // sample_dir;
    ray.TMin = 0.0005f; // 0.5mm
    ray.TMax = 1000.0f;

    TraceRay(scene_tlas,
            RAY_FLAG_FORCE_OPAQUE // skip anyhit
            ,
            0xff, // uint InstanceInclusionMask,
            0, // uint RayContributionToHitGroupIndex,
            0, // uint MultiplierForGeometryContributionToHitGroupIndex,
            0, // uint MissShaderIndex,
            ray, // RayDesc Ray,
            payload // inout payload_t Payload
        );

    // Compute Radiance for the sample point

    float3 radiance = 0.0f;
    if (payload.missed) {
        radiance = skycube.SampleLevel(sampler_linear_clamp, ray.Direction, 0.0f).rgb;
        // Construct a hit point at skybox if miss
        payload.position_ws = ray.Origin + ray.Direction * ray.TMax;
        payload.normal_ws = -ray.Direction;
    }
    else 
    {
        radiance = 0.0;

        // try to read from prev frame color
        //if (pc.has_prev_frame) {
        if (has_prev_frame) {
            float4 prev_hpos = mul(frame_params.prev_view_proj, float4(payload.position_ws, 1.0f));
            float2 ndc = prev_hpos.xy / prev_hpos.w;
            float reproj_depth = prev_hpos.z / prev_hpos.w;

            bool in_view = all(abs(ndc.xy) < 1.0);
            if (in_view)
            {
                uint2 buffer_size;
                prev_depth.GetDimensions(buffer_size.x, buffer_size.y);

                // nearest neighbor sampling
	            float2 prev_screen_uv = ndc.xy * 0.5f + 0.5f;
                uint2 prev_pixcoord = uint2(floor(prev_screen_uv * buffer_size));

                float prev_depth_value = prev_depth[prev_pixcoord];
                float DEPTH_TOLERANCE = 0.0005f; // TODO compare in world space?
                if (abs(reproj_depth - prev_depth_value) < DEPTH_TOLERANCE) {
                    radiance = prev_color[prev_pixcoord].rgb;
                }
            }
        } 

        // TODO world space cache
    }

    TraceResult ret;
    ret.radiance = radiance;
    ret.position_ws = payload.position_ws;
    ret.normal_ws = payload.normal_ws;
    return ret;
}

// Utilities

float3 sample_hemisphere_uniform_with_normal(float2 u, float3 normal) 
{
    float3 L_local = sample_hemisphere_uniform(u);
    float4 rot_from_local = invert_rotation(get_rotation_to_z_from(normal));
    return rotate_point(rot_from_local, L_local);
}