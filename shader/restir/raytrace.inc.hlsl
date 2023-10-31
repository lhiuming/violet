#pragma once 

#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"
#include "../raytrace/geometry_ray.inc.hlsl"
#include "../raytrace/shadow_ray.inc.hlsl"
#include "../sampling.hlsl"

#include "hash_grid.inc.hlsl"

#define RAYTRACE_INC_SAMPLE_WORLD_RADIANCE_CACHE 1

#define RAYTRACE_INC_SAMPLE_SCREEN_RADIANCE_CACHE 1

#define RAYTRACE_INC_FIREFLY_SUPPRESS 1

RaytracingAccelerationStructure scene_tlas;
TextureCube<float4> skycube;
Texture2D<float3> prev_indirect_diffuse_texture;
Texture2D<float> prev_depth_texture;

#ifndef RAYTRACE_INC_HASH_GRID_STORATE_BUFFER
#define RAYTRACE_INC_HASH_GRID_STORATE_BUFFER hash_grid_storage_buffer
StructuredBuffer<HashGridCell> hash_grid_storage_buffer;
#endif

RWStructuredBuffer<HashGridQuery> rw_hash_grid_query_buffer;
RWStructuredBuffer<uint> rw_hash_grid_query_counter_buffer;

// return: miss or not
bool trace_shadow(float3 ray_origin, float3 ray_dir)
{
    ShadowRayPayload payload;
    payload.missed = false;

    RayDesc ray;
    ray.Origin = ray_origin;
    ray.Direction = ray_dir;
    ray.TMin = 0.0005f; // 0.5mm
    ray.TMax = 1000.0f;

    TraceRay(scene_tlas,
        RAY_FLAG_FORCE_OPAQUE // skip anyhit
        | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER // skip closest hit
        | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH // just light source visibility
        ,
        0xff, // uint InstanceInclusionMask,
        0, // uint RayContributionToHitGroupIndex,
        0, // uint MultiplierForGeometryContributionToHitGroupIndex,
        0, // uint MissShaderIndex,
        ray, // RayDesc Ray,
        payload // inout payload_t Payload
    );

    return payload.missed;
}

struct RadianceTraceResult {
    float3 position_ws;
    float3 normal_ws;
    float3 radiance;
};

// The Trace function, used by sample generation pass and sample validation pass.
RadianceTraceResult trace_radiance(float3 ray_origin, float3 ray_dir, uint has_prev_frame) 
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
        // Shade the hit point //
        float3 diffuse_rho = get_diffuse_rho(payload.base_color, payload.metallic);

        // Directl lighting (diffuse only)
        float3 sun_dir = frame_params.sun_dir.xyz;
        float3 sun_inten = frame_params.sun_inten.rgb;
        bool missed = trace_shadow(payload.position_ws, sun_dir);
        if (missed) 
        {
            // SPECULAR_SUPRESSION
            float perceptual_roughness = max(payload.perceptual_roughness, 0.045f);

            #if RAYTRACE_INC_FIREFLY_SUPPRESS
            perceptual_roughness = max(perceptual_roughness, 0.5f);
            #endif

            float NoL = saturate(dot(payload.normal_ws, sun_dir));
            float3 specular_f0 = get_specular_f0(payload.base_color, payload.metallic);
            float3 direct_lighting = eval_GGX_Lambertian(-ray_dir, sun_dir, payload.normal_ws, perceptual_roughness, diffuse_rho, specular_f0) * NoL * sun_inten;
            float3 direct_diffuse = diffuse_rho * (Fd_Lambert() * NoL) * sun_inten;
            if (0) {
                radiance = direct_diffuse;
            }
            else {
                radiance = direct_lighting;
            }
        }
        else 
        {
            radiance = 0.0f;
        }

        bool sample_world_radiance_cache = true;

        #if RAYTRACE_INC_SAMPLE_SCREEN_RADIANCE_CACHE

        // Indirect lighting from screen (diffuse only)
        if (bool(has_prev_frame)) { 
            float4 prev_hpos = mul(frame_params.prev_view_proj, float4(payload.position_ws, 1.0f));
            float2 ndc = prev_hpos.xy / prev_hpos.w - frame_params.jitter.zw;
            float reproj_depth = prev_hpos.z / prev_hpos.w;

            bool in_view = all(abs(ndc.xy) < 1.0);
            if (in_view)
            {
                uint2 buffer_size;
                prev_depth_texture.GetDimensions(buffer_size.x, buffer_size.y);

                // nearest neighbor sampling
	            float2 prev_screen_uv = ndc.xy * 0.5f + 0.5f;
                uint2 prev_pixcoord = uint2(floor(prev_screen_uv * buffer_size));
                float prev_depth_value = prev_depth_texture[prev_pixcoord];

                // TODO compare in world space? Currently this cause a lot light leaking in micro geometry details (where typically AO should works on).
                float DEPTH_TOLERANCE = 0.0001f; 
                if (abs(reproj_depth - prev_depth_value) < DEPTH_TOLERANCE) {
                    radiance += diffuse_rho * prev_indirect_diffuse_texture[prev_pixcoord].rgb;
                    sample_world_radiance_cache = false;
                }
            }
        } 

        #endif

        #if RAYTRACE_INC_SAMPLE_WORLD_RADIANCE_CACHE

        // Indirect lighting from world space cache (fallback from screen)
        if (sample_world_radiance_cache)
        {
            // TODO jittering 
            VertexDescriptor vert = VertexDescriptor::create(payload.position_ws, ray.Direction);

            // try to fetch radiacne 
            HashGridCell cell;
            uint cell_addr;
            if ( hash_grid_find(RAYTRACE_INC_HASH_GRID_STORATE_BUFFER, vert.hash(), vert.checksum(), /*out*/ cell, /*out*/ cell_addr) )
            {
                radiance += diffuse_rho * cell.radiance;
            }

            // Allocate a slot to store the query
            uint query_index;
            InterlockedAdd(rw_hash_grid_query_counter_buffer[0], 1, query_index);

            // Record the query vertex (to be used in a cache update pass)
            if (query_index < HASH_GRID_MAX_NUM_QUERIES)
            {
                rw_hash_grid_query_buffer[query_index] = HashGridQuery::create(vert.hash(), vert.checksum(), payload.position_ws, payload.normal_geo_ws); 
            }
        }

        #endif
    }

    RadianceTraceResult ret;
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