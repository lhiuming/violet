#include "scene_bindings.hlsl"
RaytracingAccelerationStructure rayTracingScene;
RWTexture2D<float4> rw_color;

struct Payload {
    bool missed;
};

[shader("raygeneration")]
void raygen() {
    uint3 dispatch_ray_index = DispatchRaysIndex();
    uint3 dispatch_ray_dim = DispatchRaysDimensions();

    float2 uv = (dispatch_ray_index.xy + 0.5f) / dispatch_ray_dim.xy;
    float3 dir = normalize(view_params.view_ray_top_left + uv.x * view_params.view_ray_right_shift + uv.y * view_params.view_ray_down_shift);

    Payload payload;
    payload.missed = false;
    RayDesc ray;
    ray.Origin = view_params.view_pos;
    ray.Direction = dir;
    ray.TMin = 0.0f;
    ray.TMax = 100.0f;
    TraceRay(rayTracingScene,
            RAY_FLAG_FORCE_OPAQUE // skip anyhit
            | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH // shadow 
            | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // skip closest hit
            0xff, // uint InstanceInclusionMask,
            0, // uint RayContributionToHitGroupIndex,
            0, // uint MultiplierForGeometryContributionToHitGroupIndex,
            0, // uint MissShaderIndex,
            ray, // RayDesc Ray,
            payload // inout payload_t Payload
        );

    float4 color = rw_color[dispatch_ray_index.xy];
    color.r = payload.missed;
    color.a = 1.0f;
    rw_color[dispatch_ray_index.xy] = color;
}

[shader("miss")]
void miss(inout Payload payload)
{
    payload.missed = true;
}