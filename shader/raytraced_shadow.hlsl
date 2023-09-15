#include "scene_bindings.hlsl"
#include "frame_bindings.hlsl"
#include "raytrace/shadow_ray.hlsli"

RaytracingAccelerationStructure scene_tlas;
Texture2D<float> gbuffer_depth;
RWTexture2D<float4> rw_shadow;

[shader("raygeneration")]
void main() {
    uint3 dispatch_ray_index = DispatchRaysIndex();
    float depth = gbuffer_depth[dispatch_ray_index.xy];

    // early out if no geometry
    if (depth == 0.0f)
    {
#if 0
        rw_shadow[dispatch_ray_index.xy] = 1.0f;
#endif
        return;
    }

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

	// world position reconstruction from depth buffer
    float depth_error = 1.f / 16777216.f; // 24 bit unorm depth 
    float2 screen_pos = (dispatch_ray_index.xy + 0.5f) / float2(buffer_size);
	float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_pos * 2.0f - 1.0f, depth + depth_error, 1.0f));
	float3 position_ws = position_ws_h.xyz / position_ws_h.w;

    ShadowRayPayload payload;
    payload.missed = false;
    RayDesc ray;
    ray.Origin = position_ws;
    ray.Direction = frame_params.sun_dir.xyz;
    ray.TMin = 0.0005f; // 0.5mm
    ray.TMax = 1000.0f;
    TraceRay(scene_tlas,
            RAY_FLAG_FORCE_OPAQUE // skip anyhit
            | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER // skip closest hit
            | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH // shadow 
            ,
            0xff, // uint InstanceInclusionMask,
            0, // uint RayContributionToHitGroupIndex,
            0, // uint MultiplierForGeometryContributionToHitGroupIndex,
            0, // uint MissShaderIndex,
            ray, // RayDesc Ray,
            payload // inout payload_t Payload
        );

    rw_shadow[dispatch_ray_index.xy] = payload.missed ? 1.0f : 0.0f;
}