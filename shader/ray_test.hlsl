//RaytracingAccelerationStructure rayTracingScene;
RWTexture2D<float4> rw_color;

/*
struct PayLoad {
    float hit;
};
*/

[shader("raygeneration")]
void raygen() {
    uint3 dispatch_ray_index = DispatchRaysIndex();
    uint3 dispatch_ray_dim = DispatchRaysDimensions();

    float4 c = rw_color[dispatch_ray_index.xy];
    rw_color[dispatch_ray_index.xy] = c + float4(0.01, 0.01, 0.5, 0.0f);

    /*
    Payload payload {
        .hit = 0.0f
    }
    RayDesc ray = RayDesc(
        float3(0, 0, 0),
        float3(0, 0, 1),
        0.0f,
        100.0f
    );
    TraceRay(rayTracingScene,
            0, // uint RayFlags,
            0, // uint InstanceInclusionMask,
            0, // uint RayContributionToHitGroupIndex,
            0, // uint MultiplierForGeometryContributionToHitGroupIndex,
            0, // uint MissShaderIndex,
            ray, // RayDesc Ray,
            payload, // inout payload_t Payload
        );
    */
}