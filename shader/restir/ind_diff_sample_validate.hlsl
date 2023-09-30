#include "../constants.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"
#include "../util.hlsl"

#include "raytrace.inc.hlsl"
#include "reservoir.hlsl"

#define DO_VALIDATION 1

RWStructuredBuffer<Reservoir> rw_prev_reservoir_buffer;
// RWTexture2D<float4> rw_debug_texture;

struct PushConstants
{
    // uint frame_index;
};
[[vk::push_constant]]
PushConstants pc;

[shader("raygeneration")]
void main()
{
#if !DO_VALIDATION
    return;
#endif

    uint2 dispatch_id = DispatchRaysIndex().xy;

    uint2 buffer_size;
    prev_depth.GetDimensions(buffer_size.x, buffer_size.y);

    // Read sample (from reservoir) to validate
    // TODO samples can be shared among reservoirs; maybe use a separate buffer for samples, and validate it without duplication
    Reservoir reservoir = rw_prev_reservoir_buffer[buffer_size.x * dispatch_id.y + dispatch_id.x];

    // early out if no sample to validate reservoir (sky pixel, etc.)
    if (reservoir.M == 0)
    {
        // uint buffer_index = buffer_size.x * dispatch_id.y + dispatch_id.x;
        // rw_prev_reservoir_buffer[buffer_index] = null_reservoir();
        return;
    }

    float3 sample_origin_ws = reservoir.z.pixel_pos;
    float3 sample_dir_ws = normalize(reservoir.z.hit_pos - sample_origin_ws);

    // Trace for up-to-date radiance
    RadianceTraceResult hit = trace_radiance(sample_origin_ws, sample_dir_ws, true);

    // Just replace the sample if too different
    // TODO should blend in natually
    float3 prev_radiance = reservoir.z.hit_radiance;
    float max_radiance = component_max(prev_radiance);
    bool changed = any(abs(prev_radiance - hit.radiance) > max_radiance * 0.5f);
    if (changed)
    {
        reservoir.M = 1;
        reservoir.W = 1.0 / TWO_PI;
    }

    // Update hit
    reservoir.z.hit_pos = hit.position_ws;
    reservoir.z.hit_normal = hit.normal_ws;
    reservoir.z.hit_radiance = hit.radiance;

    rw_prev_reservoir_buffer[buffer_size.x * dispatch_id.y + dispatch_id.x] = reservoir;
}
