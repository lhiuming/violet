#include "../constants.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"
#include "../util.hlsl"

#include "raytrace.hlsli"
#include "reservoir.hlsl"

Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
StructuredBuffer<Reservoir> prev_reservoir_buffer;
RWStructuredBuffer<Reservoir> rw_temporal_reservoir_buffer;
// RWTexture2D<float4> rw_debug_texture;

struct PushConstants
{
    uint frame_index;
};
[[vk::push_constant]]
PushConstants pc;

[shader("raygeneration")]
void main()
{
    uint2 dispatch_id = DispatchRaysIndex().xy;

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    float depth = gbuffer_depth[dispatch_id];

    // early out if no geometry
    if (has_no_geometry_via_depth(depth))
    {
        uint buffer_index = buffer_size.x * dispatch_id.y + dispatch_id.x;
        rw_temporal_reservoir_buffer[buffer_index] = null_reservoir();
        return;
    }

    // Pixel world position
    float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);

    // Read reservoir from last frame
    float4 prev_hpos = mul(frame_params.prev_view_proj, float4(position_ws, 1.0f));
    float2 prev_ndc = prev_hpos.xy / prev_hpos.w;
    bool sample_prev_frame = all(abs(prev_ndc.xy) < 1.0);

    bool generate_new_sample = true;
    float3 sample_origin_ws;
    float3 sample_dir_ws;
    Reservoir reservoir;
    float3 src_normal; // not used
    if (sample_prev_frame)
    {
        // Shot the same ray in for the selected sample in reservoir

        // Nearest neighbor sampling
        float2 prev_screen_uv = prev_ndc.xy * 0.5f + 0.5f;
        uint2 prev_pos = uint2(prev_screen_uv * buffer_size); // value are positive; trucated
        uint buffer_index = buffer_size.x * prev_pos.y + prev_pos.x;
        reservoir = prev_reservoir_buffer[buffer_index];

        // NOTE: it is possible to sample invalid sample (sky, etc.)
        // TODO maybe just validation and not reproject in this pass. Only reproject in a seperate temporal result pass.
        if (reservoir.M > 0)
        {
            sample_origin_ws = reservoir.z.pixel_pos;
            sample_dir_ws = normalize(reservoir.z.hit_pos - sample_origin_ws);
            generate_new_sample = false;
        }
    }

    // For invalid reprojection (sampling the sky, dis occlusion, etc.), generate a new sample, to keep the similar cost as sample generate frame, and avoid under sampling.
    if (generate_new_sample)
    {
        // Genetate new sample for disocclusion
        sample_origin_ws = position_ws;

        GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);
        uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

        float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
        sample_dir_ws = sample_hemisphere_uniform_with_normal(u, gbuffer.normal);

        src_normal = gbuffer.normal;
    }

    TraceResult hit = trace(sample_origin_ws, sample_dir_ws, true);

    if (generate_new_sample)
    {
        // Make a new reservoir
        RestirSample z = make_restir_sample(sample_origin_ws, src_normal, hit.position_ws, hit.normal_ws, hit.radiance);
        uint M = 1;
        float W = TWO_PI;
        reservoir = init_reservoir(z, M, W);
    }
    else
    {
        // Just replace the sample if too different
        // TODO should blend in natually
        float3 prev_radiance = reservoir.z.hit_radiance;
        float max_radiance = max(max(prev_radiance.x, prev_radiance.y), prev_radiance.z);
        bool changed = any((prev_radiance - hit.radiance) > max_radiance * 0.5f);
        if (changed)
        {
            reservoir.M = 1;
        }

        reservoir.z.pixel_pos = sample_origin_ws;
        reservoir.z.pixel_normal = reservoir.z.pixel_normal; // TODO not used
        reservoir.z.hit_pos = hit.position_ws;
        reservoir.z.hit_normal = hit.normal_ws;
        reservoir.z.hit_radiance = hit.radiance;
    }

    rw_temporal_reservoir_buffer[buffer_size.x * dispatch_id.y + dispatch_id.x] = reservoir;
}
