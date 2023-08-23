#include "../brdf.hlsl"
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
RWTexture2D<float4> rw_debug_texture;

struct PushConstants
{
    uint frame_index;
    uint has_prev_frame;
    //    uint accumulated_count;
};
[[vk::push_constant]]
PushConstants pc;

[shader("raygeneration")]
void main()
{
    uint2 dispatch_id = DispatchRaysIndex().xy;
    float depth = gbuffer_depth[dispatch_id.xy];

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    // early out if no geometry
    if (has_no_geometry_via_depth(depth))
    {
        // Set null reservoir to avoid stale sample during temporal accumulation
        uint buffer_index = buffer_size.x * dispatch_id.y + dispatch_id.x;
        rw_temporal_reservoir_buffer[buffer_index] = null_reservoir();
        return;
    }

    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);

    // world position reconstruction from depth buffer
    float depth_error = 1.f / 16777216.f; // 24 bit unorm depth
    float2 screen_pos = (dispatch_id.xy + 0.5f) / float2(buffer_size);
    float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_pos * 2.0f - 1.0f, depth + depth_error, 1.0f));
    float3 position_ws = position_ws_h.xyz / position_ws_h.w;

    uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

    // Generate Sample Point

    // uniform hemisphere sampling
    // TODO blue noise
    float3 sample_dir;
    {
        float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
        sample_dir = sample_hemisphere_uniform_with_normal(u, gbuffer.normal);
    }

    // Rrace
    TraceResult trace_result = trace(position_ws, sample_dir, pc.has_prev_frame);

    const RestirSample new_sample = make_restir_sample(
        position_ws,
        gbuffer.normal,
        trace_result.position_ws,
        trace_result.normal_ws,
        trace_result.radiance);

    // Reservoir Temporal Resampling

    float4 prev_hpos = mul(frame_params.prev_view_proj, float4(position_ws, 1.0f));
    float2 prev_ndc = prev_hpos.xy / prev_hpos.w;
    bool in_view = all(abs(prev_ndc.xy) < 1.0);
    bool sample_prev_frame = pc.has_prev_frame && in_view;

    Reservoir reservoir;
    if (sample_prev_frame)
    {
        // Read reservoir from prev frame

        // Nearest neighbor sampling
        float2 prev_screen_uv = prev_ndc.xy * 0.5f + 0.5f;
        uint2 prev_pos = uint2(prev_screen_uv * buffer_size); // value are positive; trucated == floor
        if (0)
        {
            // permutation sampling
            // TODO add some per-frame jitter
            prev_pos.xy ^= 3;
            prev_pos = clamp(prev_pos, uint2(0, 0), buffer_size - uint2(1, 1));
        }
        uint buffer_index = buffer_size.x * prev_pos.y + prev_pos.x;
        reservoir = prev_reservoir_buffer[buffer_index];

        // Bound the temporal information to avoid stale sample
        if (1)
        {
            // const uint M_MAX = 20; // [Benedikt 2020]
            const uint M_MAX = 30; // [Ouyang 2021]
            // const uint M_MAX = 10; // [h3r2tic 2022]
            reservoir.M = min(reservoir.M, M_MAX);
        }

        float reservoir_target_pdf = luminance(reservoir.z.hit_radiance);
        float w_sum = reservoir.W * reservoir_target_pdf * float(reservoir.M);

        float new_target_pdf = luminance(new_sample.hit_radiance);
        float new_w = new_target_pdf * TWO_PI; // source_pdf = 1 / TWO_PI;

        // update reservoir with new sample
        w_sum += new_w;
        float chance = new_w / w_sum;
        if (lcg_rand(rng_state) < chance)
        {
            reservoir.z = new_sample;
            reservoir_target_pdf = new_target_pdf;
        }
        reservoir.M += 1;

        // update the W
        if (reservoir_target_pdf > 0.0) // avoid deviding zero
        {
            reservoir.W = w_sum / (reservoir_target_pdf * float(reservoir.M));
        }
        else
        {
            // if reservoir_target_pdf is 0, it may not come from new_target_pdf (chance will be 0 or NaN in that case), therefore previously target_pdf (and w_sum) must be 0. Then new_target_pdf must be 0 too (otherwise chance is 1).
            reservoir.W = 1.0f / float(reservoir.M);
        }
    }
    else
    {
        // New reservoir
        uint M = 1;
        uint W = TWO_PI; // := w / (target_pdf * M);
        reservoir = init_reservoir(new_sample, M, W);
    }

    // store updated reservoir
    uint buffer_index = buffer_size.x * dispatch_id.y + dispatch_id.x;
    rw_temporal_reservoir_buffer[buffer_index] = reservoir;

#if 0
    float3 selected_dir = normalize(reservoir.z.hit_pos - position_ws);
    float NoL = saturate(dot(gbuffer.normal, selected_dir));
    float brdf = ONE_OVER_PI;
    float3 RIS_estimator_diffuse = reservoir.z.hit_radiance * brdf * NoL * reservoir.W ;
    rw_debug_texture[dispatch_id] = float4(RIS_estimator_diffuse, 1.0f);
#endif
}
