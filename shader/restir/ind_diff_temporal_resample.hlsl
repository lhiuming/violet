/*
 * Reservoir Temporal Resampling.
 */

#include "../constants.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../util.hlsl"

#include "reservoir.hlsl"

#define IND_DIFF_ENABLE_TEMPORAL_REUSE 1

Texture2D<float> prev_gbuffer_depth;
Texture2D<float> gbuffer_depth;
// New samples
Texture2D<uint2> new_hit_pos_normal_texture;
Texture2D<float3> new_hit_radiance_texture;
// Previous reservoir (with selected sample)
Texture2D<uint> prev_reservoir_texture;
Texture2D<uint2> prev_hit_pos_normal_texture;
Texture2D<float3> prev_hit_radiance_texture;
// Merged reservoir (with selected sample)
RWTexture2D<uint> rw_reservoir_texture; // TODO try to use uint
RWTexture2D<uint2> rw_hit_pos_normal_texture;
RWTexture2D<float3> rw_hit_radiance_texture;

//RWTexture2D<float4> rw_debug_texture;

struct PushConstants
{
    uint frame_index;
    uint has_prev_frame;
    uint has_new_sample;
};
[[vk::push_constant]]
PushConstants pc;

float radiance_reduce(float3 radiance)
{
    return luminance(radiance);
    //return dot(radiance, 1.0.rrr);
    //return max3(radiance);
}

// Put reservior and sample together
struct WorkingReservoir
{
    uint M;
    float W;
    float3 hit_radiance;
    uint2 hit_pos_normal_encoded;
};

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    const float depth = gbuffer_depth[dispatch_id.xy];

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    // early out if no geometry
    if (has_no_geometry_via_depth(depth))
    {
        // Set null reservoir to avoid stale/undefined sample get reused in spatial pass.
        rw_reservoir_texture[dispatch_id] = 0;
    }

    //
    // Reproject previous frame reservoir
    //

    float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);

    float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(position_ws, 1.0f));
    float3 ndc_reproj = hpos_reproj.xyz / hpos_reproj.w - float3(frame_params.jitter.zw, 0.0f);
    float2 uv_reproj = ndc_reproj.xy * 0.5f + 0.5f;

    bool in_view = all(abs(ndc_reproj.xy) < 1.0);
    bool sample_prev_frame = bool(pc.has_prev_frame) && in_view;

    // Reject prev sample if dis-occlusion
    if (sample_prev_frame)
    {
        // TODO try something like PCF and reject softly?
        float reproj_depth = ndc_reproj.z;
        float prev_depth = prev_gbuffer_depth.SampleLevel(sampler_linear_clamp, uv_reproj, 0); // NO, dont lerp the depth!
        const float DEPTH_TOLERANCE = 0.001f;
        if (abs(prev_depth - reproj_depth) > DEPTH_TOLERANCE)
        {
            sample_prev_frame = false;
        }
    }

    #if !IND_DIFF_ENABLE_TEMPORAL_REUSE
    sample_prev_frame = false;
    #endif

    WorkingReservoir reservoir;
    if (sample_prev_frame)
    {
        // Read reservoir from prev frame

        // Nearest neighbor sampling
        // TODO since we can not do filtering, we should try to seach around a small neighbor?
        uint2 prev_pos = uint2(uv_reproj * buffer_size); // value are positive; trucated == floor

        // permutation sampling
        // TODO add some per-frame jitter
        // NOTE: not used; adding too much noise if use it naively.
        if (0)
        {
            prev_pos.xy ^= 3;
            prev_pos = clamp(prev_pos, uint2(0, 0), buffer_size - uint2(1, 1));
        }

        uint prev_rsv_encoded = prev_reservoir_texture[prev_pos];
        ReservoirSimple rsv_simple = ReservoirSimple::decode_32b(prev_rsv_encoded);
        reservoir.M = rsv_simple.M;
        reservoir.W = rsv_simple.W;
        reservoir.hit_radiance = prev_hit_radiance_texture[prev_pos];
        // TODO can read it later
        reservoir.hit_pos_normal_encoded = prev_hit_pos_normal_texture[prev_pos];

        // Discard invalid pixels (sky box, etc.)
        // TODO use depth-compare result, with multi-tap depth test
        if (reservoir.M == 0)
        {
            // guard against Inf and NaN
            reservoir.hit_radiance = 0.0f;
            reservoir.hit_pos_normal_encoded = 0;
        }

        // Bound the temporal information to avoid stale sample
        if (1)
        {
            // const uint M_MAX = 20; // [Benedikt 2020]
            // const uint M_MAX = 30; // [Ouyang 2021]
            // const uint M_MAX = 10; // [h3r2tic 2022]
            const uint M_MAX = 16;
            reservoir.M = min(reservoir.M, M_MAX);
        }
    }
    else
    {
        // no reservoir
        reservoir.M = 0;
        reservoir.W = 0.0;
        reservoir.hit_radiance = 0.0;
        reservoir.hit_pos_normal_encoded = 0;
    }

    //
    // Incorporate new sample
    //

    // NOTE: not new sample in sample validation frame
    if (bool(pc.has_new_sample))
    {
        float reservoir_target_pdf = radiance_reduce(reservoir.hit_radiance);
        float prev_w = reservoir.W * reservoir_target_pdf * float(reservoir.M);

        float3 new_hit_radiance = new_hit_radiance_texture[dispatch_id.xy];
        float new_target_pdf = radiance_reduce(new_hit_radiance);
        float new_w = new_target_pdf * TWO_PI; // source_pdf = 1 / TWO_PI;

        uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

        // update reservoir with new sample
        float w_sum = prev_w + new_w;
        float chance = new_w / w_sum;
        if (lcg_rand(rng_state) < chance)
        {
            reservoir.hit_pos_normal_encoded = new_hit_pos_normal_texture[dispatch_id.xy];
            reservoir.hit_radiance = new_hit_radiance;
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
            // NOTE: if reservoir_target_pdf is 0, it may not come from new_target_pdf (chance will be 0 or NaN in that case), therefore previously target_pdf (and w_sum) must be 0. Then new_target_pdf must be 0 too (otherwise chance is 1).
            reservoir.W = 1.0f / float(reservoir.M);
        }
    }

    ReservoirSimple rsv_simple = { reservoir.M, reservoir.W };
    rw_reservoir_texture[dispatch_id] = rsv_simple.encode_32b();
    rw_hit_pos_normal_texture[dispatch_id] = reservoir.hit_pos_normal_encoded;
    rw_hit_radiance_texture[dispatch_id] = reservoir.hit_radiance;

#if 0
    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);
    float3 selected_dir = normalize(reservoir.z.hit_pos - position_ws);
    float NoL = saturate(dot(gbuffer.normal, selected_dir));
    float brdf = ONE_OVER_PI;
    float3 RIS_estimator_diffuse = reservoir.z.hit_radiance * brdf * NoL * reservoir.W ;
    rw_debug_texture[dispatch_id] = float4(RIS_estimator_diffuse, 1.0f);
#endif
}
