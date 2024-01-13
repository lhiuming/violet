#include "../brdf.hlsl"
#include "../constants.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../util.hlsl"

#include "config.inc.hlsl"
#include "reservoir.hlsl"

#define IND_SPEC_ENABLE_TEMPORAL_REUSE 1

// Mirror-like surface has near-zero noise, so we can use a smaller reservoir size
#define ADAPTIVE_HISTORY_CONFIDENCE_BY_ROUGHNESS 1

// Thie is necessary; using only lumi as target function will just add noise (naive VNDF sampling is better for most cases)
#define IND_SPEC_TARGET_FUNCTION_HAS_BRDF 1

// Enable proper resampling MIS weight (balanced heuristic, approximated with target function) for domain changes due to TAA jitter
#define IND_SPEC_PROPER_RESAMPLING_MIS_WEIGHT 1

#if !IND_SPEC_TARGET_FUNCTION_HAS_BRDF
    #undef IND_SPEC_PROPER_RESAMPLING_MIS_WEIGHT
#endif

GBUFFER_TEXTURE_TYPE prev_gbuffer_color;
Texture2D<float> prev_gbuffer_depth;
GBUFFER_TEXTURE_TYPE gbuffer_color;
Texture2D<float> gbuffer_depth;
// Previous frame's reservoir
Texture2D<uint2> prev_reservoir_texture;
Texture2D<float4> prev_hit_pos_texture;
Texture2D<float3> prev_hit_radiance_texture;
// Current frame's reservoir
RWTexture2D<uint2> rw_reservoir_texture;
RWTexture2D<float4> rw_hit_pos_texture;
RWTexture2D<float3> rw_hit_radiance_texture;

[[vk::push_constant]]
struct PushConstants
{
    uint frame_index;
    uint has_prev_frame;
} pc;

// simplified from:
// ( smith_lambda_GGX( cos_theta, ... ) + 1 ) * cos_theta
float smith_lambda_GGX_plus_one_mul_cos(float cos_theta, float roughness)
{
    float r2 = roughness * roughness;
    float c = cos_theta;
    return ( c + sqrt( (-r2 * c + c) * c + r2 ) ) / 2.0f;
}

struct Sampling
{
    float source_pdf;
    #if IND_SPEC_TARGET_FUNCTION_HAS_BRDF
    float3 brdf_NoL_over_source_pdf;
    #endif
};

Sampling calc_things(float3 view_pos, float3 position_ws, float3 normal, float perceptual_roughness, float3 specular_f0, float3 sample_hit_pos) 
{
    // VNDF sampling
    // (After relfect operator)
    // source_pdf = pdf_H * Jacobian_refl(V, H) 
    //            = ( G1(V) * VoH * D(H) / NoV ) * ( 1 / (4 * VoH) )
    //            = G1(V) * D(H) / (4 * NoV)

    Sampling ret;

    float roughness = perceptual_roughness * perceptual_roughness;
    if (IND_SPEC_R_MIN > 0)
    {
        roughness = max(roughness, IND_SPEC_R_MIN);
    }

    float3 view_dir = normalize(view_pos - position_ws);
    float3 light_dir = normalize(sample_hit_pos - position_ws);
    float3 H = normalize(view_dir + light_dir);

    float NoV = saturate(dot(normal, view_dir));
    float NoH = saturate(dot(normal, H));

    // NoH == 1          -> multiple of 1.0/roughness
    // NoH == 1 & r == 0 -> NaN
    float D = D_GGX(NoH, roughness); // NaN if NoH is one and roughness is zero; min will return the non-NaN side

    #if 0

    // DON'T USE: need clampping for edge cases (NaN)
    float lambda_V = smith_lambda_GGX(NoV, roughness);
    float G1_V = 1.0 / (1.0 + lambda_V);
    source_pdf = G1_V * D / (4 * NoV);

    #else

    // NoV == 0 and r == 0 -> 0
    // NoV == 0            -> r/2
    // r   == 0            -> NoV
    float lambda_pomc = smith_lambda_GGX_plus_one_mul_cos(NoV, roughness); 

    // Original formular:
    // source_pdf = G1_V * D / (4 * NoV);
    //            = D / (4 * NoV * (1 + lambda_V));
    //            = D / (4 * NoV * (1 + lambda_V));
    //            = 0.25 * D / (NoV * (1 + lambda_V));
    ret.source_pdf = 0.25 * D / lambda_pomc;

    #endif

#if IND_SPEC_TARGET_FUNCTION_HAS_BRDF

    float NoL = saturate(dot(normal, light_dir));
    float LoH = saturate(dot(light_dir, H));

    float3 F = F_Schlick(LoH, specular_f0);

    // NoL == 0 -> 1/inf (0?)
    // NoV == 0 -> NaN
    // r   == 0 -> 1
    float G2_over_G1 = smith_G2_over_G1_height_correlated_GGX(NoL, NoV, roughness);
    if (NoL <= 0.0)
    {
        G2_over_G1 = 0.0;
    }
    if (NoV <= 0.0)
    {
        G2_over_G1 = 0.0;
    }

    // Original formular:
    // pdf               = G1 * D / (4 * NoV)
    // brdf              = F * D * G2 / (4 * NoV * NoL);
    // brdf_NoL_over_pdf = bdrf * NoL / pdf
    //                   = F * D * G2 * NoL / (4 * NoV * NoL * pdf)
    //                   = F * G2 / G1
    ret.brdf_NoL_over_source_pdf = F * G2_over_G1;

#endif

    return ret;
}

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    float depth = gbuffer_depth[dispatch_id];

    if (has_no_geometry_via_depth(depth))
    {
        rw_reservoir_texture[dispatch_id] = 0;
        return;
    }

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    const float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);

    //
    // Reproject previous reservoir
    //

    // Reprojection 
    float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(position_ws, 1.0f));
    float3 ndc_reproj = hpos_reproj.xyz / hpos_reproj.w - float3(frame_params.jitter.zw, 0.0f);
    float2 uv_reproj = ndc_reproj.xy * 0.5f + 0.5f;

    // Reject if disocclusion
    bool sample_prev_frame = all(uv_reproj == saturate(uv_reproj)) && bool(pc.has_prev_frame);
    if (sample_prev_frame)
    {
        // TODO try something like PCF and reject softly?
        float reproj_depth = ndc_reproj.z;
        float prev_depth = prev_gbuffer_depth.SampleLevel(sampler_linear_clamp, uv_reproj, 0);
        const float DEPTH_TOLERANCE = 0.001f;
        if (abs(prev_depth - reproj_depth) > DEPTH_TOLERANCE)
        {
            sample_prev_frame = false;
        }
    }

    #if !IND_SPEC_ENABLE_TEMPORAL_REUSE
    sample_prev_frame = false;
    #endif
    
    // Read reservoir from prev frame
    ReservoirSimple reservoir;
    float3 reservoir_hit_pos;
    float3 reservoir_hit_radiance;
    float reservoir_target_function;
    #if IND_SPEC_PROPER_RESAMPLING_MIS_WEIGHT
    GBuffer prev_gbuffer;
    #endif
    if (sample_prev_frame)
    {
        // Nearest neighbor sampling
        // TODO since we can not do filtering, we should try to seach around a small neighbor and test geometry similarity?
        uint2 prev_pos = uint2(uv_reproj * buffer_size); // value are positive; trucated == floor

        // permutation sampling
        // TODO add some per-frame jitter
        // NOTE: not used; adding too much noise if use it naively.
        if (0)
        {
            prev_pos.xy ^= 3;
            prev_pos = clamp(prev_pos, uint2(0, 0), buffer_size - uint2(1, 1));
        }

        reservoir = reservoir_decode_u32(prev_reservoir_texture[prev_pos]);
        float4 prev_hit_pos_xyzw = prev_hit_pos_texture[prev_pos];
        reservoir_hit_pos = prev_hit_pos_xyzw.xyz;
        reservoir_hit_radiance = prev_hit_radiance_texture[prev_pos];
        #if IND_SPEC_TARGET_FUNCTION_HAS_BRDF 
        reservoir_target_function = prev_hit_pos_xyzw.w;
        #else
        reservoir_target_function = luminance(reservoir_hit_radiance);
        #endif

        #if IND_SPEC_PROPER_RESAMPLING_MIS_WEIGHT
        prev_gbuffer = load_gbuffer(prev_gbuffer_color, prev_pos);
        #endif
    }
    else
    {
        reservoir = reservoir_decode_u32(uint2(0, 0));
        reservoir_hit_pos = 0.0;
        reservoir_hit_radiance = 0.0;
        reservoir_target_function = 0.0;
        #if IND_SPEC_PROPER_RESAMPLING_MIS_WEIGHT
        prev_gbuffer = (GBuffer) 0;
        #endif
    }

    //
    // Incorporate new sample
    //

    float3 new_sample_hit_pos = rw_hit_pos_texture[dispatch_id].xyz;
    float3 new_sample_hit_radiance = rw_hit_radiance_texture[dispatch_id];

    GBuffer gbuffer = load_gbuffer(gbuffer_color, dispatch_id);

    // Bound the temporal information to avoid stale sample
    if (true)
    {
        // const uint M_MAX = 20; // [Benedikt 2020]
        // const uint M_MAX = 30; // [Ouyang 2021]
        // const uint M_MAX = 10; // [h3r2tic 2022]
        const uint M_MAX = 16;

#if ADAPTIVE_HISTORY_CONFIDENCE_BY_ROUGHNESS
        // Drop the history if the surface is mirror-like (not much noise).
        uint M_max = uint(float(M_MAX) * gbuffer.perceptual_roughness);
#else
        uint M_max = M_MAX;
#endif
        reservoir.M = min(reservoir.M, M_max);
    }

    // Sampling parameters for current sample on current pixel (brdf)
    Sampling curr_sampling = calc_things(view_params().view_pos, position_ws, gbuffer.normal, gbuffer.perceptual_roughness, get_specular_f0(gbuffer.color, gbuffer.metallic), new_sample_hit_pos);

    // Resampling stuffs
    #if IND_SPEC_TARGET_FUNCTION_HAS_BRDF
    float new_w = luminance(new_sample_hit_radiance * curr_sampling.brdf_NoL_over_source_pdf);
    float new_target_function = new_w * curr_sampling.source_pdf;
    #else
    float new_target_function = luminance(new_sample_hit_radiance);
    // NOTE: we can safely divide by source_pdf because it is not zero with roughness clamping
    float new_w = new_target_function / curr_sampling.source_pdf;
    #endif

    float prev_w = reservoir.W * (reservoir_target_function * float(reservoir.M));

    // Resampling with proper MIS weight: balanced heuristic approximated with target function
    // ref: [ReSTIR course 2023]
    #if IND_SPEC_PROPER_RESAMPLING_MIS_WEIGHT 
    if (reservoir.M > 0)
    {
        // MIS weight for curr sample (new sample)

        // evaluate previous frame's target function on curr sample
        Sampling prev_on_curr = calc_things(frame_params.prev_view_pos.xyz, position_ws, prev_gbuffer.normal, prev_gbuffer.perceptual_roughness, get_specular_f0(prev_gbuffer.color, prev_gbuffer.metallic), new_sample_hit_pos);
        float prev_target_function_on_curr = luminance(new_sample_hit_radiance * prev_on_curr.brdf_NoL_over_source_pdf) * prev_on_curr.source_pdf;

        float curr_m = new_target_function / (new_target_function + prev_target_function_on_curr * float(reservoir.M));
        if (new_target_function == 0.0)
        {
            curr_m = 0.0;
        }
        new_w = curr_m * new_w;

        // MIS weight for prev sample (resevoir sample)

        // evaluate current frame's target function on prev sample
        Sampling curr_on_prev = calc_things(frame_params.view_pos.xyz, position_ws, gbuffer.normal, gbuffer.perceptual_roughness, get_specular_f0(gbuffer.color, gbuffer.metallic), reservoir_hit_pos);
        float curr_target_function_on_prev = luminance(reservoir_hit_radiance * curr_on_prev.brdf_NoL_over_source_pdf) * curr_on_prev.source_pdf;

        float m_num = reservoir_target_function * float(reservoir.M);
        float prev_m = m_num / (m_num + curr_target_function_on_prev);
        if (m_num == 0.0)
        {
            prev_m = 0.0;
        }
        prev_w = prev_m * (curr_target_function_on_prev * reservoir.W);

        // update prev (reservoir) target function under curr domain (view + BRDF)
        reservoir_target_function = curr_target_function_on_prev;
    }
    #endif

    uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

    // update reserviour with the new sample
    const float w_sum = prev_w + new_w;
#if 0
    const float chance_new = new_w / w_sum;
    const bool use_new_sample = lcg_rand(rng_state) < chance_new;
    const bool use_prev_sample = !use_new_sample;
#else
    // This path prefer new sample when both prev_w and new_w are 0. 
    // NOTE: typically equavalent to using `chance = new_w / w_sum`. But when 
    // reflection hit point is dark (e.g. hit a metal surface, but radiance 
    // cache is diffuse only), then we may prefer the up-to-date sample,
    // because it has more useful `hit_pos` value.
    const float chance_prev = prev_w / w_sum; 
    const bool use_prev_sample = lcg_rand(rng_state) < chance_prev;
    const bool use_new_sample = !use_prev_sample;
#endif
    if (use_new_sample)
    {
        // replace the sample
        reservoir_hit_pos = new_sample_hit_pos;
        reservoir_hit_radiance = new_sample_hit_radiance;
        reservoir_target_function = new_target_function;
    }
    reservoir.M += 1;
    if (reservoir_target_function > 0.0)
    {
        reservoir.W = w_sum / (reservoir_target_function * float(reservoir.M));
        #if IND_SPEC_PROPER_RESAMPLING_MIS_WEIGHT 
        // 1/M is in w_sum 
        reservoir.W = w_sum / reservoir_target_function;
        #endif
    }
    else
    {
        reservoir.W = 0.0;
    }

    // store updated reservoir (and selected sample)
    rw_reservoir_texture[dispatch_id] = reservoir_encode_u32(reservoir);
    rw_hit_pos_texture[dispatch_id] = float4(reservoir_hit_pos, reservoir_target_function);
    // only write if necessary (reprojecting history sample instead of just takes the new sample)
    if (use_prev_sample)
    {
        rw_hit_radiance_texture[dispatch_id] = reservoir_hit_radiance;
    }
}