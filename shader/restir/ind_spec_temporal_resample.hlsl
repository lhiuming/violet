#include "../brdf.hlsl"
#include "../constants.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../util.hlsl"

#include "reservoir.hlsl"

#define IND_SPEC_ENABLE_TEMPORAL_REUSE 1

// Thie is necessary; using only lumi as target function will just add noise (naive VNDF sampling is better for most cases)
#define IND_SPEC_TARGET_FUNCTION_HAS_BRDF 1

Texture2D<float> prev_gbuffer_depth;
Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
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
        reservoir = reservoir_decode_u32(uint2(0, 0));
        reservoir_hit_pos = 0.0;
        reservoir_hit_radiance = 0.0;
        reservoir_target_function = 0.0;
    }

    //
    // Incorporate new sample
    //

    float3 new_sample_hit_pos = rw_hit_pos_texture[dispatch_id].xyz;

    // VNDF sampling
    // (After relfect operator)
    // source_pdf = pdf_H * Jacobian_refl(V, H) 
    //            = ( G1(V) * VoH * D(H) / NoV ) * ( 1 / (4 * VoH) )
    //            = G1(V) * D(H) / (4 * NoV)
    float source_pdf;
    float3 brdf_NoL_over_source_pdf;
    {
        GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id]);
        float roughness = gbuffer.perceptual_roughness * gbuffer.perceptual_roughness;
        if (IND_SPEC_R_MIN > 0)
        {
            roughness = max(roughness, IND_SPEC_R_MIN);
        }

        float3 view_dir = normalize(view_params().view_pos - position_ws);
        float3 light_dir = normalize(new_sample_hit_pos - position_ws);
        float3 H = normalize(view_dir + light_dir);

        float NoL = saturate(dot(gbuffer.normal, light_dir));
        float NoV = saturate(dot(gbuffer.normal, view_dir));
        float NoH = saturate(dot(gbuffer.normal, H));

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
        source_pdf = 0.25 * D / lambda_pomc;

        #endif

#if IND_SPEC_TARGET_FUNCTION_HAS_BRDF

        float3 specular_f0 = get_specular_f0(gbuffer.color, gbuffer.metallic);
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
        brdf_NoL_over_source_pdf = F * G2_over_G1;

#endif
    }

    float3 new_sample_hit_radiance = rw_hit_radiance_texture[dispatch_id];

    #if IND_SPEC_TARGET_FUNCTION_HAS_BRDF
    float new_w = luminance(new_sample_hit_radiance * brdf_NoL_over_source_pdf);
    float new_target_function = new_w * source_pdf;
    #else
    float new_target_function = luminance(new_sample_hit_radiance);
    // NOTE: we can safely divide by source_pdf because it is not zero with above clampping
    float new_w = new_target_function / source_pdf;
    #endif

    float w_sum = reservoir.W * (reservoir_target_function * float(reservoir.M));

    uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

    // update reserviour with the new sample
    w_sum += new_w;
    float chance = new_w / w_sum;
    bool replace_sample = (lcg_rand(rng_state) < chance);
    if (replace_sample)
    {
        // replace the sample
        reservoir_hit_pos = new_sample_hit_pos;
        reservoir_hit_radiance = new_sample_hit_radiance;
        reservoir_target_function = new_target_function;
    }
    reservoir.M += 1;
    // NOTE: if reservoir_target_pdf is zero, the sample will not contribute to lighting and will be replaced soon, so we just ignore it.
    if (reservoir_target_function > 0.0) // avoid dviding by zero
    {
        reservoir.W = w_sum / (reservoir_target_function * float(reservoir.M));
    }
    else
    {
        reservoir.W = 0.0;
    }

    // store updated reservoir (and selected sample)
    rw_reservoir_texture[dispatch_id] = reservoir_encode_u32(reservoir);
    rw_hit_pos_texture[dispatch_id] = float4(reservoir_hit_pos, reservoir_target_function);
    // only write if necessary (reprojecting history sample instead of just takes the new sample)
    if (!replace_sample)
    {
        rw_hit_radiance_texture[dispatch_id] = reservoir_hit_radiance;
    }
}