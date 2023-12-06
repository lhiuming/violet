#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../util.hlsl"

#include "config.inc.hlsl"
#include "reservoir.hlsl"

#define IND_SPEC_ENABLE_SPATIAL_REUSE 1

#define MAX_ITERATION 4
#define INIT_RADIUS_FACTOR 0.05
#define MIN_RADIUS 1.5

#define ADAPTIVE_RADIUS 1

GBUFFER_TEXTURE_TYPE gbuffer_color;
Texture2D<float> gbuffer_depth;
Texture2D<uint2> reservoir_texture;
Texture2D<float4> hit_pos_texture;
Texture2D<float3> hit_radiance_texture;
RWTexture2D<float3> rw_lighting_texture;

[[vk::push_constant]]
struct PushConstants
{
    uint frame_index;
} pc;

// simplified from:
// ( smith_lambda_GGX( cos_theta, ... ) + 1 ) * cos_theta
float smith_lambda_GGX_plus_one_mul_cos(float cos_theta, float roughness)
{
    float r2 = roughness * roughness;
    float c = cos_theta;
    return ( c + sqrt( (-r2 * c + c) * c + r2 ) ) / 2.0f;
}

struct Sampling {
    float source_pdf;
    float3 hit_radiance;
    float3 brdf_NoL_over_source_pdf;

    float eval_w()
    {
        return luminance(hit_radiance * brdf_NoL_over_source_pdf);
    }

    float eval_target_function()
    {
        return eval_w() * source_pdf;
    }
};

struct SpecTragetFunction {
    float3 view_pos;
    float3 position_ws;
    float3 normal;
    float3 specular_f0;
    float perceptual_roughness;

    Sampling on_sample(float3 sample_hit_pos, float3 sample_hit_radiance)
    {
    // VNDF sampling
    // (After relfect operator)
    // source_pdf = pdf_H * Jacobian_refl(V, H) 
    //            = ( G1(V) * VoH * D(H) / NoV ) * ( 1 / (4 * VoH) )
    //            = G1(V) * D(H) / (4 * NoV)

    Sampling ret;
    ret.hit_radiance = sample_hit_radiance;

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

#if 1 // IND_SPEC_TARGET_FUNCTION_HAS_BRDF

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
};

SpecTragetFunction new_spec_target_function(
    uint2 pixcoord,
    uint2 buffer_size,
    float depth, 
    GBuffer gbuffer
) {
    SpecTragetFunction ret;
    ret.view_pos = frame_params.view_pos.xyz;
    ret.position_ws = cs_depth_to_position(pixcoord, buffer_size, depth);
    ret.normal = gbuffer.normal;
    ret.specular_f0 = get_specular_f0(gbuffer.color, gbuffer.metallic);
    ret.perceptual_roughness = gbuffer.perceptual_roughness;
    return ret;
}

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    float depth = gbuffer_depth[dispatch_id];
    if (has_no_geometry_via_depth(depth))
    {
        rw_lighting_texture[dispatch_id.xy] = 0.0;
        return;
    }

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    const float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);

    GBuffer gbuffer = load_gbuffer(gbuffer_color, dispatch_id.xy);

    ReservoirSimple reservoir = reservoir_decode_u32(reservoir_texture[dispatch_id.xy]);
    float4 hit_pos_xyzw = hit_pos_texture[dispatch_id.xy];
    float3 hit_pos = hit_pos_xyzw.xyz;
    float3 hit_radiance = hit_radiance_texture[dispatch_id.xy];

    #if IND_SPEC_ENABLE_SPATIAL_REUSE

    // Spatial Resampling //

    uint2 pixcoord_center = dispatch_id.xy;
    SpecTragetFunction center_target_function = new_spec_target_function(pixcoord_center, buffer_size, depth, gbuffer);
    float reservoir_target_function = hit_pos_xyzw.w;

    // Search neighborhood
    float radius = min(buffer_size.x, buffer_size.y) * INIT_RADIUS_FACTOR;
    uint rng_state = lcg_init(dispatch_id, buffer_size, pc.frame_index);
    //float rand_angle = lcg_rand(rng_state) * TWO_PI;
    for (uint i = 0; i < MAX_ITERATION; i++)
    {
        float x, y;
        // Uniform sampling in a disk
        {
            float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
            float r = sqrt(u.x);
            float theta = TWO_PI * u.y;
            x = r * cos(theta);
            y = r * sin(theta);
        }

        int2 offset = int2(float2(x, y) * radius);
        uint2 pixcoord_n = clamp(int2(pixcoord_center) + offset, 0, int2(buffer_size) - 1);

        // skip the center sample
        if (all(pixcoord_n == pixcoord_center))
        {
            continue;
        }

        GBuffer gbuffer_n = load_gbuffer(gbuffer_color, pixcoord_n);
        float depth_n = gbuffer_depth[pixcoord_n];

        // Geometry similarity test
        bool geometrical_diff = false;
        // normal test
        const float NORMAL_ANGLE_TELERANCE = PI * (25.0 / 180.0); //[Ouyang 2021]
        geometrical_diff |= dot(gbuffer.normal, gbuffer_n.normal) < cos(NORMAL_ANGLE_TELERANCE);
        // plane distance test
        float3 position_ws_n = cs_depth_to_position(pixcoord_n, buffer_size, depth_n);
        float dist_to_plane = dot(position_ws_n - position_ws, gbuffer.normal);
        const float PLANE_DISTANCE_TOLERANCE = 0.05;
        geometrical_diff |= abs(dist_to_plane) > PLANE_DISTANCE_TOLERANCE;
        if (geometrical_diff)
        {
#if ADAPTIVE_RADIUS
            radius = max(radius * 0.5, MIN_RADIUS);
#endif
            continue;
        }

        // Jacobian determinant for spatial reuse
        // TODO

        // Merge reservoir
        ReservoirSimple reservoir_n = reservoir_decode_u32(reservoir_texture[pixcoord_n]);
        float4 hit_pos_n_xyzw = hit_pos_texture[pixcoord_n];
        float3 hit_pos_n = hit_pos_n_xyzw.xyz;
        float3 hit_radiance_n = hit_radiance_texture[pixcoord_n];
        float reservoir_target_function_n = hit_pos_n_xyzw.w;

        // Resampling with proper MIS weight: balanced heuristic approximated with target function
        float w_neighb; // w for neighbor
        float w_center; // w for center
        {
            //float center_tf_on_center = center_target_function.on_sample(hit_pos, hit_radiance).eval_target_function(); 
            float center_tf_on_center = reservoir_target_function;

            SpecTragetFunction neighb_target_function = new_spec_target_function(pixcoord_n, buffer_size, depth_n, gbuffer_n);

            // MIS weight for neighbor sample
            //float neighb_tf_on_neighb = neighb_target_function.on_sample(hit_pos_n, hit_radiance_n).eval_target_function();
            float neighb_tf_on_neighb = reservoir_target_function_n;
            float center_tf_on_neighb = center_target_function.on_sample(hit_pos_n, hit_radiance_n).eval_target_function();
            float m_neighb = ( neighb_tf_on_neighb * float(reservoir_n.M) ) / ( neighb_tf_on_neighb * float(reservoir_n.M) + center_tf_on_neighb * float(reservoir.M) );
            if ( neighb_tf_on_neighb * float(reservoir_n.M) == 0.0 )
            {
                m_neighb = 0.0;
            }

            //w_neighb = m_neighb * neighb_target_function.on_sample(hit_pos_n, hit_radiance_n).eval_w();
            w_neighb = m_neighb * neighb_tf_on_neighb * reservoir_n.W;

            // MIS weight for center sample (current selected sample)
            float neighb_tf_on_center = neighb_target_function.on_sample(hit_pos, hit_radiance).eval_target_function();
            float m_center = ( center_tf_on_center * float(reservoir.M) ) / ( center_tf_on_center * float(reservoir.M) + neighb_tf_on_center * float(reservoir_n.M) );
            if ( center_tf_on_center * float(reservoir.M) == 0.0 )
            {
                m_center = 0.0;
            }

            //w_center = m_center * center_target_function.on_sample(hit_pos, hit_radiance).eval_w();
            w_center = m_center * center_tf_on_center * reservoir.W;
        }

        float w_sum = w_center + w_neighb;
        float chance = w_neighb / w_sum;
        if (lcg_rand(rng_state) < chance)
        {
            hit_pos = hit_pos_n;
            hit_radiance = hit_radiance_n;
            reservoir_target_function = reservoir_target_function_n;
        }
        reservoir.M += reservoir_n.M;
        if (reservoir_target_function > 0.0)
        {
            reservoir.W = w_sum / reservoir_target_function; 
        }
        else
        {
            reservoir.W = 0.0;
        }
    }

    #endif

    // Evaluate specular component
    float3 lighting;
    {
        float3 specular_f0 = get_specular_f0(gbuffer.color, gbuffer.metallic);
        float roughness = gbuffer.perceptual_roughness * gbuffer.perceptual_roughness;
        if (IND_SPEC_R_MIN > 0)
        {
            roughness = max(roughness, IND_SPEC_R_MIN);
        }

        float3 selected_dir = normalize(hit_pos - position_ws);
        float3 view_dir = normalize(view_params().view_pos - position_ws);
        float3 H = normalize(view_dir + selected_dir);

        float NoL = saturate(dot(gbuffer.normal, selected_dir));
        float NoV = saturate(dot(gbuffer.normal, view_dir));
        float NoH = saturate(dot(gbuffer.normal, H));
        float LoH = saturate(dot(selected_dir, H));

        float3 F = F_Schlick(LoH, specular_f0);

        #if 0

        // Evaluate the RIS estimator naively 

        float D = D_GGX(NoH, roughness);
        float V = vis_smith_G2_height_correlated_GGX(NoL, NoV, roughness);
        float3 brdf = min(D * V, F32_SIGNIFICANTLY_LARGE) * F;

        lighting = hit_radiance * brdf * NoL * reservoir.W;

        #else

        // Evaluate the RIS estimator with VNDF decomposition

        // NoH == 1          -> multiple of 1.0/roughness
        // NoH == 1 & r == 0 -> NaN
        float D = D_GGX(NoH, roughness);

        // NoV == 0 and r == 0 -> 0
        // NoV == 0            -> r/2
        // r   == 0            -> NoV
        float lambda_pomc = smith_lambda_GGX_plus_one_mul_cos(NoV, roughness); // zero if both NoV and roughness are zero

        // Original formular (pdf_L):
        // source_pdf = G1_V * D / (4 * NoV);
        //            = D / (4 * NoV * (1 + lambda_V));
        //            = D / (4 * NoV * (1 + lambda_V));
        //            = 0.25 * D / (NoV * (1 + lambda_V));
        float vndf = 0.25 * D / lambda_pomc;

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
        // given:
        // brdf = F * D * G2 / (4 * NoV * NoL)
        // brdf_NoL = F * D * G2 / (4 * NoV)
        //          = F * D * G2_over_G1 * G1 / (4 * NoV)
        //          = F * G2_over_G1 * G1 * D / (4 * NoV)
        // , then: 
        // lighting = hit_radiance * brdf * NoL * reservoir.W
        //          = hit_radiance * F * G2_over_G1 * G1_V * D / (4 * NoV) * reservoir.W 
        //          = hit_radiance * F * G2_over_G1 * (G1_V * D / (4 * NoV)) * reservoir.W 
        lighting = hit_radiance * ( (F * G2_over_G1) * (vndf * reservoir.W) );
        //lighting = ( (F * G2_over_G1) * (vndf * reservoir.W) );
        //lighting = vndf * reservoir.W;
        //lighting = reservoir.W;
        //lighting = vndf;
        //lighting = float3(vndf, reservoir.W, 0.0f);

        #if DEMODULATE_INDIRECT_SPECULAR_FOR_DENOISER
        lighting /= max(1e-6, ggx_brdf_integral_approx(NoV, roughness, specular_f0));
        #endif

        #if 0
        if (roughness <= 0.0)
        {
            lighting = float3(0, 0, 10);
        }
        if (NoV <= 0.0)
        {
            lighting = float3(10, 0, 0);
        }
        if (NoL <= 0.0f)
        {
            lighting = float3(0, 10, 0);
        }
        #endif

        #if 0
        if ( (dispatch_id.x/128) % 2 == 0)
        {
            lighting = reservoir.W;
        }
        #endif

        #endif
    }

    rw_lighting_texture[dispatch_id.xy] = lighting;
}