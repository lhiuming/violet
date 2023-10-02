#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../util.hlsl"

#include "reservoir.hlsl"

Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
Texture2D<uint2> reservoir_texture;
Texture2D<float3> hit_pos_texture;
Texture2D<float3> hit_radiance_texture;
RWTexture2D<float3> rw_lighting_texture;

// simplified from:
// ( smith_lambda_GGX( cos_theta, ... ) + 1 ) * cos_theta
float smith_lambda_GGX_plus_one_mul_cos(float cos_theta, float roughness)
{
#if 0
    // Reference
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    float tan_theta = sin_theta / cos_theta;
    float a = rcp(roughness * tan_theta); // Heitz
          a = cos_theta * rcp(roughness * sin_theta); // Jakub
          a = cos_theta * rcp( roughness * sqrt(1.0f - cos_theta*cos_theta) ); // Jakub
    return ( cos_theta + cos_theta * sqrt( 1.0f + rcp(a*a) ) ) / 2.0f;
#else
    // Simplified
    float r2 = roughness * roughness;
    float c = cos_theta;
    return ( c + sqrt( (-r2 * c + c) * c + r2 ) ) / 2.0f;
#endif
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

    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);

    float3 hit_pos = hit_pos_texture[dispatch_id.xy];
    float3 hit_radiance = hit_radiance_texture[dispatch_id.xy];

    ReservoirSimple reservoir = reservoir_decode_u32(reservoir_texture[dispatch_id.xy]);

    // Evaluate specular component
    float3 lighting;
    {
        float3 specular_f0 = get_specular_f0(gbuffer.color, gbuffer.metallic);
        float roughness = gbuffer.perceptual_roughness * gbuffer.perceptual_roughness;

        float3 selected_dir = normalize(hit_pos - position_ws);
        float3 view_dir = normalize(frame_params.view_pos.xyz - position_ws);
        float3 H = normalize(view_dir + selected_dir);

        float NoL = saturate(dot(gbuffer.normal, selected_dir));
        float NoV = saturate(dot(gbuffer.normal, view_dir));
        float NoH = saturate(dot(gbuffer.normal, H));
        float LoH = saturate(dot(selected_dir, H));

        float D = min( D_GGX(NoH, roughness), F32_SIGNIFICANTLY_LARGE);
        float3 F = F_Schlick(LoH, specular_f0);
        float V = vis_smith_G2_height_correlated_GGX(NoL, NoV, roughness);
        float3 brdf = min(D * V, F32_SIGNIFICANTLY_LARGE) * F;

        #if 0

        // faking (from 1 uniform sample)
        float source_pdf = 1.0 / TWO_PI;
        float W = 1.0 / source_pdf;

        lighting = hit_radiance * brdf * (NoL * W);

        #else

        // VNDF sampling
        // (After relfect operator)
        // pdf_L = pdf_H * Jacobian_refl(V, H) 
        //       = ( G1(V) * VoH * D(H) / NoV ) * ( 1 / (4 * VoH) )
        //       = G1(V) * D(H) / (4 * NoV)

        #if 0
        float NoL_safe = max(F32_SIGNIFICANTLY_SMALL, NoL);
        float NoV_safe = max(F32_SIGNIFICANTLY_SMALL, NoL);
        float G2_over_G1 = smith_G2_over_G1_height_correlated_GGX(NoL_safe, NoV_safe, roughness);
        #else
        // Clear to zero if either NoV or NoL is not positive (before saturate)
        float G2_over_G1 = smith_G2_over_G1_height_correlated_GGX(NoL, NoV, roughness);
        G2_over_G1 = sign(NoV * NoL) * min(G2_over_G1, F32_SIGNIFICANTLY_LARGE); // min to stop inf
        #endif

        // Original formular,
        // given:
        // pdf  = G1 * D / (4 * NoV)
        // brdf = F * D * G2 / (4 * NoV * NoL)
        // , then:
        // brdf_NoL_over_pdf = bdrf * NoL / pdf
        //                   = F * D * G2 * NoL / (4 * NoV * NoL * pdf)
        //                   = F * G2 / G1
        float3 brdf_NoL_over_pdf = F * G2_over_G1;

        // Original formular (faking one sample restir),
        // given:
        // W = 1.0 / source_pdf
        // , then:
        // lighting = hit_radiance * brdf * NoL * W
        //          = hit_radiance * brdf * NoL / source_pdf
        lighting = hit_radiance * brdf_NoL_over_pdf;

        float lambda_V = smith_lambda_GGX(NoV, roughness);
        float G1_V = 1.0 / (1.0 + lambda_V);
        D = min( D_GGX(NoH, roughness), F32_SIGNIFICANTLY_LARGE);
        V = vis_smith_G2_height_correlated_GGX(NoL, NoV, roughness);

        //lighting = hit_radiance * brdf * (NoL * reservoir.W);

        float vis = G1_V * D / (4 * NoV);

        float lamb_pomc = smith_lambda_GGX_plus_one_mul_cos(NoV, roughness); // zero if both NoV and roughness are zero
        vis = 0.25 * D / max(lamb_pomc, 1e-6);

        // D can be zero (due to numerially error?); it hapeens when roughness is zero but NoH is not one, which should not happen because it should he a mirror reflection. since H is nor error prone, we decide to trust rougheness.
        vis = select(roughness > 0.0, vis, 1e3);

        // given:
        // brdf = F * D * G2 / (4 * NoV * NoL)
        // brdf_NoL = F * D * G2 / (4 * NoV)
        // brdf_NoL = F * D * G2_over_G1 * G1 / (4 * NoV)
        // brdf_NoL = F * G2_over_G1 * G1 * D / (4 * NoV)
        // , then: 
        // lighting = hit_radiance * brdf * NoL * reservoir.W
        //          = hit_radiance * F * G2_over_G1 * G1_V * D / (4 * NoV) * reservoir.W 
        lighting = hit_radiance * ( (F * G2_over_G1) * (vis * reservoir.W) );

        #endif
    }

    rw_lighting_texture[dispatch_id.xy] = lighting;
}