#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../util.hlsl"

#define SPECULAR_SUPRESSION 1

Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
Texture2D<float3> hit_pos_texture;
Texture2D<float3> hit_radiance_texture;
RWTexture2D<float3> rw_lighting_texture;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    float depth = gbuffer_depth[dispatch_id];

    if (has_no_geometry_via_depth(depth))
    {
        rw_lighting_texture[dispatch_id.xy] = 0.0;
        return;
    }

    const float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);

    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);

    float3 hit_pos = hit_pos_texture[dispatch_id.xy];
    float3 hit_radiance = hit_radiance_texture[dispatch_id.xy];

    // Evaluate specular component
    float3 lighting;
    {
        float3 specular_f0 = get_specular_f0(gbuffer.color, gbuffer.metallic);
        #if SPECULAR_SUPRESSION
        float perceptual_roughness = max(gbuffer.perceptual_roughness, 0.045f);
        float roughness = perceptual_roughness * perceptual_roughness;
        #else
        float roughness = gbuffer.perceptual_roughness * gbuffer.perceptual_roughness;
        #endif
        float3 selected_dir = normalize(hit_pos - position_ws);
        float3 view_dir = normalize(frame_params.view_pos.xyz - position_ws);
        float3 H = normalize(view_dir + selected_dir);

        float NoL = saturate(dot(gbuffer.normal, selected_dir));
        float NoV = saturate(dot(gbuffer.normal, view_dir));
        float NoH = saturate(dot(gbuffer.normal, H));
        float LoH = saturate(dot(selected_dir, H));

        float D = D_GGX(NoH, roughness);
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
        //G2_over_G1 = select((NoV * NoL) > 0.0,  G2_over_G1, 0.0);
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

        #endif
    }

    rw_lighting_texture[dispatch_id.xy] = lighting;
}