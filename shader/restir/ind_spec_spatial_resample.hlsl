#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../util.hlsl"

#include "reservoir.hlsl"

Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
Texture2D<uint2> reservoir_texture;
Texture2D<float4> hit_pos_texture;
Texture2D<float3> hit_radiance_texture;
RWTexture2D<float3> rw_lighting_texture;

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
        rw_lighting_texture[dispatch_id.xy] = 0.0;
        return;
    }

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    const float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);

    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);

    float3 hit_pos = hit_pos_texture[dispatch_id.xy].xyz;
    float3 hit_radiance = hit_radiance_texture[dispatch_id.xy];

    ReservoirSimple reservoir = reservoir_decode_u32(reservoir_texture[dispatch_id.xy]);

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