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

        // faking (from 1 uniform sample)
        float source_pdf = 1.0 / TWO_PI;
        float W = 1.0 / source_pdf;

        lighting = hit_radiance * brdf * (NoL * W);
    }

    rw_lighting_texture[dispatch_id.xy] = lighting;
}