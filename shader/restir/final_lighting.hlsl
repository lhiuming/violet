#include "../gbuffer.hlsl"
#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"

#include "config.inc.hlsl"

#define SPECULAR_SUPRESSION 1

GBUFFER_TEXTURE_TYPE gbuffer_color;
Texture2D<float> shadow_mask_buffer;
Texture2D<float3> indirect_diffuse_texture;
Texture2D<float3> indirect_specular_texture;
RWTexture2D<float3> rw_color_buffer;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id : SV_DispatchThreadID) {
    GBuffer gbuffer = load_gbuffer(gbuffer_color, dispatch_id);

    if (has_no_geometry(gbuffer))
    {
        // Written by skybox shader
        return;
    }

    uint2 buffer_size = get_gbuffer_dimension_2d(gbuffer_color);

    // Direct lighting
    float direct_atten = shadow_mask_buffer[dispatch_id];
    float3 view_dir = -cs_view_ray_direction(dispatch_id, buffer_size, 0.0f.xx);
#if SPECULAR_SUPRESSION
	// Clamp for cheap specular aliasing under punctual light (Frostbite)
    // ref: https://google.github.io/filament/Filament.html
    // TODO may be disk lighting
    float perceptual_roughness = max(gbuffer.perceptual_roughness, 0.045f);
#else
    float perceptual_roughness = gbuffer.perceptual_roughness;
#endif
    float3 diffuse_rho = get_diffuse_rho(gbuffer.color, gbuffer.metallic);
    float3 specular_f0 = get_specular_f0(gbuffer.color, gbuffer.metallic);
    float NoL = saturate(dot(gbuffer.normal, frame_params.sun_dir.xyz));
    float3 directi_lighting = eval_GGX_Lambertian(view_dir, frame_params.sun_dir.xyz, gbuffer.normal, perceptual_roughness, diffuse_rho, specular_f0) * (NoL * direct_atten) * frame_params.sun_inten.rgb;

    #if DEMODULATE_INDIRECT_SPECULAR_FOR_DENOISER
    float NoV = saturate(dot(gbuffer.normal, view_dir));
    float3 specular_reflectance = max(1e-6, ggx_brdf_integral_approx(NoV, perceptual_roughness * perceptual_roughness, specular_f0));
    #else
    float3 specular_reflectance = 1.0;
    #endif

    // Indirect lighting
    float3 indirect_diffuse = indirect_diffuse_texture[dispatch_id];
    float3 indirect_specular = indirect_specular_texture[dispatch_id];
    float3 indirect_lighting = indirect_diffuse * diffuse_rho + indirect_specular * specular_reflectance;

    // Output
    rw_color_buffer[dispatch_id] = directi_lighting + indirect_lighting;
}