#include "../gbuffer.hlsl"
#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"

#define SPECULAR_SUPRESSION 1

Texture2D<uint4> gbuffer_color;
Texture2D<float> shadow_mask_buffer;
Texture2D<float3> indirect_diffuse_texture;
Texture2D<float3> indirect_specular_texture;
RWTexture2D<float3> rw_color_buffer;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id : SV_DispatchThreadID) {
    uint4 gbuffer_enc = gbuffer_color[dispatch_id];
    GBuffer gbuffer = decode_gbuffer(gbuffer_enc);

    if (has_no_geometry(gbuffer))
    {
        // Written by skybox shader
        return;
    }

    uint2 buffer_size;
    gbuffer_color.GetDimensions(buffer_size.x, buffer_size.y);

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

    // Indirect lighting
    float3 indirect_diffuse = indirect_diffuse_texture[dispatch_id];
    float3 indirect_specular = indirect_specular_texture[dispatch_id];
    float3 indirect_lighting = indirect_diffuse * diffuse_rho + indirect_specular;

    // Output
    rw_color_buffer[dispatch_id] = directi_lighting + indirect_lighting;
}