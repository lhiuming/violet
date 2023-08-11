#include "../gbuffer.hlsl"
#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"

Texture2D<uint4> gbuffer_color;
Texture2D<float> shadow_mask_buffer;
Texture2D<float3> indirect_diffuse_buffer;
RWTexture2D<float3> rw_color_buffer;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id : SV_DispatchThreadID) {
    uint4 gbuffer_enc = gbuffer_color[dispatch_id];
    GBuffer gbuffer = decode_gbuffer(gbuffer_enc);

    uint2 buffer_size;
    gbuffer_color.GetDimensions(buffer_size.x, buffer_size.y);

    if (gbuffer.shading_path == 0)
    {
        // Written by skybox shader
        return;
    }

    // Direct lighting
    float direct_atten = shadow_mask_buffer[dispatch_id];
    float3 view_dir = -cs_view_ray_direction(dispatch_id, buffer_size, 0.0f.xx);
    float3 diffuse_rho = get_diffuse_rho(gbuffer);
    float3 specular_f0 = get_specular_f0(gbuffer);
    float NoL = saturate(dot(gbuffer.normal, frame_params.sun_dir.xyz));
    float3 directi_lighting = eval_GGX_Lambertian(view_dir, frame_params.sun_dir.xyz, gbuffer.normal, gbuffer.perceptual_roughness, diffuse_rho, specular_f0) * (NoL * direct_atten) * frame_params.sun_inten.rgb;

    // Indirect lighting
    float3 indirect_diffuse = indirect_diffuse_buffer[dispatch_id];
    float3 indirect_specular = 0.0f;
    float3 indirect_lighting = indirect_diffuse * diffuse_rho + indirect_specular;

    // Output
    rw_color_buffer[dispatch_id] = directi_lighting + indirect_lighting;
}