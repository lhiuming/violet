#include "display.hlsl"

#define DISPLAY_MAPPING 1

Texture2D<float3> src_color_texture;
RWTexture2D<float4> rw_target_buffer;

struct PushConstants
{
    float exposure_scale;
    //uint debug_view;
};
[[vk::push_constant]]
PushConstants pc;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    float3 scene_referred = src_color_texture[dispatch_id.xy];

    // Exposure adjustment
    scene_referred *= pc.exposure_scale;

    // Display mapping (a.k.a. neutral tonemapping)
#if DISPLAY_MAPPING
    float3 display_referred = applyHuePreservingShoulder(scene_referred);
#else
    float3 display_referred = scene_referred;
#endif

    // Output encoding
    float3 display_encoded = srgb_eotf_inv_float3(display_referred);

    rw_target_buffer[dispatch_id.xy] = float4(display_encoded, 1.0f);
}
