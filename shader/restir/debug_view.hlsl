#include "../gbuffer.hlsl"

GBUFFER_TEXTURE_TYPE gbuffer_texture;
Texture2D<float3> color_texture;
Texture2D<uint3> uint_texture;
RWTexture2D<float3> rw_output_texture;

[[vk::push_constant]]
struct PC {
    uint is_gbuffer;
    uint is_uint;
} pc;

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) {
    float3 col;
    if (bool(pc.is_gbuffer))
    {
        GBuffer gbuffer = load_gbuffer(gbuffer_texture, dispatch_id);
        switch (pc.is_gbuffer)
        {
            case 1: {
                float3 n = gbuffer.normal;
                col = n * 0.5 + 0.5;
                break;
            }
            case 2: {
                col = gbuffer.perceptual_roughness.rrr;
                break;
            }
            default: {
                col = float3(1, 0, 1);
                break;
            }
        }
    }
    else if (bool(pc.is_uint))
    {
        uint3 inp_u = uint_texture[dispatch_id];
        col = inp_u / 255.0;
    }
    else
    {
        col = color_texture[dispatch_id];
    }
    rw_output_texture[dispatch_id] = col;
}