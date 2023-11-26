Texture2D<float3> color_texture;
Texture2D<uint3> uint_texture;
RWTexture2D<float3> rw_output_texture;

[[vk::push_constant]]
struct PC {
    uint is_uint;
} pc;

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) {
    float3 col;
    if (bool(pc.is_uint))
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