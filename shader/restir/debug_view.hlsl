Texture2D<float3> color_texture;
RWTexture2D<float3> rw_output_texture;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) {
    float3 input = color_texture[dispatch_id];
    rw_output_texture[dispatch_id] = input;
}