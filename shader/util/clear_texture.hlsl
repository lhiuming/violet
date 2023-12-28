RWTexture2D<uint> rw_texture;

[[vk::push_constant]]
struct PushConstants
{
    uint2 size;
} pc;

[numthreads(16, 16, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    if (all(dispatch_id < pc.size))
    {
        rw_texture[dispatch_id] = 0;
    }
}