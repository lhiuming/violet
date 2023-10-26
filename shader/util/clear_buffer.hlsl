RWStructuredBuffer<uint> rw_buffer;

[[vk::push_constant]]
struct PushConstants
{
    uint length;
} pc;

[numthreads(128, 1, 1)]
void main(uint dispatch_id: SV_DispatchThreadID)
{
    if (dispatch_id < pc.length)
    {
        rw_buffer[dispatch_id] = 0;
    }
}