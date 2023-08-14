Texture2D<float3> source;
Texture2D<float3> history;
RWTexture2D<float3> rw_target;

struct PushConstants
{
    uint has_history;
};
[[vk::push_constant]]
PushConstants pc;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id : SV_DispatchThreadID) {
    float3 src = source[dispatch_id];

    float3 blended;
    if (pc.has_history)
    {
        // TODO blend in display linear space
        float3 hsty = history[dispatch_id];
        blended = lerp(hsty, src, 1.0 / 8);
    }
    else
    {
        blended = src;
    }

    rw_target[dispatch_id] = blended;
}