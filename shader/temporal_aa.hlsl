
Texture2D<float3> source;
Texture2D<float3> history;
RWTexture2D<float4> rw_target;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id : SV_DispatchThreadID) {
    float3 src = source[dispatch_id];
    float3 hsty = history[dispatch_id];
    float3 blended = lerp(hsty, src, 0.125);
    rw_target[dispatch_id] = float4(blended, 1.0f);
}