Texture2D<float> gbuffer_depth;
RWTexture2D<float4> rwcolor;

[numthreads(8, 8, 1)]
void main(uint2 dipatch_id : SV_DispatchThreadID) {
    
    float depth = gbuffer_depth[dipatch_id];
    float4 color = float4(depth, depth, depth, 1.0f);

    rwcolor[dipatch_id] = color;
}