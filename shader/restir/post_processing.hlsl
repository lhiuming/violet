Texture2D<float3> src_color_buffer;
RWTexture2D<float4> rw_target_buffer;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) {
    float3 color = src_color_buffer[dispatch_id.xy];

    // hack tone mapping
    #if 1
    float old_lumi = dot(color, float3(0.2126, 0.7152, 0.0722));
    float boost = 2.f;
    float lumi = boost * pow(old_lumi, 0.55);
    color *= lumi / old_lumi;
    #endif

    rw_target_buffer[dispatch_id.xy] = float4(color, 1.0f);
}