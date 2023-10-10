#include "../frame_bindings.hlsl"
#include "../util.hlsl"

Texture2D<float> depth_buffer;
Texture2D<float3> history_texture;
Texture2D<float3> source_texture;
RWTexture2D<float3> rw_filtered_texture;

#define BOX_CLIPPING 1
//#define VARIANCE_CLIPPING 1

#define NEIGHBORHOOD_SAMPLING (BOX_CLIPPING || VARIANCE_CLIPPING)

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) 
{
    // early out
    float depth = depth_buffer[dispatch_id];
    if (has_no_geometry_via_depth(depth))
    {
        rw_filtered_texture[dispatch_id] = float3(0, 0, 0);
        return;
    }

    uint2 buffer_size;
    depth_buffer.GetDimensions(buffer_size.x, buffer_size.y);

    #if NEIGHBORHOOD_SAMPLING 

    // Neighborhood Sampling

    float3 source;
    float3 neighborhood_min = 1e7;
    float3 neighborhood_max = -1e7;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            int2 pixcoord = int2(dispatch_id) + int2(x, y);
            pixcoord = clamp(pixcoord, 0, buffer_size - 1);

            float3 neighbor = source_texture[pixcoord];

            neighborhood_min = min(neighborhood_min, neighbor);
            neighborhood_max = max(neighborhood_max, neighbor);

            if (x == 0 && y == 0)
            {
                source = neighbor;
            }
        }
    }

    #else

    float3 source = source_texture[dispatch_id];

    #endif

    // History //

    // Reproject
    float2 motion_vector;
    {
        // TODO save motion vector in a buffer so that we can reuse thi calculation in different passes
        float3 position = cs_depth_to_position(dispatch_id, buffer_size, depth);
        float4 hpos = mul(frame_params.view_proj, float4(position, 1));
        float2 ndc = hpos.xy / hpos.w - frame_params.jitter.xy;
        float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(position, 1));
        float2 ndc_reproj = hpos_reproj.xy / hpos_reproj.w - frame_params.jitter.zw;
        motion_vector = (ndc - ndc_reproj) * 0.5;
    }
    float2 src_uv = (float2(dispatch_id) + 0.5f) / float2(buffer_size); 
    float2 history_uv = src_uv - motion_vector;

    // Sample
    float3 history;
    bool history_in_last_view = all(history_uv == saturate(history_uv));
    if (history_in_last_view)
    {
        history = history_texture.SampleLevel(sampler_linear_clamp, history_uv, 0.0);
    }
    else
    {
        history = source;
    }

    // Blend //

    // Box Clipping
    #if BOX_CLIPPING 
    history = clamp(history, neighborhood_min, neighborhood_max);
    #endif

    // Blend
    // TODO Box clamping to reduce ghosting
    const float BLEND_FACTOR = 1.0 / 16.0f;
    float3 filtered = lerp(history, source, BLEND_FACTOR);

    rw_filtered_texture[dispatch_id] = filtered;
}