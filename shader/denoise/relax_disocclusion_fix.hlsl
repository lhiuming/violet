#include "../gbuffer.hlsl"
#include "../util.hlsl"

#include "svgf.inc.hlsl"

#define ENABLE_DISOCCLUSION_FIX 1

#define DISOCCLUSION_FIX_FRAME_LEN 4

// frame after disocclusion: 14, 7, 4
#define DISOCCLUSION_FIX_TAP_STRIDE 14

// Enable this to skip thin disocclusion (e.g. 1-pixel wide) to save time.
#define SKIP_THIN_DISOCCLUSION 1

Texture2D<float> depth_texture;
GBUFFER_TEXTURE_TYPE gbuffer_texture;
Texture2D<float> history_len_texture;
Texture2D<float3> diff_texture;
Texture2D<float3> spec_texture;
Texture2D<float2> moments_texture;

RWTexture2D<float3> rw_diff_texture;
RWTexture2D<float3> rw_spec_texture;
RWTexture2D<float2> rw_moments_texture;

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID, uint2 group_id: SV_GroupID, uint local_thread_index: SV_GroupIndex)
{
    uint2 buffer_size;
    history_len_texture.GetDimensions(buffer_size.x, buffer_size.y);

    #if SKIP_THIN_DISOCCLUSION 
    // Remap 8x8 threads as 4x4 quads (with in a group of 8x8 threads)
    // ref: https://github.com/Microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics#quad-wide-shuffle-operations
    uint quad_id_flat = local_thread_index >> 2;
    uint quad_index_flat = local_thread_index & 0x3;
    uint2 quad_id = uint2(quad_id_flat & 0x3, quad_id_flat >> 2);
    uint2 quad_index = uint2(quad_index_flat & 0x1, quad_index_flat >> 1);
    dispatch_id = (group_id << 3) + (quad_id << 1) + quad_index;
    #endif

    float histoty_len = history_len_texture[dispatch_id];

    #if SKIP_THIN_DISOCCLUSION 
    // Access quad members before early out
    float max_hisotry_len = max(histoty_len, QuadReadAcrossX(histoty_len));
    max_hisotry_len = max(max_hisotry_len, QuadReadAcrossY(max_hisotry_len));
    #endif

    // Early out (no geometry)
    if (histoty_len == 0.0f)
    {
        return;
    }

    #if SKIP_THIN_DISOCCLUSION 
    // Use max history_len in the quad instead
    histoty_len = max_hisotry_len;
    #endif

    // Fill disocclusion areas
    if ((histoty_len < (DISOCCLUSION_FIX_FRAME_LEN * HISTORY_LEN_UNIT)) && (ENABLE_DISOCCLUSION_FIX != 0))
    {
        const float depth_center = depth_texture[dispatch_id];
        const GBuffer gbuffer = load_gbuffer(gbuffer_texture, dispatch_id);
        const EdgeStoppingFunc edge_stopping = EdgeStoppingFunc::create(depth_center, gbuffer.fwidth_z, gbuffer.normal)
            .sigma_depth(5.0) // relaxed depth test
            .sigma_normal(32.0); // relaxed normal test

        const int2 center_coord = int2(dispatch_id.xy);

        // 5x5 bilateral filter, using Edge-stopping weights (depth and normal).
        float2 weight_sum = 0.0;
        float3 diff_sum = 0.0;
        float3 spec_sum = 0.0;
        float2 sec_momoents_sum = 0.0;
        const int stride = int((DISOCCLUSION_FIX_TAP_STRIDE * HISTORY_LEN_UNIT) / histoty_len);
        const int radius = 2;
        for (int y = -radius; y <= radius; y++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                int2 coord_offset = int2(x, y) * stride;
                int2 coord_unclampped = center_coord + coord_offset;
                int2 coord = clamp(coord_unclampped, int2(0, 0), int2(buffer_size - 1));
                bool valid_coord = all(coord_unclampped == coord);
                if (valid_coord)
                {
                    // TODO pack normal&depth?
                    float depth = depth_texture[coord];
                    float3 normal = load_gbuffer(gbuffer_texture, coord).normal;
                    float2 weight = edge_stopping.weight(length(float2(coord_offset)), depth, normal, float2(0, 0));

                    float3 diff = diff_texture[coord];
                    float3 spec = spec_texture[coord];
                    float2 sec_moments = moments_texture[coord];

                    weight_sum += weight;
                    diff_sum += diff * weight.x;
                    spec_sum += spec * weight.y;
                    sec_momoents_sum += sec_moments * weight.xy;
                }
            }
        }

        float3 diff = diff_sum / weight_sum.x;
        float3 spec = spec_sum / weight_sum.y;
        float2 sec_moments = sec_momoents_sum / weight_sum.xy;

        rw_diff_texture[dispatch_id] = diff;
        rw_spec_texture[dispatch_id] = spec;
        rw_moments_texture[dispatch_id] = sec_moments;
    }
    else
    {
        // TODO collapse copy using conditional read in next shader?
        rw_diff_texture[dispatch_id] = diff_texture[dispatch_id];
        rw_spec_texture[dispatch_id] = spec_texture[dispatch_id];
        rw_moments_texture[dispatch_id] = moments_texture[dispatch_id];
    }
}