#include "../gbuffer.hlsl"
#include "../util.hlsl"

#include "svgf.inc.hlsl"

#define ENABLE_SPATIAL_FALLBACK_FILTER 1

// Enable this to skip thin disocclusion (e.g. 1-pixel wide).
// Otherwise a bunch of short-history/disoccluded pixels along geometry edges, genreated by TAA jitter, will trigger this spatial filter, cost a non-trivial frame time (~0.2 ms) constantly (even if camera and scene is static).
#define SKIP_THIN_DISOCCLUSION 1

Texture2D<float> depth_texture;
GBUFFER_TEXTURE_TYPE gbuffer_texture;
Texture2D<float> history_len_texture;
Texture2D<float3> diff_texture;
Texture2D<float3> spec_texture;
Texture2D<float4> moments_texture;
Texture2D<float2> variance_texture;

RWTexture2D<float3> rw_diff_texture;
RWTexture2D<float3> rw_spec_texture;
RWTexture2D<float2> rw_variance_texture;

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

    // Early out
    if (histoty_len == 0.0f)
    {
        return;
    }

    #if SKIP_THIN_DISOCCLUSION 
    // Use max history_len in the quad instead
    histoty_len = max_hisotry_len;
    #endif

    // Variance fallback for pixels with short history
    if ((histoty_len < (4.0 * HISTORY_LEN_UNIT)) && (ENABLE_SPATIAL_FALLBACK_FILTER != 0))
    {
        float depth_center = depth_texture[dispatch_id];
        GBuffer gbuffer = load_gbuffer(gbuffer_texture, dispatch_id);
        float3 diff_center = diff_texture[dispatch_id];
        float3 spec_center = spec_texture[dispatch_id];
        float2 lumi_center = float2(luminance(diff_center), luminance(spec_center));
        float2 lumi_var_center = 1.0;
        float sigma_z_boost = 3.0; // ref: SVGF code sample
        EdgeStoppingFunc edge_stopping = EdgeStoppingFunc::create(depth_center, gbuffer.fwidth_z * sigma_z_boost, gbuffer.normal, lumi_center, lumi_var_center);

        const int2 center_coord = int2(dispatch_id.xy);

        // 7x7 box filter, with Edge-stopping weights.
        float2 weight_sum = 0.0;
        float3 diff_sum = 0.0;
        float3 spec_sum = 0.0;
        float4 momoents_sum = 0.0;
        const int radius = 3;
        for (int y = -radius; y <= radius; y++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                int2 coord_unclampped = center_coord + int2(x, y);
                int2 coord = clamp(coord_unclampped, int2(0, 0), int2(buffer_size - 1));
                bool valid_coord = all(coord_unclampped == coord);
                if (valid_coord)
                {
                    float3 diff = diff_texture[coord];
                    float3 spec = spec_texture[coord];
                    float4 moments = moments_texture[coord];

                    float depth = depth_texture[coord];
                    float3 normal = load_gbuffer(gbuffer_texture, coord).normal;
                    float2 lumi = float2(luminance(diff), luminance(spec));
                    float2 weight = edge_stopping.weight(length(float2(x, y)), depth, normal, lumi);

                    weight_sum += weight;
                    diff_sum += diff * weight.x;
                    spec_sum += spec * weight.y;
                    momoents_sum += moments * weight.xyxy;
                }
            }
        }

        float3 diff = diff_sum / weight_sum.x;
        float3 spec = spec_sum / weight_sum.y;
        float4 moments = momoents_sum / weight_sum.xyxy;
        float2 variance = max(moments.zw - moments.xy * moments.xy, 0.0);

        // Variance boost for first frames 
        variance *= (4.0 * HISTORY_LEN_UNIT) / histoty_len;

        rw_diff_texture[dispatch_id] = diff;
        rw_spec_texture[dispatch_id] = spec;
        rw_variance_texture[dispatch_id] = variance;
    }
    else
    {
        // TODO try just R/W on the same texture such that we can skip this?
        // NOTE: 0.08 ms is OK for now
        rw_diff_texture[dispatch_id] = diff_texture[dispatch_id];
        rw_spec_texture[dispatch_id] = spec_texture[dispatch_id];
        rw_variance_texture[dispatch_id] = variance_texture[dispatch_id];
    }
}