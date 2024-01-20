#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../util.hlsl"

#include "svgf.inc.hlsl"

#define ENABLE_TEMPORAL_FILTER 1

#define DUAL_SOURCE_REPROJECTION_FOR_SPEC 1

// Mirror-like surface has near-zero noise, so we can use a very small history
// length to avoid ghosting, which presents even with dual-source reprojection
// due to bilinar sampling.
#define ADAPTIVE_BLEND_BY_ROUGHNESS 1

Texture2D<float> depth_texture;
GBUFFER_TEXTURE_TYPE gbuffer_texture;
Texture2D<float3> diff_source_texture;
Texture2D<float3> spec_source_texture;
Texture2D<float> spec_ray_len_texture;

Texture2D<float> prev_depth_texture;
GBUFFER_TEXTURE_TYPE prev_gbuffer_texture;
Texture2D<float3> diff_history_texture;
Texture2D<float3> spec_history_texture;
Texture2D<float2> moments_history_texture;
Texture2D<float> history_len_texture;

Texture2D<float3> diff_fast_history_texture;
Texture2D<float3> spec_fast_history_texture;

// TODO pack diffuse and spec to a uint2 texture
RWTexture2D<float3> rw_diff_texture; 
RWTexture2D<float3> rw_spec_texture;
RWTexture2D<float2> rw_moments_texture; // // 2nd moment of: x: diffuse, y: specular
RWTexture2D<float> rw_history_len_texture;

RWTexture2D<float3> rw_diff_fast_history_texture;
RWTexture2D<float3> rw_spec_fast_history_texture;

[[vk::push_constant]]
struct PC 
{
    float2 buffer_size;
    uint has_history;
    uint fast_history;
} pc;

float2x2 bilinear_weights(float2 f)
{
    float b = f.y; // bottom
    float r = f.x; // right
    float t = 1.0 - b; // top
    float l = 1.0 - r; // left
    return float2x2(
        t * l, t * r,
        b * l, b * r
        );
}

struct GeometryTest 
{
    float center_depth;
    float3 center_normal_ws;
    float depth_tol;
    float normal_tol_sq;

    static GeometryTest create(float depth, GBuffer gbuffer)
    {
        float depth_tol = gbuffer.fwidth_z * 4.0 + 1e-4;
        float normal_tol = gbuffer.fwidth_n * 16.0 + 1e-2;
        GeometryTest ret = {
            depth,
            gbuffer.normal,
            depth_tol,
            normal_tol * normal_tol
        };
        return ret;
    }

    bool is_tap_valid(float depth, float3 normal_ws)
    {
        float3 n_diff = normal_ws - center_normal_ws;
        return ( abs(depth - center_depth) < depth_tol ) 
            && ( dot(n_diff, n_diff) < normal_tol_sq );
    }
};

// Dual-source reprojection weight
float dsr_weight(float3 color, float3 mean, float3 deviation)
{
    float dist = luminance(abs(color - mean) / deviation);
    return exp2(-10 * dist);
}

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) 
{
    // early out
    float depth = depth_texture[dispatch_id];
    if (has_no_geometry_via_depth(depth))
    {
        // TODO fill zeros
        rw_history_len_texture[dispatch_id] = 0.0;
        return;
    }

    const float2 buffer_size = pc.buffer_size;
    const int2 buffer_size_int = int2(buffer_size);
    const float2 screen_uv = (float2(dispatch_id) + 0.5f) / float2(buffer_size);

    // TODO only load when needed?
    GBuffer gbuffer = load_gbuffer(gbuffer_texture, dispatch_id);

#if DUAL_SOURCE_REPROJECTION_FOR_SPEC
    // Neighbourhood Clamping
    // TODO LDS optimization? 
    float3 momentum1_spec = 0.0;
    float3 momentum2_spec = 0.0;
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            int2 pixcoord = dispatch_id + int2(x, y);
            pixcoord = clamp(pixcoord, 0, buffer_size_int - 1);
            float3 neighbor_spec = spec_source_texture[pixcoord];
            momentum1_spec += neighbor_spec;
            momentum2_spec += neighbor_spec * neighbor_spec;
        }
    }
    momentum1_spec *= 1.0 / 9.0;
    momentum2_spec *= 1.0 / 9.0;
#endif

    // Virtual Reproject
    // TODO skip if:  motion_vector ~= virtual_motion_vector
    // TODO current virtual reprojection only work with planar surface; try taking curvature into account
#if DUAL_SOURCE_REPROJECTION_FOR_SPEC 
    float3 virtual_spec_history;
    float3 virtual_spec_fast_history;
    {
	    float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_uv * 2.0f - 1.0f, depth, 1.0f));
        float3 position = position_ws_h.xyz / position_ws_h.w;

        float3 view_ray_dir = normalize(position - view_params().view_pos);
        float ray_len = spec_ray_len_texture[dispatch_id];
        float3 virtual_position = position + view_ray_dir * ray_len;

        float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(virtual_position, 1));
        float2 ndc_reproj = hpos_reproj.xy / hpos_reproj.w - frame_params.jitter.zw;
        float2 virtual_history_uv = (ndc_reproj + frame_params.jitter.xy) * 0.5 + 0.5f;

        // Sample prev-frame data
        bool valid = all(virtual_history_uv == saturate(virtual_history_uv)) && bool(pc.has_history);
        if (valid)
        {
            virtual_spec_history = spec_history_texture.SampleLevel(sampler_linear_clamp, virtual_history_uv, 0);
            virtual_spec_fast_history = spec_fast_history_texture.SampleLevel(sampler_linear_clamp, virtual_history_uv, 0);
        }
        else
        {
            // disocclusion
            virtual_spec_history = float3(0.0, 0.0, 0.0);
            virtual_spec_fast_history = float3(0.0, 0.0, 0.0);
        }
    }
#endif

    // Reproject
    float2 motion_vector;
    float reproj_depth;
    {
        // TODO save motion vector in a buffer so that we can reuse thi calculation in different passes

        float3 position;
        {
            float2 screen_uv = (float2(dispatch_id) + 0.5f) / buffer_size;
	        float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_uv * 2.0f - 1.0f, depth, 1.0f));
            position = position_ws_h.xyz / position_ws_h.w;
        }
        float4 hpos = mul(frame_params.view_proj, float4(position, 1));
        float2 ndc = hpos.xy / hpos.w - frame_params.jitter.xy;
        float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(position, 1));
        float2 ndc_reproj = hpos_reproj.xy / hpos_reproj.w - frame_params.jitter.zw;
        motion_vector = (ndc - ndc_reproj) * 0.5;
        reproj_depth = hpos_reproj.z / hpos_reproj.w;
    }
    float2 src_uv = (float2(dispatch_id) + 0.5f) / float2(buffer_size); 
    float2 history_uv = src_uv - motion_vector;

    // Sample prev-frame data
    float3 diff_history;
    float3 spec_history;
    float2 sec_moments_history;
    float history_len;
    float3 diff_fast_history;
    float3 spec_fast_history;
    bool valid = all(history_uv == saturate(history_uv)) && bool(pc.has_history);
    if (valid)
    {
        float2 pix_coord_center = history_uv * buffer_size - 0.5;
        float2 pix_coord_frac = frac(pix_coord_center);
        int2 pix_coord_base = int2(pix_coord_center); // truncate ~= floor

        // NOTE: SVGF code sample uses (linear) depth instead of reprojected (linaer) depth 
        GeometryTest test = GeometryTest::create(reproj_depth, gbuffer);

        // 2x2 bilinear filter with geometry test
        float3 diff_sum = 0.0;
        float3 spec_sum = 0.0;
        float2 moments_sum = 0.0;
        float history_len_sum = 0.0;
        float3 diff_fast_sum = 0.0;
        float3 spec_fast_sum = 0.0;
        float weight_sum = 0.0;
        float2x2 weights = bilinear_weights(pix_coord_frac);
        for (int x = 0; x < 2; ++x)
        {
            for (int y = 0; y < 2; ++y)
            {
                int2 pix_coord = pix_coord_base + int2(x, y);
                bool coord_valid = all(pix_coord == clamp(pix_coord, int2(0, 0), buffer_size_int - int2(1, 1)));

                // TODO pack depth and normal (low precision, e.g. 16 + 16, should be enough)
                float3 normal = GBufferNormal::load(prev_gbuffer_texture, pix_coord).normal;
                bool sample_valid = coord_valid && test.is_tap_valid(prev_depth_texture[pix_coord], normal);

                if (sample_valid)
                {
                    float weight = weights[y][x]; // y row, x col
                    weight_sum += weight;
                    diff_sum += weight * diff_history_texture[pix_coord];
                    spec_sum += weight * spec_history_texture[pix_coord];
                    moments_sum += weight * moments_history_texture[pix_coord];
                    history_len_sum += weight * history_len_texture[pix_coord];
                    if (bool(pc.fast_history))
                    {
                        diff_fast_sum += weight * diff_fast_history_texture[pix_coord];
                        spec_fast_sum += weight * spec_fast_history_texture[pix_coord];
                    }
                }
            }
        }

        //valid = weight_sum > 0.015625; // (1/8)^2
        valid = weight_sum > 0.00390625; // (1/16)^2
        if (valid)
        {
            float w_sum_inv = rcp(weight_sum);
            diff_history = diff_sum * w_sum_inv;
            spec_history = spec_sum * w_sum_inv;
            sec_moments_history = moments_sum * w_sum_inv;
            history_len = history_len_sum * w_sum_inv;
            diff_fast_history = diff_fast_sum * w_sum_inv;
            spec_fast_history = spec_fast_sum * w_sum_inv;
        }
        else
        {
            // TOOD fallback 3x3 filter?
        }
    }

    if (!valid)
    {
        // disocclusion
        diff_history = float3(0.0, 0.0, 0.0);
        spec_history = float3(0.0, 0.0, 0.0);
        sec_moments_history = float2(0.0, 0.0);
        history_len = 0.0f;
        diff_fast_history = float3(0.0, 0.0, 0.0);
        spec_fast_history = float3(0.0, 0.0, 0.0);
    }
    history_len += HISTORY_LEN_UNIT;
    
    float3 diff_source = diff_source_texture[dispatch_id];
    float3 spec_source = spec_source_texture[dispatch_id];
    float2 lumi_source = float2(luminance(diff_source), luminance(spec_source));
    float2 sec_moments_source = lumi_source * lumi_source;

    // Blend ("Integrate")
    const float BLEND_FACTOR_DIFF = 1.0 / 20.0f;
    const float BLEND_FACTOR_FAST = 1.0 / 3.0f;
#if ADAPTIVE_BLEND_BY_ROUGHNESS
    // TODO very heuristic
    float blurness = sqrt(saturate(gbuffer.perceptual_roughness * 6.0));
    const float BLEND_FACTOR_SPEC = lerp(7.0/8.0, BLEND_FACTOR_DIFF, blurness);
    const float BLEND_FACTOR_SPEC_FAST = lerp(1.0, BLEND_FACTOR_FAST, blurness);
#else
    const float BLEND_FACTOR_SPEC = BLEND_FACTOR_DIFF;
    const float BLEND_FACTOR_SPEC_FAST = BLEND_FACTOR_FAST;
#endif
    // linearly blend before gathering a long history (1/BLEND_FACTOR)
    float2 alpha = max(float2(BLEND_FACTOR_DIFF, BLEND_FACTOR_SPEC), HISTORY_LEN_UNIT / history_len);
    float3 diff_filtered = lerp(diff_history, diff_source, alpha.x);
    float3 spec_filtered = lerp(spec_history, spec_source, alpha.y);
    float2 sec_moments_filtered = lerp(sec_moments_history, sec_moments_source, alpha.xy);

    float2 alpha_fast = max(float2(BLEND_FACTOR_FAST, BLEND_FACTOR_SPEC_FAST), HISTORY_LEN_UNIT / (history_len));
    float3 diff_filtered_fast = lerp(diff_fast_history, diff_source, alpha_fast.x);
    float3 spec_filtered_fast = lerp(spec_fast_history, spec_source, alpha_fast.y);

#if DUAL_SOURCE_REPROJECTION_FOR_SPEC
    // Weighted blend [Stachowiak 2018]
    {
        #if 1
        // Clamp with spatial variance
        float3 spec_mean = momentum1_spec;
        float3 spec_dev = max(sqrt(abs(momentum2_spec - spec_mean * spec_mean)), 1e-19);
        #else
        // TODO Clamp with temporal variance ?
        #endif

        float weight_virtual = dsr_weight(virtual_spec_history, spec_mean, spec_dev);
        float weight_history = dsr_weight(spec_history, spec_mean, spec_dev);
        float3 spec_history_dual = 
            weight_virtual * virtual_spec_history + weight_history * spec_history;
        float w_sum = weight_virtual + weight_history;
        spec_history_dual /= max(w_sum, 1e-37);
        spec_filtered = lerp(spec_history_dual, spec_source, alpha.y);

        if (bool(pc.fast_history))
        {
            float weight_virtual = dsr_weight(virtual_spec_fast_history, spec_mean, spec_dev);
            float weight_history = dsr_weight(spec_fast_history, spec_mean, spec_dev);
            float3 spec_fast_history_dual = 
                weight_virtual * virtual_spec_fast_history + weight_history * spec_fast_history;
            float w_sum = weight_virtual + weight_history;
            spec_fast_history_dual /= max(w_sum, 1e-37);
            spec_filtered_fast = lerp(spec_fast_history_dual, spec_source, alpha_fast.y);
        }
    }
#endif

#if ENABLE_TEMPORAL_FILTER
    rw_diff_texture[dispatch_id] = diff_filtered;
    rw_spec_texture[dispatch_id] = spec_filtered;
    rw_moments_texture[dispatch_id] = sec_moments_filtered;
    rw_history_len_texture[dispatch_id] = history_len;
    if (bool(pc.fast_history))
    {
        #if 1
        rw_diff_fast_history_texture[dispatch_id] = diff_filtered_fast;
        rw_spec_fast_history_texture[dispatch_id] = spec_filtered_fast;
        #else
        rw_diff_fast_history_texture[dispatch_id] = diff_source;
        rw_spec_fast_history_texture[dispatch_id] = spec_source;
        #endif
    }
#else
    rw_diff_texture[dispatch_id] = diff_source;
    rw_spec_texture[dispatch_id] = spec_source;
    rw_moments_texture[dispatch_id] = sec_moments_source;
    rw_history_len_texture[dispatch_id] = HISTORY_LEN_UNIT;
    #if ALLOW_FAST_HISTORY
    if (bool(pc.fast_history))
    {
        rw_diff_fast_history_texture[dispatch_id] = diff_source;
        rw_spec_fast_history_texture[dispatch_id] = spec_source;
    }
    #endif
#endif
}