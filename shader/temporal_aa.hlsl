#include "frame_bindings.hlsl"
#include "util.hlsl"

Texture2D<float> gbuffer_depth;
Texture2D<float3> source_texture;
Texture2D<float3> history_texture;
RWTexture2D<float3> rw_target;

struct PushConstants
{
    uint has_history;
};
[[vk::push_constant]]
PushConstants pc;

// ref: https://github.com/TheRealMJP/MSAAFilter/blob/master/MSAAFilter/Resolve.hlsl
float mitchell(float x) {
    const float B = 1 / 3.0f;
    const float C = 1 / 3.0f;
    float y = 0.0f;
    float x2 = x * x;
    float x3 = x * x * x;
    if(x < 1)
        y = (12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B);
    else if (x <= 2)
        y = (-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C);

    return y / 6.0f;
}

float blackman_harris(float x) {
    float x2 = x * x;
    return exp(-2.29 * x2);
}

// modified from: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
// See http://vec3.ca/bicubic-filtering-in-fewer-taps/ for more details
float3 SampleTextureCatmullRom(in Texture2D<float3> tex, in SamplerState linearSampler, in float2 uv, in float2 texSize)
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    float2 samplePos = uv * texSize;
    float2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    float2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    float2 texPos0 = texPos1 - 1;
    float2 texPos3 = texPos1 + 2;
    float2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    float3 result = 0.0f;
    result += tex.SampleLevel(linearSampler, float2(texPos0.x, texPos0.y), 0.0f) * w0.x * w0.y;
    result += tex.SampleLevel(linearSampler, float2(texPos12.x, texPos0.y), 0.0f) * w12.x * w0.y;
    result += tex.SampleLevel(linearSampler, float2(texPos3.x, texPos0.y), 0.0f) * w3.x * w0.y;

    result += tex.SampleLevel(linearSampler, float2(texPos0.x, texPos12.y), 0.0f) * w0.x * w12.y;
    result += tex.SampleLevel(linearSampler, float2(texPos12.x, texPos12.y), 0.0f) * w12.x * w12.y;
    result += tex.SampleLevel(linearSampler, float2(texPos3.x, texPos12.y), 0.0f) * w3.x * w12.y;

    result += tex.SampleLevel(linearSampler, float2(texPos0.x, texPos3.y), 0.0f) * w0.x * w3.y;
    result += tex.SampleLevel(linearSampler, float2(texPos12.x, texPos3.y), 0.0f) * w12.x * w3.y;
    result += tex.SampleLevel(linearSampler, float2(texPos3.x, texPos3.y), 0.0f) * w3.x * w3.y;

    return result;
}

// fast clip modified from: https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
float3 clip_aabb(float3 aabb_center, float3 aabb_half_extent, float3 q /*point*/)
{
    // float3 aabb_center = 0.5 * (aabb_max + aabb_min);
	// float3 aabb_half_extent = 0.5 * (aabb_max - aabb_min);

    const float FLT_EPS = 0.00000001f;

    // note: only clips towards aabb center (but fast!)
    float3 p_clip = aabb_center;
	float3 e_clip = aabb_half_extent + FLT_EPS;

	float3 v_clip = q - p_clip;
	float3 v_unit = v_clip.xyz / e_clip;
	float3 a_unit = abs(v_unit);
	float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

	if (ma_unit > 1.0)
		return p_clip + v_clip / ma_unit;
	else
		return q;// point inside aabb
}

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id : SV_DispatchThreadID) {
    // Morden TAA
    // ref: https://alextardif.com/TAA.html

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    // Neighborhood sampling //

    // Produce a sharpe and stable (un-jittered) image for the new frame, and calculate neighborhood variance.

    // subsample offset in pixel space, due to jittering
    // jittering moves object by `jitter.xy`, or equally, creats samples that are offseted by -jitter.xy relative to the unjittered center of the pixel.
    float2 sample_offset_pix = - frame_params.jitter.xy * 0.5 * float2(buffer_size.xy);

    // TODO LDS optimization 
    float3 neighbor_accu = 0.0;
    float neighbor_weight_accu = 0.0;
    float3 neighborhood_min = 1e5;
    float3 neighborhood_max = -1e5;
    float3 momentum1 = 0.0;
    float3 momentum2 = 0.0;
    float closest_depth = 0.0;
    int2 closest_depth_pixcoord = int2(0, 0);
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            int2 pixcoord = dispatch_id + int2(x, y);
            pixcoord = clamp(pixcoord, 0, buffer_size.xy - 1);

            float2 subsample_offset = float2(pixcoord - int2(dispatch_id)) + sample_offset_pix;

            // NOTE: Brian Karis used Blackman-Harris; Tomasz Stachowiak and Alex Tardif suggest Mitchell-Netravali.
            float3 neighbor = source_texture[pixcoord];
            float subsample_dist = length(subsample_offset);
            float subsample_weight = mitchell(subsample_dist); // looks sharper
            //float subsample_weight = blackman_harris(subsample_dist);

            neighbor_accu += neighbor * subsample_weight;
            neighbor_weight_accu += subsample_weight;

            neighborhood_min = min(neighborhood_min, neighbor);
            neighborhood_max = max(neighborhood_max, neighbor);

            momentum1 += neighbor;
            momentum2 += neighbor * neighbor;

            // Alex Tardif suggest using closest depth subsample position for reprojection.
            // NOTE: using reversed depth
            float subsample_depth = gbuffer_depth[pixcoord];
            if (subsample_depth > closest_depth) {
                closest_depth = subsample_depth;
                closest_depth_pixcoord = pixcoord;
            }
        }
    }

    // TODO enbale source filtering once we have jitterred subsamples
    // TODO this seems to amplify the propagating bright spot artifacts
#if 1
    // filtered source pixel
    float3 source = neighbor_accu / neighbor_weight_accu;
#else
    float3 source = source_texture[dispatch_id];
#endif

    // Reprojection //

    // TODO we are not generating movtion vector buffers (not having moving objects); so we reproject directly, asumming everying geometry is static.
    float2 motion_vector;
    {
        // motion vector for the closest depth subsample position, as suggested by Alex Tarif.
        float3 position = cs_depth_to_position(closest_depth_pixcoord, buffer_size, closest_depth);
        position = cs_depth_to_position(dispatch_id, buffer_size, gbuffer_depth[dispatch_id]);
        float4 hpos = mul(frame_params.view_proj, float4(position, 1));
        float2 ndc = hpos.xy / hpos.w - frame_params.jitter.xy;
        float4 hpos_reproj = mul(frame_params.prev_view_proj, float4(position, 1));
        float2 ndc_reproj = hpos_reproj.xy / hpos_reproj.w - frame_params.jitter.zw;
        motion_vector = (ndc - ndc_reproj) * 0.5;
    }
    float2 src_uv = (float2(dispatch_id) + 0.5f) / float2(buffer_size); 
    float2 history_uv = src_uv - motion_vector;

    // early out
    bool not_in_last_view = any(history_uv != saturate(history_uv));
    if (not_in_last_view || !pc.has_history)
    {
        rw_target[dispatch_id] = source;
        return;
    }

#if 0
    // TODO this filtering is doing 9-tap color sampling; maybe optimize with gather?
    float3 history = SampleTextureCatmullRom(history_texture, sampler_linear_clamp, history_uv, float2(buffer_size));
#else
    float3 history = history_texture.SampleLevel(sampler_linear_clamp, history_uv, 0);
#endif

    // Clipping //

#if 1
    // clip by neighbornood range
    // NOTE: this is simple and most effecive against lighting changes
    history = clamp(history, neighborhood_min, neighborhood_max);
#endif

#if 1
    // clip by neighborhood variance
    // [Marco Salvi 2016], suggested by Alex Tarif
    // TODO still too much ghosting, espcially not working well when chroma is changing.
    // TODO this can possibly introduce aliasing along dark/bright edges.
    float ONE_OVER_SAMPLE_COUNT = 1.0 / 9.0;
    float GAMMA = 1.0;
    float3 m1_u = momentum1 * ONE_OVER_SAMPLE_COUNT;
    float3 m2_u = momentum2 * ONE_OVER_SAMPLE_COUNT;
    float3 sigma = sqrt(abs(m2_u - m1_u * m1_u));
    //float3 col_min = m1_u - GAMMA * sigma;
    //float3 col_max = m1_u + GAMMA * sigma;
    history = clip_aabb(m1_u, GAMMA * sigma, history);
#endif

    // Blending //

    // linear factors
    const float BLEND_FACTOR = 1.0 / 32.0;
    float src_weight = BLEND_FACTOR;
    float hist_weight = 1.0 - BLEND_FACTOR;

#if 1
    // 'anti-flicker' weighting suggested in the Alex Tarif post
    // TODO maybe use something more akin to the display mapping transform
    {
        float3 src_compressed = source * rcp(max(max(source.r, source.g), source.b) + 1.0);
        float3 hist_compressed = history * rcp(max(max(history.r, history.g), history.b) + 1.0);
        float src_lumi = luminance(src_compressed);
        float hist_lumi = luminance(hist_compressed);
        src_weight *= rcp(1.0 + src_lumi);
        hist_weight *= rcp(1.0 + hist_lumi);
    }
#endif

    float3 result = (source * src_weight + history * hist_weight) / max(src_weight + hist_weight, 1e-5);

#if 0
    // naive NaN stopping
    if (any(isnan(result))) {
        result = float3(0.9, 0.1, 0.9) * 2;
    }
#endif

    rw_target[dispatch_id] = result;
}