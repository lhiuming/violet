#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../util.hlsl"

Texture2D<float> depth_texture;
GBUFFER_TEXTURE_TYPE gbuffer_texture;
Texture2D<float3> diff_source_texture;
Texture2D<float3> spec_source_texture;

Texture2D<float> prev_depth_texture;
GBUFFER_TEXTURE_TYPE prev_gbuffer_texture;
Texture2D<float3> diff_history_texture;
Texture2D<float3> spec_history_texture;
Texture2D<float4> moments_history_texture;
Texture2D<uint> history_len_texture;

// TODO pack diffuse and spec to a uint2 texture
RWTexture2D<float3> rw_diff_texture; 
RWTexture2D<float3> rw_spec_texture;
RWTexture2D<float4> rw_moments_texture; // xy: 1st moment; zw: 2nd moment
RWTexture2D<uint> rw_history_len_texture;
// TODO pack to uint ?
RWTexture2D<float2> rw_variance_texture; // x: diffuse variance, y: specular variance

[[vk::push_constant]]
struct PC 
{
    float2 buffer_size;
    uint has_history;
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

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) 
{
    // early out
    float depth = depth_texture[dispatch_id];
    if (has_no_geometry_via_depth(depth))
    {
        // TODO fill zeros
        return;
    }

    const float2 buffer_size = pc.buffer_size;

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
    float4 moments_history;
    float history_len;
    bool valid = all(history_uv == saturate(history_uv)) && bool(pc.has_history);
    if (valid)
    {
        float2 pix_coord_center = history_uv * buffer_size - 0.5;
        float2 pix_coord_frac = frac(pix_coord_center);
        int2 pix_coord_base = int2(pix_coord_center); // truncate ~= floor

        // NOTE: SVGF code sample uses (linear) depth instead of reprojected (linaer) depth 
        GeometryTest test = GeometryTest::create(reproj_depth, load_gbuffer(gbuffer_texture, dispatch_id));

        // 2x2 bilinear filter with geometry test
        float3 diff_sum = 0.0;
        float3 spec_sum = 0.0;
        float4 moments_sum = 0.0;
        float history_len_sum = 0.0;
        float weight_sum = 0.0;
        float2x2 weights = bilinear_weights(pix_coord_frac);
        for (int x = 0; x < 2; ++x)
        {
            for (int y = 0; y < 2; ++y)
            {
                int2 pix_coord = pix_coord_base + int2(x, y);
                bool coord_valid = all(pix_coord == clamp(pix_coord, int2(0, 0), int2(buffer_size) - int2(1, 1)));

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
                    history_len_sum += weight * float(history_len_texture[pix_coord]);
                }
            }
        }

        valid = weight_sum > 0.015625; // (1/8)^2
        if (valid)
        {
            diff_history = diff_sum / weight_sum;
            spec_history = spec_sum / weight_sum;
            moments_history = moments_sum / weight_sum;
            history_len = history_len_sum / weight_sum;
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
        moments_history = float4(0.0, 0.0, 0.0, 0.0);
        history_len = 0.0f;
    }
    history_len += 1.0f;

    float3 diff_source = diff_source_texture[dispatch_id];
    float3 spec_source = spec_source_texture[dispatch_id];
    float4 moments_source;
    moments_source.xy = float2(luminance(diff_source), luminance(spec_source));
    moments_source.zw = moments_source.xy * moments_source.xy;

    // Blend ("Integrate")
    const float BLEND_FACTOR = 1.0 / 16.0f;
    // linearly blend before gathering a long history (1/BLEND_FACTOR)
    float alpha = max(BLEND_FACTOR, 1.0 / float(history_len));
    float3 diff_filtered = lerp(diff_history, diff_source, alpha);
    float3 spec_filtered = lerp(spec_history, spec_source, alpha);
    float4 moments_filtered = lerp(moments_history, moments_source, alpha);

    // Pre-compute variance
    float2 variance = max(moments_filtered.zw - moments_filtered.xy * moments_filtered.xy, 0.0);

    rw_diff_texture[dispatch_id] = diff_filtered;
    rw_spec_texture[dispatch_id] = spec_filtered;
    rw_moments_texture[dispatch_id] = moments_filtered;
    rw_history_len_texture[dispatch_id] = uint(history_len); // truncate
    rw_variance_texture[dispatch_id] = variance;
}