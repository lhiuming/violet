#include "../frame_bindings.hlsl"
#include "../util.hlsl"

Texture2D<float> depth_buffer;
Texture2D<float3> diff_history_texture;
Texture2D<float3> spec_history_texture;
//Texture2D<float3> diff_moment_history_texture;
//Texture2D<float3> spec_moment_history_texture;
Texture2D<float3> diff_source_texture;
Texture2D<float3> spec_source_texture;
RWTexture2D<float3> rw_diff_texture;
RWTexture2D<float3> rw_spec_texture;
//RWTexture2D<float3> rw_diff_moment_texture;
//RWTexture2D<float3> rw_spec_moment_texture;


[[vk::push_constant]]
struct PC {
    float2 buffer_size;
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

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) 
{
    // early out
    float depth = depth_buffer[dispatch_id];
    if (has_no_geometry_via_depth(depth))
    {
        // TODO fill zeros
        return;
    }

    const float2 buffer_size = pc.buffer_size;

    // Reproject
    float2 motion_vector;
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
    }
    float2 src_uv = (float2(dispatch_id) + 0.5f) / float2(buffer_size); 
    float2 history_uv = src_uv - motion_vector;

    // Sample
    float3 diff_history;
    float3 spec_history;
    bool valid = all(history_uv == saturate(history_uv));
    if (valid)
    {
        float2 pix_coord_center = history_uv * buffer_size - 0.5;
        float2 pix_coord_frac = frac(pix_coord_center);
        int2 pix_coord_base = int2(pix_coord_center); // truncate ~= floor

        // 2x2 bilinear filter with geometry test
        float3 diff_sum = 0.0;
        float3 spec_sum = 0.0;
        float weight_sum = 0.0;
        float2x2 weights = bilinear_weights(pix_coord_frac);
        for (int x = 0; x < 2; ++x)
        {
            for (int y = 0; y < 2; ++y)
            {
                int2 pix_coord = pix_coord_base + int2(x, y);
                pix_coord = clamp(pix_coord, int2(0, 0), int2(buffer_size) - int2(1, 1));
                float weight = weights[y][x]; // y row, x col

                // TODO test
                weight_sum += weight;
                diff_sum += weight * diff_history_texture[pix_coord];
                spec_sum += weight * spec_history_texture[pix_coord];
            }
        }

        valid = weight_sum > 0.0;
        if (valid)
        {
            diff_history = diff_sum / weight_sum;
            spec_history = spec_sum / weight_sum;
        }
        else
        {
            // TOOD 3x3 filter

        }
    }

    if (!valid)
    {
        // disocclusion
        diff_history = float3(0.0, 0.0, 0.0);
        spec_history = float3(0.0, 0.0, 0.0);
    }

    float3 diff_source = diff_source_texture[dispatch_id];
    float3 spec_source = spec_source_texture[dispatch_id];

    // Blend
    const float BLEND_FACTOR = 1.0 / 16.0f;
    float3 diff_filtered = lerp(diff_history, diff_source, BLEND_FACTOR);
    float3 spec_filtered = lerp(spec_history, spec_source, BLEND_FACTOR);

    rw_diff_texture[dispatch_id] = diff_filtered;
    rw_spec_texture[dispatch_id] = spec_filtered;
}