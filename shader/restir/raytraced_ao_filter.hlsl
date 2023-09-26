#include "../frame_bindings.hlsl"
#include "../util.hlsl"

Texture2D<float> depth_buffer;
Texture2D<float4> ao_texture;
RWTexture2D<float4> rw_filtered_ao_texture;

[[vk::push_constant]]
struct PushConstants {
} pc;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    float depth = depth_buffer[dispatch_id.xy];

    uint2 buffer_size;
    depth_buffer.GetDimensions(buffer_size.x, buffer_size.y);

    // early out
    if (has_no_geometry_via_depth(depth)) {
        rw_filtered_ao_texture[dispatch_id] = 0.0f;
        return;
    }

    float4 center_ao = ao_texture[dispatch_id.xy];

    // Spatial filtering
    float4 ao_sum = 0.0f;
    float ao_weight_sum = 0.0f;
    float4 col_med[3];
    float4 col_min[3];
    float4 col_max[3];
    for (int y = -1; y <= 1; y++) {
        float4 row[3];
        for (int x = -1; x <= 1; x++) {
            uint2 coord = uint2(min(dispatch_id.xy + int2(x, y), int2(buffer_size)));
            float sample_depth = depth_buffer[coord];

            const float DEPTH_TOLERANCE = 0.0005f;
            if (abs(depth - sample_depth) > DEPTH_TOLERANCE) {
                row[x + 1] = center_ao;
                continue;
            };

            float4 sample_ao = ao_texture[coord];
            ao_sum += sample_ao;
            ao_weight_sum += 1.0f;

            row[x + 1] = sample_ao;
        }

        col_med[y + 1] = clamp(row[0], row[1], row[2]);
        col_min[y + 1] = min3(row[0], row[1], row[2]);
        col_max[y + 1] = max3(row[0], row[1], row[2]);
    }
    float4 A = min3(col_max[0], col_max[1], col_max[2]);
    float4 B = max3(col_min[0], col_min[1], col_min[2]);
    float4 C = clamp(col_med[0], col_med[1], col_med[2]);
    float4 med = clamp(C, A, B);
    float4 min = min3(col_min[0], col_min[1], col_min[2]);
    float4 avg = ao_sum / ao_weight_sum;

    //float4 filtered_ao = avg;
    //float4 filtered_ao = raw;
    //float4 filtered_ao = med;
    //float4 filtered_ao = min;
    float4 filtered_ao = lerp(med, min, 0.5);
    rw_filtered_ao_texture[dispatch_id] = filtered_ao;
}
