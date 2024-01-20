#include "../color.inc.hlsl"

#define KERNEL_HALF_WIDTH 1

Texture2D<float3> diff_fast_history_texture;
Texture2D<float3> spec_fast_history_texture;

RWTexture2D<float3> rw_diff_texture;
RWTexture2D<float3> rw_spec_texture;

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    uint2 buffer_size;
    diff_fast_history_texture.GetDimensions(buffer_size.x, buffer_size.y);
    const int2 coord_max = buffer_size - int2(1, 1);

    // Read Neighbourhood
    // TODO load to LDS first?
    const int2 center_coord = int2(dispatch_id);
    float3 diff_m1 = 0;
    float3 diff_m2 = 0;
    float3 spec_m1 = 0;
    float3 spec_m2 = 0;
    for (int y = -KERNEL_HALF_WIDTH; y <= KERNEL_HALF_WIDTH; y++)
    {
        for (int x = -KERNEL_HALF_WIDTH; x <= KERNEL_HALF_WIDTH; x++)
        {
            int2 coord = center_coord + int2(x, y);
            coord = clamp(coord, int2(0, 0), coord_max);

            float3 diff = diff_fast_history_texture[coord];
            float3 spec = spec_fast_history_texture[coord];

            diff = rgb_to_yCoCg(diff);
            spec = rgb_to_yCoCg(spec);

            diff_m1 += diff;
            diff_m2 += diff * diff;

            spec_m1 += spec;
            spec_m2 += spec * spec;
        }
    }
    const float weight_sum = (KERNEL_HALF_WIDTH * 2 + 1) * (KERNEL_HALF_WIDTH * 2 + 1);
    diff_m1 /= weight_sum;
    diff_m2 /= weight_sum;
    spec_m1 /= weight_sum;
    spec_m2 /= weight_sum;

    // Calculate standard deviation
    float3 diff_dev = sqrt(diff_m2 - diff_m1 * diff_m1);
    float3 spec_dev = sqrt(spec_m2 - spec_m1 * spec_m1);

    #if 1
    // Reduce ghosting further
    const float sigma = 0.75;
    diff_dev *= sigma;
    spec_dev *= sigma;
    #endif

    // Clamp (in YCoCg color space)
    float3 diff = rw_diff_texture[dispatch_id];
    float3 spec = rw_spec_texture[dispatch_id];
    diff = rgb_to_yCoCg(diff);
    spec = rgb_to_yCoCg(spec);
    diff = clamp(diff, diff_m1 - diff_dev, diff_m1 + diff_dev);
    spec = clamp(spec, spec_m1 - spec_dev, spec_m1 + spec_dev);

    // Done
    diff = yCoCg_to_rgb(diff);
    spec = yCoCg_to_rgb(spec);
    rw_diff_texture[dispatch_id] = diff;
    rw_spec_texture[dispatch_id] = spec;
}