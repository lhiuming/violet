#include "reservoir.hlsl"
#include "../constants.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../frame_bindings.hlsl"
#include "../util.hlsl"

#define MAX_ITERATION 9

Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
StructuredBuffer<Reservoir> reservoir_temporal_buffer;
RWTexture2D<float3> rw_lighting_texture;
RWTexture2D<float4> rw_debug_texture;

[[vk::push_constant]]
struct PushConstants  {
    uint frame_index;
} pc;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) {
    float depth = gbuffer_depth[dispatch_id.xy];

    // early out if not geometry 
    if (has_no_geometry_via_depth(depth))
    {
        return;
    }

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    Reservoir reservoir;
    float w_sum;
    {
        uint index = dispatch_id.y * buffer_size.x + dispatch_id.x;
        reservoir = reservoir_temporal_buffer[index];
        float target_pdf = luminance(reservoir.z.hit_radiance);
        w_sum = reservoir.W * target_pdf * reservoir.M;
    }

    float radius = 32; // px
    uint rng_state = lcg_init(dispatch_id, buffer_size, pc.frame_index);
    for (uint i = 0; i < MAX_ITERATION; i++) {
        // Uniform sampling in a disk
        // TODO use some jittered/rotated constant int2 kernel
        float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
        float x, y;
        {
            float r = sqrt(u.x);
            float theta = TWO_PI * u.y;
            x = r * cos(theta);
            y = r * sin(theta);
        }

        int2 offset = int2(float2(x, y) * radius);
        uint2 sample_pos = clamp(int2(dispatch_id) + offset, int2(0, 0), int2(buffer_size) - 1); 

        // TODO geometry similarity test, and continue
        if (0)
        {
            continue;
        }

        uint sample_index = sample_pos.x + sample_pos.y * buffer_size.x;
        Reservoir r_neighbor = reservoir_temporal_buffer[sample_index];

        // TODO calculate reuse jacobian
        float reuse_jacobian = 1.0f;

        float target_pdf_neighbor = luminance(r_neighbor.z.hit_radiance) / reuse_jacobian;

        // TODO visibility test, and continue
        if (0)
        {
            continue;
        }

        // Merge reservoir (sample reuse)
        {
            if (0) {
                // How can M ever exceed 500? (M is clamped to 30+1 in temporal, and up to 10 merged reservoir after spatial)
                uint M_MAX = 500; // [Ouyang 2021]
                r_neighbor.M = min(r_neighbor.M, M_MAX);
            }

            float w_sum_neighbor = r_neighbor.W * target_pdf_neighbor * r_neighbor.M;
            w_sum += w_sum_neighbor;
            float chance = w_sum_neighbor / w_sum;
            if (lcg_rand(rng_state) < chance) {
                reservoir.z = r_neighbor.z;
            }
            reservoir.M += r_neighbor.M;
        }
    }

    // TODO bias correction? [Ouyang 2021, algo 4]

    // update the W
    {
        float target_pdf = luminance(reservoir.z.hit_radiance); 

        // Ideally we use a branch (like in sample_gen.hlsl) instead of injecting a bit of bias. But it is kind of okay here because we are not using the W for anything else beside multiplying it with the hit radiance.
        target_pdf = max(target_pdf, 1e-7f); 

        reservoir.W = w_sum / (target_pdf * float(reservoir.M));
    }

    // Evaluate the RIS esimator
    float3 lighting;
    {
	    // world position reconstruction from depth buffer
        float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);

        // Normal
        GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);

        float3 selected_dir = normalize(reservoir.z.hit_pos - position_ws);
        float NoL = saturate(dot(gbuffer.normal, selected_dir));
        float brdf = ONE_OVER_PI;
        lighting = reservoir.z.hit_radiance * brdf * NoL * reservoir.W ;
    }

    rw_lighting_texture[dispatch_id] = lighting;
}