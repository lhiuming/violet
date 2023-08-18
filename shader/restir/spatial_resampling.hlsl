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

    // Normal (for similarity test)
    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);

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

        GBuffer gbuffer_n = decode_gbuffer(gbuffer_color[sample_pos]);
        float depth_n = gbuffer_depth[sample_pos];

        // Geometry similarity test
        bool geometrical_diff = false;
        // normal test (within 25 degree) [Ouyang 2021]
        geometrical_diff |= dot(gbuffer.normal, gbuffer_n.normal) < cos(PI * (25.0/180.0));
        // depth test (within 0.05 of depth range) [Ouyang 2021]
        // TODO should be normalized depth
        // TODO wall viewed from grazing-angle will become noisy because depth gradient is large in this case. So a better test should check if two visble sample are on the same surface (plane)
        geometrical_diff |= abs(depth - depth_n) > 0.0005f;
        if (geometrical_diff)
        {
            continue;
        }

        uint sample_index = sample_pos.x + sample_pos.y * buffer_size.x;
        Reservoir reservoir_n = reservoir_temporal_buffer[sample_index];

        // Jacobian determinant for spatial reuse
        float reuse_jacobian = 1.0f;
        if (1)
        {
            // [Ouyang 2021, eq 11]
            // x_1^q-x_2^q
            float3 offset_qq = reservoir_n.z.pixel_pos - reservoir_n.z.hit_pos;
            // x_1^r-x_2^q
            float3 offset_rq = reservoir.z.pixel_pos   - reservoir_n.z.hit_pos;
            float3 hit_normal = reservoir_n.z.hit_normal;
            // cos(phi_2^r)
            float cos_rq = dot(hit_normal, normalize(offset_rq));
            // cos(phi_2^q)
            float cos_qq = dot(hit_normal, normalize(offset_qq));
            reuse_jacobian = (cos_rq / cos_qq) * (dot(offset_qq, offset_qq) / dot(offset_rq, offset_rq));

            // Clamp to avoid fireflies.
            // Jacobian of large value can happen when the reused hit is too close to current surface. Basically we apply the jocobian for the purpose of darkening the sample to avoid over-brightness artifacts at corners, so clampping high value should be fine.
            reuse_jacobian = clamp(reuse_jacobian, 0, 10);
        }

        // NOTE: should be a typo in the ReSTIR GI paper [Ouyang 2021, algo 4]: It should be `target_pdf * jacobian` instead of `target_pdf / jacobian`
        float target_pdf_neighbor = luminance(reservoir_n.z.hit_radiance) * reuse_jacobian;

        // TODO visibility test, and continue
        // At least NoL test is cheap enough?
        if (0)
        {
            continue;
        }

        // Merge reservoir (sample reuse)
        {
            if (0) {
                // How can M ever exceed 500? (M is clamped to 30+1 in temporal, and up to 10 merged reservoir after spatial)
                uint M_MAX = 500; // [Ouyang 2021]
                reservoir_n.M = min(reservoir_n.M, M_MAX);
            }

            float w_sum_neighbor = reservoir_n.W * target_pdf_neighbor * reservoir_n.M;
            w_sum += w_sum_neighbor;
            float chance = w_sum_neighbor / w_sum;
            if (lcg_rand(rng_state) < chance) {
                reservoir.z = reservoir_n.z;
            }
            reservoir.M += reservoir_n.M;
        }
    }

    // TODO bias correction? [Ouyang 2021, algo 4]
    // Maybe not necessary since we are doing geometrical test 

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

        float3 selected_dir = normalize(reservoir.z.hit_pos - position_ws);
        float NoL = saturate(dot(gbuffer.normal, selected_dir));
        float brdf = ONE_OVER_PI;
        lighting = reservoir.z.hit_radiance * brdf * NoL * reservoir.W ;
    }

    rw_lighting_texture[dispatch_id] = lighting;
}