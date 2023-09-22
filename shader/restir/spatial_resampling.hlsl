#include "../constants.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../util.hlsl"
#include "reservoir.hlsl"

#define MAX_ITERATION 9
#define INIT_RADIUS_FACTOR 0.1
#define MIN_RADIUS 1.5
#define USE_AO 1

// Using cosine-weighted PDF (instead of uniformly distrubuted) PDF reduce noise alot
#define UNIFORM_PDF 0

Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
Texture2D<float> ao_texture;
StructuredBuffer<Reservoir> temporal_reservoir_buffer;
RWStructuredBuffer<Reservoir> rw_spatial_reservoir_buffer;
RWTexture2D<float3> rw_lighting_texture;
RWTexture2D<float4> rw_debug_texture;

[[vk::push_constant]]
struct PushConstants
{
    uint frame_index;
} pc;

// TODO: using luminance() might cause bias toward green samples ...
float radiance_reduce(float3 radiance)
{
    //return luminance(radiance);
    //return dot(radiance, 1.0.rrr);
    return max3(radiance);
}

#if UNIFORM_PDF
#define TARGET_PDF_FOR_SAMPLE(z) radiance_reduce(z.hit_radiance)
#else
#define TARGET_PDF_FOR_SAMPLE(z) (saturate(dot(gbuffer.normal, normalize(z.hit_pos - position_ws))) * radiance_reduce(z.hit_radiance))
#endif

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    float depth = gbuffer_depth[dispatch_id.xy];

    // early out if not geometry
    if (has_no_geometry_via_depth(depth))
    {
        return;
    }

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    const float3 position_ws =
        cs_depth_to_position(dispatch_id, buffer_size, depth);

    // Normal (for similarity test and consine pdf)
    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);
    float ao = ao_texture[dispatch_id.xy];

    Reservoir reservoir;
    float w_sum;
    {
        uint index = dispatch_id.y * buffer_size.x + dispatch_id.x;
        reservoir = temporal_reservoir_buffer[index];
        float target_pdf = TARGET_PDF_FOR_SAMPLE(reservoir.z);
        w_sum = reservoir.W * target_pdf * float(reservoir.M);
    }

    //float radius = MAX_RADIUS; // px
    float radius = min(buffer_size.x, buffer_size.y) * INIT_RADIUS_FACTOR;
    #if USE_AO
    radius = max(radius * ao * ao, MIN_RADIUS);
    #endif
    uint rng_state = lcg_init(dispatch_id, buffer_size, pc.frame_index);
    for (uint i = 0; i < MAX_ITERATION; i++)
    {
#if 1
        // Uniform sampling in a disk
        float x, y;
        float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
        {
            float r = sqrt(u.x);
            float theta = TWO_PI * u.y;
            x = r * cos(theta);
            y = r * sin(theta);
        }
#else
        // Sampling in a golden spiral with max radius 1
        // TODO per frame jitter + per pixel rotation
        const float b = log(1.61803398) / (0.5 * PI);
        const float a = 1.0f / exp(b * TWO_PI);
        float theta = TWO_PI * ((i + 1) / float(MAX_ITERATION));
        float r = a * exp(b * theta);
        float x = r * cos(theta);
        float y = r * sin(theta);
#endif

        int2 offset = int2(float2(x, y) * radius);
        uint2 pixcood_n =
            clamp(int2(dispatch_id) + offset, int2(0, 0), int2(buffer_size) - 1);

        GBuffer gbuffer_n = decode_gbuffer(gbuffer_color[pixcood_n]);
        float depth_n = gbuffer_depth[pixcood_n];

#if 0
        // Sampled invalid reservoir
        // Only usefull if we do not clear the reservoir in sky pixel
        if (has_no_geometry_via_depth(depth_n))
        {
            continue;
        }
#endif

        // Geometry similarity test
        bool geometrical_diff = false;
        // normal test (within 25 degree) [Ouyang 2021]
        const float NORMAL_ANGLE_TELERANCE = PI * (25.0 / 180.0);
        geometrical_diff |=
            dot(gbuffer.normal, gbuffer_n.normal) < cos(NORMAL_ANGLE_TELERANCE);
        // depth test (within 0.05 of depth range) [Ouyang 2021]
        // TODO should be normalized depth
        // TODO wall viewed from grazing-angle will become noisy because depth
        // gradient is large in this case. So a better test should check if two
        // visble sample are on the same surface (plane)
        const float DEPTH_TOLERANCE = 0.0005f;
        geometrical_diff |= abs(depth - depth_n) > DEPTH_TOLERANCE;
        if (geometrical_diff)
        {
            // [Ouyang 2021]
            radius = max(radius * 0.5, MIN_RADIUS);
            continue;
        }

        uint index_n = pixcood_n.x + pixcood_n.y * buffer_size.x;
        Reservoir reservoir_n = temporal_reservoir_buffer[index_n];

        // Jacobian determinant for spatial reuse
        float reuse_jacobian = 1.0f;
        if (1)
        {
            // Read the reused path:
            //   first vertex: neighbor position
            //   second vertex: reused sample hit position
            // NOTE: the first vertex must be the neighbor position, not the original
            // sample ray origin, otherwise a same (bright) path can get reused
            // multiple times with the same (possibly high) jacobian in the iteration
            // and the pixel blows up.
            float3 position_ws_n =
                cs_depth_to_position(pixcood_n, buffer_size, depth_n);
            float3 sample_hit_pos_ws = reservoir_n.z.hit_pos;
            float3 sample_hit_normal_ws = reservoir_n.z.hit_normal;

            // [Ouyang 2021, eq 11]
            // x_1^q-x_2^q
            float3 offset_qq = position_ws_n - sample_hit_pos_ws;
            // x_1^r-x_2^q
            float3 offset_rq = position_ws - sample_hit_pos_ws;
            float L2_qq = dot(offset_qq, offset_qq);
            float L2_rq = dot(offset_rq, offset_rq);
            // cos(phi_2^r)
            float cos_rq = dot(sample_hit_normal_ws, offset_rq * rsqrt(L2_rq));
            // cos(phi_2^q)
            float cos_qq = dot(sample_hit_normal_ws, offset_qq * rsqrt(L2_qq));
            reuse_jacobian = (cos_rq / cos_qq) * (L2_qq / L2_rq);

            // Clamp to avoid fireflies.
            // Jacobian of large value can happen when the reused hit is too close to
            // current surface. Basically we apply the jocobian for the purpose of
            // darkening the sample to avoid over-brightness artifacts at corners, so
            // clampping high value should be fine.
            reuse_jacobian = clamp(reuse_jacobian, 0, 10);

            // HACK clamp to 1.0 to avoid bright spot
            // TODO still dont know what produces the bright propagating spot in the
            // scene sometime (comment out bellow clamp and you will see it under a
            // fix cam pos in the scene). May by something to do the the random/LCG
            // sampling in the neighboor hood?
            reuse_jacobian = clamp(reuse_jacobian, 0, 1);
        }

        // NOTE: should be a typo in the ReSTIR GI paper [Ouyang 2021, algo 4]: It
        // should be `target_pdf * jacobian` instead of `target_pdf / jacobian`
        float target_pdf_neighbor = TARGET_PDF_FOR_SAMPLE(reservoir_n.z) * reuse_jacobian;

        // TODO visibility test, and continue
        // At least NoL test is cheap enough?
        //if (0)
        if (target_pdf_neighbor <= 0.0)
        {
            // [Ouyang 2021]
            //radius = max(radius * 0.5, 3.0);
            continue;
        }

        // Merge reservoir (sample reuse)
        {
            if (1)
            {
                // How can M ever exceed 500? (M is clamped to 30+1 in temporal, and up
                // to 10 merged reservoir after spatial)
                uint M_MAX = 500; // [Ouyang 2021]
                reservoir_n.M = min(reservoir_n.M, M_MAX);
            }

            float w_sum_neighbor =
                reservoir_n.W * target_pdf_neighbor * float(reservoir_n.M);
            w_sum += w_sum_neighbor;
            float chance = w_sum_neighbor / w_sum;
            if (lcg_rand(rng_state) < chance)
            {
                reservoir.z = reservoir_n.z;
            }
            reservoir.M += reservoir_n.M;
        }
    }

    // TODO bias correction? [Ouyang 2021, algo 4]
    // Maybe not necessary since we are doing geometrical test

    // update the W
    {
        float target_pdf = TARGET_PDF_FOR_SAMPLE(reservoir.z);
        if (target_pdf > 0)
        {
            reservoir.W = w_sum / (target_pdf * float(reservoir.M));
        }
        else
        {
            // just keep the original W
        }
    }

    // Store for next frame temporal reuse
    {
        uint index = dispatch_id.y * buffer_size.x + dispatch_id.x;
        rw_spatial_reservoir_buffer[index] = reservoir;
    }

    // Evaluate the RIS esimator
    float3 lighting;
    {
        float3 selected_dir = normalize(reservoir.z.hit_pos - position_ws);
        float NoL = saturate(dot(gbuffer.normal, selected_dir));
        float brdf = ONE_OVER_PI;
        lighting = reservoir.z.hit_radiance * brdf * NoL * reservoir.W;
    }

    rw_lighting_texture[dispatch_id] = lighting;

#if 0
    // Debug view
    float M_intent = pow(reservoir.M / 300.0, 0.5);
    rw_debug_texture[dispatch_id.xy] = float4(M_intent, 0, 0, 0);
#endif
}
