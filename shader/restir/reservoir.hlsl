#pragma once
#include "../enc.inc.hlsl"

struct RestirSample
{
    float3 pixel_pos; // "visible point", used in sample validation pass
    float3 hit_pos; // "sample point"
    float3 hit_normal;
    float3 hit_radiance;
    // float3 rand_seed; // Just be lousy
};

RestirSample make_restir_sample(float3 pixel_pos, float3 hit_pos, float3 hit_normal, float3 hit_radiance)
{
    RestirSample s;
    s.pixel_pos = pixel_pos;
    s.hit_pos = hit_pos;
    s.hit_normal = hit_normal;
    s.hit_radiance = hit_radiance;
    return s;
}

// Mapping the ReSTIR GI paper
// w_sum, a.k.a sample's accumulated relative weight (:= Sum{ (target_pdf(z_j) / source_pdf(z_j) }), should be reconstructed by:
//   w_sum := W * taget_pdf(z) * M
struct Reservoir
{
    RestirSample z; // the sample
    uint M;         // sample count
    float W;        // RIS weight := rcp(target_pdf(z) * M) * Sum{ (target_pdf(z_j) / source_pdf(z_j) }
};

Reservoir null_reservoir()
{
    Reservoir r;
    r.z = (RestirSample)0;
    r.M = 0;
    r.W = 0.0f;
    return r;
}

Reservoir init_reservoir(RestirSample z, uint M, float W)
{
    Reservoir r;
    r.z = z;
    r.M = M;
    r.W = W;
    return r;
}

struct ReservoirSimple
{
    uint M;
    float W;
};

uint2 reservoir_encode_u32(ReservoirSimple r)
{
    uint2 u;
    u.x = r.M;
    u.y = asuint(r.W);
    return u;
}

ReservoirSimple reservoir_decode_u32(uint2 u)
{
    ReservoirSimple r;
    r.M = u.x;
    r.W = asfloat(u.y);
    return r;
}

#define IND_SPEC_R_MIN 0.002
