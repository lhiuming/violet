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

    /// Encode W using 24bit unsigned float point, M as 8bit unsigned integer.
    /// Should be enough for indirect diffuse.
    uint encode_32b()
    {
        uint u = min(M, 0xFF);
        u |= ((asuint(W) << 1) & 0xFFFFFF00);
        return u;
    }

    static ReservoirSimple decode_32b(uint u)
    {
        ReservoirSimple r;
        r.M = u & 0xFF;
        r.W = asfloat((u >> 1) & 0x7FFFFFE0);
        return r;
    }
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

// Encode hit_pos (XYZ in fp16) and hit_normal (octahedral in 16bit) into a uint2
struct HitPosNormal
{
    float3 pos;
    float3 normal;

    uint2 encode()
    {
        uint2 enc;
        enc.x = f32tof16(pos.x) | (f32tof16(pos.y) << 16);
        uint oct_u16 = normal_encode_oct_u16(normal);
        enc.y = f32tof16(pos.z) | (oct_u16 << 16);
        return enc;
    }

    static HitPosNormal decode(uint2 enc)
    {
        HitPosNormal ret;
        ret.pos = float3(
            f16tof32(enc.x & 0xFFFF),
            f16tof32(enc.x >> 16),
            f16tof32(enc.y & 0xFFFF));
        uint oct_u16 = enc.y >> 16;
        ret.normal = normal_decode_oct_u16(oct_u16);
        return ret;
    }
};