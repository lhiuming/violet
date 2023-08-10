#pragma once

struct RestirSample {
    float3 pixel_pos; // "visible point"
    float3 pixel_normal;
    float3 hit_pos; // "sample point"
    float3 hit_normal;
    float3 hit_radiance;
    //float3 rand_seed; // Just be lousy
};

RestirSample make_restir_sample(float3 pixel_pos, float3 pixel_normal, float3 hit_pos, float3 hit_normal, float3 hit_radiance) {
    RestirSample s;
    s.pixel_pos = pixel_pos;
    s.pixel_normal = pixel_normal;
    s.hit_pos = hit_pos;
    s.hit_normal = hit_normal;
    s.hit_radiance = hit_radiance;
    return s;
}

// Mapping the ReSTIR GI paper
struct Reservoir {
    RestirSample z; // the sample
    //float w_sum; // sample's relative weight (accumulated) := Sum{ (target_pdf(z_j) / source_pdf(z_j) }; can be reconstruct by W * taget_pdf(z) * M
    uint M; // sample count
    float W; // RIS weight := rcp(target_pdf(z) * M) * Sum{ (target_pdf(z_j) / source_pdf(z_j) }
};

Reservoir init_reservoir(RestirSample z, float target_pdf, float w) {
    Reservoir r;
    r.z = z;
    //r.w_sum = w;
    r.M = 1;
    r.W = w / target_pdf;
    return r;
}