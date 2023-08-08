#pragma once

// ---
// LCG
//----

// from: Ray Tracing Gems II, Chapter 14

uint jenkins_hash(uint x) {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    return x;
}

// original: init_rng
uint lcg_init(uint2 pixel_coords, uint2 resolution, uint frame) {
    uint rng_state = dot(pixel_coords, uint2(1, resolution.x)) ^ jenkins_hash(frame);
    return jenkins_hash(rng_state);
}

// original:uint_to_float 
float lcg_uint_to_float(uint x) {
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

// original: xorshift
uint lcg_xorshift(inout uint x /* rng state */) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// original: rand
float lcg_rand(inout uint rng_state) {
    return lcg_uint_to_float(lcg_xorshift(rng_state));
}