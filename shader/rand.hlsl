#pragma once

//----
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
// TODO jenkins_hash(frame) can be done in CPU
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

//----
// PCG
//----

// [Jarzynski 2020 "Hash Functions for GPU Rendering"]
// https://www.shadertoy.com/view/XlGcRh
uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// [Jarzynski 2020 "Hash Functions for GPU Rendering"]
uint3 pcg_3d(uint3 v)
{
	v = v * 1664525u + 1013904223u;  
	v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;  
	v ^= v >> 16u;  
	v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;  
	return v;
}

//-------
// xxhash
//-------

// xxhash32 with un-limited input.
// modified from:
// https://www.shadertoy.com/view/XlGcRh
struct XXHash32 {
    uint state;

    static XXHash32 init(uint p) {
        const uint PRIME32_5 = 374761393U;
	    uint h32 = p + PRIME32_5;
        XXHash32 ret = { h32 };
        return ret;
    }

    XXHash32 add(uint p) {
        const uint PRIME32_3 = 3266489917U;
	    const uint PRIME32_4 = 668265263U;
        uint h32 = state;
	    h32 += p * PRIME32_3;
	    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
        XXHash32 ret = { h32 };
        return ret;
    }

    uint eval() {
        const uint PRIME32_2 = 2246822519U;
        const uint PRIME32_3 = 3266489917U;
	    const uint PRIME32_4 = 668265263U;
        uint h32 = state;
	    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
        h32 = PRIME32_2 * (h32^(h32 >> 15));
        h32 = PRIME32_3 * (h32^(h32 >> 13));
        return h32 ^ (h32 >> 16);
    }
};
