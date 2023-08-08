#pragma once

#include "constants.hlsl"

// ---------------------
// Construct local frame
// ---------------------

// modified: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h
float4 get_rotation_to_z_from(float3 v) {
    if (v.z < -0.99999f)
        return float4(1.0f, 0.0f, 0.0f, 0.0f);
    return normalize(float4(v.y, -v.x, 0.0f, 1.0f + v.z));
}

// modified: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h
float4 invert_rotation(float4 q)
{
    return float4(-q.xyz, q.w);
}

// modified: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h
float3 rotate_point(float4 q, float3 v) {
    float3 axis = q.xyz;
	return 2.0f * dot(axis, v) * axis + (q.w * q.w - dot(axis, axis)) * v + 2.0f * q.w * cross(axis, v);
}


// -------------
// BRDF sampling
// -------------

// Taking input in range [0, 1)
// modified from: http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float3 sample_hemisphere_uniform(float2 u)
{
    float cosTheta = 1.0 - u.x;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float phi = TWO_PI * u.y;
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);

    return float3(
        cos_phi * sinTheta, 
        sin_phi * sinTheta, 
        cosTheta
    );
}

// modified: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h
float3 sample_hemisphere_cosine(float2 u, out float pdf)
{
    float sin_theta = sqrt(u.x);
    float cos_theta = sqrt(1.0f - u.x); 

    pdf = cos_theta * ONE_OVER_PI;

    float phi = TWO_PI * u.y;
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);

    return float3(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    );
}