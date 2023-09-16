#pragma once

//
// Header for tracing geometry properties.
//

struct GeometryRayPayload {
    bool missed;
    float hit_t;
    float3 normal_ws;
    float3 normal_geo_ws;
    float3 position_ws;
    float3 base_color;
    float metallic;
    float perceptual_roughness;
    uint mesh_index;
    uint triangle_index;
};