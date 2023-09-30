#pragma once
#include "enc.inc.hlsl"

struct GBuffer
{
    float3 color;
    float metallic;
    float perceptual_roughness;
    float3 normal;
    uint shading_path;
};


uint4 encode_gbuffer(GBuffer gbuffer) {
    uint4 enc;

    // TODO SRGB encoding?
    float3 color = gbuffer.color;
    uint3 color_8bit = uint3((color * 255.0f));
    uint metallic_8bit = uint(gbuffer.metallic * 255.0f);
    enc.r = color_8bit.x
    | color_8bit.g << 8
    | color_8bit.b << 16
    | metallic_8bit << 24;

    float2 normal_enc = normal_encode_oct(gbuffer.normal);
    uint2 normal_unorm= uint2(normal_enc * 65535.0f);
    enc.g = normal_unorm.x | normal_unorm.y << 16;

    uint roughness_unorm = uint(gbuffer.perceptual_roughness * 255.0f);
    enc.b = roughness_unorm;

    enc.a = gbuffer.shading_path;

    return enc;
}

GBuffer decode_gbuffer(uint4 enc) {
    GBuffer gbuffer;
    gbuffer.metallic = 0.0f;
    gbuffer.perceptual_roughness = 0.0f;

    uint3 color_unorm = uint3(enc.r & 0xFF, (enc.r >> 8) & 0xFF, (enc.r >> 16) & 0xFF);
    gbuffer.color = float3(color_unorm) / 255.0f;

    uint metallic_unorm = (enc.r >> 24) & 0xFF;
    gbuffer.metallic = metallic_unorm / 255.0f;

    uint2 normal_unorm = uint2(enc.g & 0xFFFF, (enc.g >> 16) & 0xFFFF);
    float2 normal_enc = normal_unorm / 65535.0f;
    gbuffer.normal = normal_decode_oct(normal_enc);

    uint roughness_unorm = enc.b & 0xFF;
    gbuffer.perceptual_roughness = roughness_unorm / 255.0f;

    gbuffer.shading_path = enc.a;

    return gbuffer;
}

// Material interpretation

bool has_no_geometry(GBuffer gbuffer) {
    return gbuffer.shading_path == 0;
}