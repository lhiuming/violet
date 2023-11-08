#pragma once
#include "enc.inc.hlsl"

#define GBUFFER_TEXTURE_TYPE Texture2DArray<uint>

uint2 get_gbuffer_dimension_2d(GBUFFER_TEXTURE_TYPE gbuffers)
{
    uint2 size;
    uint depth;
    uint levels;
    gbuffers.GetDimensions(0, size.x, size.y, depth, levels);
    return size;
}

struct GBufferNormal
{
    float3 normal;

    uint encode() {
        float2 normal_enc = normal_encode_oct(normal);
        uint2 normal_unorm= uint2(normal_enc * 65535.0f);
        uint enc = normal_unorm.x | normal_unorm.y << 16;
        return enc;
    }

    static GBufferNormal decode(uint enc)
    {
        uint2 normal_unorm = uint2(enc & 0xFFFF, (enc >> 16) & 0xFFFF);
        float2 normal_enc = normal_unorm / 65535.0f;
        GBufferNormal gbuffer;
        gbuffer.normal = normal_decode_oct(normal_enc);
        return gbuffer;
    }

    static GBufferNormal load(GBUFFER_TEXTURE_TYPE gbuffers, uint2 coord)
    {
        return decode(gbuffers[uint3(coord, 1)]);
    }
};

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

GBuffer load_gbuffer(GBUFFER_TEXTURE_TYPE gbuffers, uint2 coord)
{
    uint4 enc = uint4(
        gbuffers[uint3(coord, 0)],
        gbuffers[uint3(coord, 1)],
        gbuffers[uint3(coord, 2)],
        gbuffers[uint3(coord, 3)]
    );
    return decode_gbuffer(enc);
}


// Material interpretation

bool has_no_geometry(GBuffer gbuffer) {
    return gbuffer.shading_path == 0;
}