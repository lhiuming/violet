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
    float perceptual_roughness;

    static GBufferNormal decode(uint enc)
    {
        GBufferNormal gbuffer;

        uint2 normal_unorm = uint2(enc & 0xFFF, (enc >> 12) & 0xFFF);
        float2 normal_enc = normal_unorm / 4095.0;
        gbuffer.normal = normal_decode_oct(normal_enc);

        uint roughness_unorm = enc >> 24;
        gbuffer.perceptual_roughness = roughness_unorm / 255.0;

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
    float3 normal;
    float perceptual_roughness;
    float fwidth_z; // for denoiser
    float fwidth_n; // for denoiser
    uint shading_path;
};

// layer 0: 24 bit color, 8 bit metallic
// layer 1: 24 bit normal, 8 bit roughness
// layer 2: 16 bit fwitdh(depth), 16 bit fwidth(normal)
uint4 encode_gbuffer(GBuffer gbuffer) {
    uint4 enc;

    // TODO maybe use hardware encode/decode (R8G8B8A8 SRGB)
    float3 color = gbuffer.color;
    uint3 color_8bit = uint3((color * 255.0f));
    uint metallic_8bit = uint(gbuffer.metallic * 255.0f);
    enc.r = color_8bit.r
    | color_8bit.g << 8
    | color_8bit.b << 16
    | metallic_8bit << 24;

    float2 normal_enc = normal_encode_oct(gbuffer.normal);
    uint2 normal_unorm= uint2(normal_enc * 4095.0);
    uint roughness_unorm = uint(gbuffer.perceptual_roughness * 255.0f);
    enc.g = normal_unorm.x | normal_unorm.y << 12 | roughness_unorm << 24;

    uint2 fwidth_zn_f16 = f32tof16(float2(gbuffer.fwidth_z, gbuffer.fwidth_n));
    enc.b = fwidth_zn_f16.x | fwidth_zn_f16.y << 16;

    enc.a = gbuffer.shading_path;

    return enc;
}

GBuffer decode_gbuffer(uint4 enc) {
    GBuffer gbuffer;

    uint3 color_unorm = uint3(enc.r & 0xFF, (enc.r >> 8) & 0xFF, (enc.r >> 16) & 0xFF);
    gbuffer.color = float3(color_unorm) / 255.0f;

    uint metallic_unorm = (enc.r >> 24) & 0xFF;
    gbuffer.metallic = metallic_unorm / 255.0f;

    uint2 normal_unorm = uint2(enc.g & 0xFFF, (enc.g >> 12) & 0xFFF);
    float2 normal_enc = normal_unorm / 4095.0;
    gbuffer.normal = normal_decode_oct(normal_enc);

    uint roughness_unorm = enc.g >> 24;
    gbuffer.perceptual_roughness = roughness_unorm / 255.0f;

    uint2 fwidth_zn_f16 = uint2(enc.b & 0xFFFF, enc.b >> 16);
    float2 fwidth_zn = f16tof32(fwidth_zn_f16);
    gbuffer.fwidth_z = fwidth_zn.x;
    gbuffer.fwidth_n = fwidth_zn.y;

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