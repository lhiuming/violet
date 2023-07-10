#pragma once

struct GBuffer
{
    float3 color;
    float metallic;
    float perceptual_roughness;
    float3 normal;
    uint shading_path;
};

//// Octahedron-normal vectors 
// from: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/

float2 OctWrap( float2 v )
{
    return ( 1.0 - abs( v.yx ) ) * ( v.xy >= 0.0 ? 1.0 : -1.0 );
}
 
float2 NormalEncode( float3 n )
{
    n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
    n.xy = n.z >= 0.0 ? n.xy : OctWrap( n.xy );
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}
 
float3 NormalDecode( float2 f )
{
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
    float t = saturate( -n.z );
    n.xy += n.xy >= 0.0 ? -t : t;
    return normalize( n );
}

// End


uint4 encode_gbuffer(GBuffer gbuffer) {
    uint4 enc;

    // TODO SRGB encoding?
    float3 color = gbuffer.color;
    uint3 color_8bit = uint3((color * 255.0f));
    uint3 metallic_8bit = uint(gbuffer.metallic * 255.0f);
    enc.r = color_8bit.x
    | color_8bit.g << 8
    | color_8bit.b << 16
    | metallic_8bit << 24;

    float2 normal_enc = NormalEncode(gbuffer.normal);
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
    gbuffer.normal = NormalDecode(normal_enc);

    uint roughness_unorm = enc.b & 0xFF;
    gbuffer.perceptual_roughness = roughness_unorm / 255.0f;

    gbuffer.shading_path = enc.a;

    return gbuffer;
}
