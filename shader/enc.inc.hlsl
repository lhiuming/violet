#pragma once

/*
 * Common data encoding/packing and decoding/unpacking. 
 */

// Octahedron-normal vectors //
// from: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/

float2 oct_wrap( float2 v )
{
    return ( 1.0 - abs( v.yx ) ) * select ( v.xy >= 0.0, 1.0, -1.0 );
}
 
float2 normal_encode_oct( float3 n )
{
    n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
    n.xy = select( n.z >= 0.0, n.xy, oct_wrap( n.xy ));
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}
 
float3 normal_decode_oct( float2 f )
{
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
    float t = saturate( -n.z );
    n.xy += select( n.xy >= 0.0, -t, t);
    return normalize( n );
}

uint normal_encode_oct_u32(float3 n)
{
    float2 normal_enc = normal_encode_oct(n);
    uint2 normal_unorm= uint2(normal_enc * 65535.0f);
    return normal_unorm.x | normal_unorm.y << 16;
}
 
float3 normal_decode_oct_u32(uint enc)
{
    uint2 normal_unorm = uint2(enc & 0xFFFF, (enc >> 16) & 0xFFFF);
    float2 normal_enc = normal_unorm / 65535.0f;
    return normal_decode_oct(normal_enc);
}