#pragma once

//
// Header for tracing geometry properties.
//

#include "../enc.inc.hlsl"

#define SHRINK_PAYLOAD 0

#if SHRINK_PAYLOAD

struct GeometryRayPayload {
    float hit_t;
    uint base_color_enc;
    uint normal_ws_enc;
    uint normal_geo_ws_enc;

    static uint enc_color_rough_metal(float3 col, float perceptual_roughness, float metallic) {
        uint ret = 0;
        uint3 col_u = uint3(col * float((1 << 8) - 1));
        ret |= col_u.r << 0;
        ret |= col_u.g << 8;
        ret |= col_u.b << 16;
        uint r_u = uint(perceptual_roughness * float((1 << 7) - 1));
        ret |= r_u << 24;
        ret |= asuint(select(metallic > 0.5f, 1 << 31, 0));
        return ret;
    }

    bool get_missed() {
        return hit_t < 0.0f;
    }

    float3 get_base_color() {
        uint3 col_u = uint3(
            base_color_enc & 0xFF,
            (base_color_enc >> 8) & 0xFF,
            (base_color_enc >> 16) & 0xFF
        );
        return float3(col_u) / float((1 << 8) - 1);
    }

    float get_perceptual_roughness() {
        return float((base_color_enc >> 24) & 0x7F) / float((1 << 7) - 1);
    }

    float get_metallic() {
        return select(bool(base_color_enc >> 31), 1.0f, 0.0f);
    }

    float3 get_normal_ws() {
        return normal_decode_oct_u32(normal_ws_enc);
    }

    float3 get_normal_geo_ws() {
        return normal_decode_oct_u32(normal_geo_ws_enc);
    }
};

#else

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

    bool get_missed() {
        return missed;
    }

    float3 get_base_color() {
        return base_color;
    }

    float get_perceptual_roughness() {
        return perceptual_roughness;
    }

    float get_metallic() {
        return metallic;
    }

    float3 get_normal_ws() {
        return normal_ws;
    }

    float3 get_normal_geo_ws() {
        return normal_geo_ws;
    }
};

#endif