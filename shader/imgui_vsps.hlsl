#include "frame_bindings.hlsl"

#define IMGUI_DESCRIPTOR_SET_INDEX 0

[[vk::binding(0, IMGUI_DESCRIPTOR_SET_INDEX)]] ByteAddressBuffer vertex_buffer;
[[vk::binding(1, IMGUI_DESCRIPTOR_SET_INDEX)]] Texture2D bindless_textures[];

[[vk::push_constant]]
struct PushConstants {
    uint vertex_offset;
    uint texture_meta; // 0xFFFF0000: is font; 0x0000FFFF: texture index
    float2 texel_size;
} pc;

// 4 f32 + 1 R8G8B8A8
const static uint VERTEX_SIZE = 4 * 5;
const static uint VERTEX_COLOR_OFFSET = 4 * 4;

float4 unpack_rgba(uint enc) {
    float r = (enc & 0xFF) / 255.0;
    float g = ((enc >> 8) & 0xFF) / 255.0;
    float b = ((enc >> 16) & 0xFF) / 255.0;
    float a = ((enc >> 24) & 0xFF) / 255.0;
    return float4(r, g, b, a);
}

void vs_main(
	uint vert_id : SV_VertexID,
	out float4 hpos : SV_Position,
	out float4 color: COLOR0,
    out float2 uv: TEXCOORD0
) {
    uint vert_byte_offset = (vert_id + pc.vertex_offset) * VERTEX_SIZE;
    float4 pos_and_uv = asfloat(vertex_buffer.Load4(vert_byte_offset));
    float4 sRGB_A = unpack_rgba(vertex_buffer.Load(vert_byte_offset + VERTEX_COLOR_OFFSET));

    float2 pos = pos_and_uv.xy * pc.texel_size;
	hpos = float4(pos * 2.0 - 1.0, 1.0, 1.0);
    uv = pos_and_uv.zw;
    // TOOD convert to linear
    color = sRGB_A;
}

float4 ps_main(float4 color: COLOR0, float2 uv: TEXCOORD0) : SV_Target0 {
    uint texture_index = pc.texture_meta & 0xFFFF;
    uint is_font = pc.texture_meta >> 16;
    float4 tex_color = bindless_textures[texture_index].SampleLevel(sampler_linear_clamp, uv, 0.0);
    if (bool(is_font)) {
        // pre-multiplied result!
        float alpha = tex_color.r;
        tex_color = alpha.rrrr;
    }
    //return float4(0, 0, 0, 0);
    return float4(0, 0.5, 0.8, 1.0) * 0.5;
    //return tex_color.aaaa;
    return color * tex_color;
}
