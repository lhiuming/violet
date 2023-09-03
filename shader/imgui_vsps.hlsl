ByteAddressBuffer vertex_buffer;

[[vk::push_constant]]
struct PushConstants {
    uint vertex_offset;
    float2 texel_size;
} pc;

const static uint VERTEX_SIZE = 4 * 5; // 5 uint32
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
    vert_id = vert_id + pc.vertex_offset;
    uint vert_byte_offset = (vert_id + pc.vertex_offset) * VERTEX_SIZE;
    float4 pos_and_uv = asfloat(vertex_buffer.Load4(vert_byte_offset));
    float4 sRGB_A = unpack_rgba(vertex_buffer.Load(vert_byte_offset + VERTEX_COLOR_OFFSET));

    float2 pos = pos_and_uv.xy * pc.texel_size;
	hpos = float4(pos * 2.0 - 1.0, 1.0, 1.0);
    uv = pos_and_uv.zw;
    color = float4(1.0, 1.0, 1.0, sRGB_A.a);
}

float4 ps_main(float4 color: COLOR0, float2 uv: TEXCOORD0) : SV_Target0 {
    return color;
}
