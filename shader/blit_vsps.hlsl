#include "frame_bindings.hlsl"

Texture2D<float4> source_texture;

void vs_main(
	uint vert_id : SV_VertexID,
	out float4 hpos : SV_Position,
    out float2 uv: TEXCOORD0
) {
	float2 pos = float2(vert_id & 1, vert_id >> 1);
	hpos = float4(pos * 4.0 - 1.0, 1.0, 1.0);
    uv = hpos.xy * 0.5 + 0.5;
}

float4 ps_main(float2 uv: TEXCOORD0) : SV_Target0 {
	return source_texture.SampleLevel(sampler_nearest_clamp, uv, 0);
}
