#include "atmosphere_felix_westin.hlsl"
#include "scene_bindings.hlsl"

void vs_main(
	uint vert_id : SV_VertexID,
	out float4 hpos : SV_Position,
	out float3 position_ws : TEXCOORD0 
) {
	float2 pos = float2(vert_id & 1, vert_id >> 1);
	hpos = float4(pos * 4.0 - 1.0, 1.0, 1.0);
	float4 position_ws_h = mul(view_params.inv_view_proj, hpos);
	position_ws = position_ws_h.xyz / position_ws_h.w;
}

float4 ps_main(float3 position_ws : TEXCOORD0) : SV_Target0 {
	float3 ray_start = view_params.view_pos;
	float3 ray_dir = normalize(position_ws - ray_start);
	float ray_len = 100000.0f;

	float3 light_dir = float3(0.0f, 0.0f, 1.0f);
	float3 light_color = float3(0.7f, 0.7f, 0.6f) * 3.14f;

	float3 _transmittance;
	float3 color = IntegrateScattering(ray_start, ray_dir, ray_len, light_dir, light_color, _transmittance);
	return float4(color, 1.0f);
}
