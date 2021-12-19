//Buffer<uint> vertex_buffer;

void vs_main(uint vert_id : SV_VertexID, out float4 hpos : SV_Position) 
{
	const float2 vbuffer[] = {
		float2(0, 1.0f),
		float2(0.5, -0.5f),
		float2(-0.5, -0.5f),
	};
	float2 v = vbuffer[vert_id % 3];
	hpos = float4(v, 0.5f, 1.0f);
}

void ps_main(out float4 output : SV_Target0)
{
	output = float4(1.0f, 1.0f, 1.0f, 1.0f);
} 