struct Vertex
{
	float3 position;
	float padding0;
	float3 normal;
	float padding1;
};

StructuredBuffer<Vertex> vertexBuffer : register(t0);
Buffer<uint> indexBuffer : register(t1);

struct RasterData
{
	float4 pos : POSITION; 
	float4 color : COLOR;
};

static const float3 vs[3] = {
	float3(-0.5f, -0.5f, 0.5f),
	float3(0.5f, -0.5f, 0.5f),
	float3(0.0f,  0.5f, 0.5f),
};

void VSMain(uint vertexID: SV_VertexID, out RasterData o)
{
	uint Index = indexBuffer[vertexID];
	Vertex v = vertexBuffer[Index];

	const float scale = 0.02f;
	const float3 bias = float3(0, 0, 1.0f);
	o.pos = float4(v.position * scale + bias, 1.0f);
	o.pos.xyz = clamp(o.pos.xyz, float3(-0.5, -0.5, 0.0f), float3(0.5, 0.5, 1.0f));

	// debug
	{
		o.pos.xyz = vs[vertexID % 3];
	}

	//o.color = float4(v.normal, 1.0f);
	o.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
}

struct PixelOutput
{
	float4 color : COLOR0;
};

void PSMain(RasterData i, out PixelOutput o)
{
	o.color = i.color;
}
