
RWTexture2D<float4> outTexture;

[numthreads(8, 4, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{ 
	const uint BlockSize = 8;
	const uint2 BlockPos = DTid.xy / BlockSize;
	const uint BlockIndex = BlockPos.y + BlockPos.x;

	float4 color = float4(0.f, 0.f, 0.f, 1.0f);
	if ((BlockIndex) & 1)
	{
		color.b = 1.0f;
	}
	else
	{
		color.r = 0.75f;
	}
	outTexture[DTid.xy] = color;
}