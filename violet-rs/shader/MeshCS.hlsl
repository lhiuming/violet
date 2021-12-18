RWTexture2D<float4> target : t0;

[numthreads(8, 4, 1)]
void main(uint2 dtid : SV_DispatchThreadID) {
	uint2 target_size; 
	target.GetDimensions(target_size.x, target_size.y);

	uint2 offset = min(dtid, (target_size - 1 - dtid));
	if (dot(offset, offset) < (128 * 128))
	{
		target[dtid] = float4(0.87, 0.13, 0.0, 1.0);
	}
}