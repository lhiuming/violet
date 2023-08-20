#include "../frame_bindings.hlsl"

#include "reservoir.hlsl"

RaytracingAccelerationStructure scene_tlas;
Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
TextureCube<float4> skycube;
Texture2D<float3> prev_color;
Texture2D<float> prev_depth;
StructuredBuffer<Reservoir> prev_reservoir_buffer;
RWStructuredBuffer<Reservoir> rw_temporal_reservoir_buffer;
RWTexture2D<float4> rw_debug_texture;

[numthreads(8, 4, 1)]
void main(uint2 dispath_id: SV_DispatchThreadID) {

}