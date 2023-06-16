// Scene/Persistent Bindings (Set #1)
[[vk::binding(0, 1)]] Buffer<uint> vertex_buffer;
[[vk::binding(1, 1)]] Texture2D bindless_textures[];
struct ViewParams 
{
	float4x4 view_proj;
	float4x4 inv_view_proj;
	float3 view_pos;
	float padding0;
};
[[vk::binding(2, 1)]] ConstantBuffer<ViewParams> view_params;
[[vk::binding(3, 1)]] SamplerState sampler_linear_clamp;

// Free bindings (Set #0)
// ...