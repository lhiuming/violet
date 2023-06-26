#define SCENE_DESCRIPTOR_SET_INDEX 1

// Scene/Persistent Bindings
[[vk::binding(0, SCENE_DESCRIPTOR_SET_INDEX)]] Buffer<uint> vertex_buffer;
[[vk::binding(1, SCENE_DESCRIPTOR_SET_INDEX)]] Texture2D bindless_textures[];
struct ViewParams 
{
	float4x4 view_proj;
	float4x4 inv_view_proj;
	float3 view_pos;
	float padding0;
	float3 sun_dir;
};
[[vk::binding(2, SCENE_DESCRIPTOR_SET_INDEX)]] ConstantBuffer<ViewParams> view_params;
[[vk::binding(3, SCENE_DESCRIPTOR_SET_INDEX)]] SamplerState sampler_linear_clamp;
