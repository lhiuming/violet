#include "gbuffer.hlsl"
#include "scene_bindings.hlsl"
#include "frame_bindings.hlsl"
#include "brdf.hlsl"


Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;

TextureCube<float4> skycube;
Texture2D<float> shadow_mask;

RWTexture2D<float4> out_lighting;

float3 cal_lighting(float3 v /*view*/, float3 l /*light*/, float3 n /*normal*/, float perceptualRoughness, float3 diffuseColor, float3 specularColor)
{
	float3 h = normalize(v + l);

    float NoV = abs(dot(n, v)) + 1e-5;
    float NoL = clamp(dot(n, l), 0.0, 1.0);
    float NoH = clamp(dot(n, h), 0.0, 1.0);
    float LoH = clamp(dot(l, h), 0.0, 1.0);

    // perceptually linear roughness to roughness (see parameterization)
	// Clamp for cheap specular aliasing under punctual light
	perceptualRoughness = max(perceptualRoughness, 0.045f);
    float roughness = perceptualRoughness * perceptualRoughness;

    float D = D_GGX(NoH, roughness);
    float3 F = F_Schlick(LoH, specularColor);
    float V = vis_smith_G2_height_correlated_GGX(NoV, NoL, roughness);

    // specular BRDF
    float3 Fr = (D * V) * F;

    // diffuse BRDF
    float3 Fd = diffuseColor * Fd_Lambert();

	return (Fd + Fr) * NoL;
}


[numthreads(8, 8, 1)]
void main(uint2 dispatch_thread_id: SV_DISPATCHTHREADID) {
    uint4 gbuffer_enc = gbuffer_color[dispatch_thread_id];
    GBuffer gbuffer = decode_gbuffer(gbuffer_enc);

    // early out if no geometry
    if (gbuffer.shading_path == 0)
        return;

    float depth = gbuffer_depth[dispatch_thread_id];
    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    float shadow_atten = shadow_mask[dispatch_thread_id.xy];

	// world position reconstruction
    float2 screen_pos = (dispatch_thread_id + 0.5f) / float2(buffer_size);
	float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_pos * 2.0f - 1.0f, depth, 1.0f));
	float3 position_ws = position_ws_h.xyz / position_ws_h.w;

    // Direct lighting
	float3 view = normalize(view_params().view_pos - position_ws);
	float3 diffuseColor = gbuffer.color.rgb * (1.0f - gbuffer.metallic);
	float3 specularColor = lerp(float3(0.04f, 0.04f, 0.04f), gbuffer.color, gbuffer.metallic);
	float3 light_inten = float3(1.2f, 1.1f, 1.0f) * PI;
	float3 lighting = cal_lighting(view, frame_params.sun_dir.xyz, gbuffer.normal, gbuffer.perceptual_roughness, diffuseColor, specularColor) * light_inten * shadow_atten;

#if 0
	// IBL fake
	//if (roughness < 0.01f)
	{
        float glossness = 1.0f - roughness;
		float3 refl_dir = reflect(-view, gbuffer.normal);
		lighting += specularColor * skycube.SampleLevel(sampler_linear_clamp, refl_dir, 0).rgb * glossness;

	    float3 ambient = 0.5f * roughness;
	    lighting += diffuseColor * ambient;
	}
#endif

    // Output
    out_lighting[dispatch_thread_id] = float4(lighting, 1.0f);

    // Debug
#if 0
    float3 debug_color = float3(0, 0, 0);
    //debug_color = gbuffer.normal * 0.5f + 0.5f;
    //debug_color = position_ws * 0.5f + 0.5f;
    //debug_color = reflect(-view, gbuffer.normal) * 0.5f + 0.5f;
    debug_color = view * 0.5f + 0.5f;
    //debug_color.x = shadow_atten;
    //debug_color = float3(gbuffer.metallic, gbuffer.perceptual_roughness, 0.0f);
    out_lighting[dispatch_thread_id] = float4(debug_color, 1.0f);
#endif
}