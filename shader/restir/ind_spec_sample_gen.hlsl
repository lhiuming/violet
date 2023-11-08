#include "../brdf.hlsl"
#include "../enc.inc.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"
#include "../util.hlsl"

#include "raytrace.inc.hlsl"
#include "reservoir.hlsl"

GBUFFER_TEXTURE_TYPE gbuffer_color;
Texture2D<float> gbuffer_depth;

// New sample propertie textures
//RWTexture2D<float3> rw_origin_pos_texture;
RWTexture2D<float4> rw_hit_pos_texture;
RWTexture2D<uint> rw_hit_normal_texture;
RWTexture2D<float3> rw_hit_radiance_texture;

RWTexture2D<float4> rw_debug_texture;

[[vk::push_constant]]
struct PushConstants
{
    uint frame_index;
    uint has_prev_frame;
} pc;

[shader("raygeneration")]
void main()
{
    uint2 dispatch_id = DispatchRaysIndex().xy;
    float depth = gbuffer_depth[dispatch_id.xy];

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    // early out if no geometry
    if (has_no_geometry_via_depth(depth))
    {
        // For Debug
        rw_hit_radiance_texture[dispatch_id.xy] = 0.0;
        return;
    }

    GBuffer gbuffer = load_gbuffer(gbuffer_color, dispatch_id.xy);

    // world position reconstruction from depth buffer
    float depth_error = 1.f / 16777216.f; // 24 bit unorm depth
    float2 screen_pos = (dispatch_id.xy + 0.5f) / float2(buffer_size);
    float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_pos * 2.0f - 1.0f, depth + depth_error, 1.0f));
    float3 position_ws = position_ws_h.xyz / position_ws_h.w;

    // Specualr may need some better noise
    uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

    // Generate sample direction
    // TODO blue noise
    float3 sample_dir;
    float3 brdf_NoL_over_pdf;
    #if 0
    // uniform hemisphere sampling
    {
        float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
        sample_dir = sample_hemisphere_uniform_with_normal(u, gbuffer.normal);
    }
    #else
    // VNDF sampling
    {
        float3 view_dir = normalize(view_params().view_pos - position_ws);
        float roughness = gbuffer.perceptual_roughness * gbuffer.perceptual_roughness;
        if (IND_SPEC_R_MIN > 0)
        {
            roughness = max(roughness, IND_SPEC_R_MIN);
        }

        float4 rot_to_local = get_rotation_to_z_from(gbuffer.normal);
        float3 V_local = rotate_point(rot_to_local, view_dir);

        // Sample a half vector (microfacet normal) from the GGX distribution of visible normals (VNDF).
        // pdf_H = D_visible(H) = ( G1(V) * VoH * D(H) / NoV ) 
        // TODO: this method allow V_local.z < 0 (viewing below the surface)
        // ref: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h#L727
        // [Heitz 2014 "Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals", section 2]
        // TODO compare methods [Heitz 2018 "Sampling the GGX Distribution of Visible Normals"]
        float3 H_local;
        if (roughness != 0.0f)
        {
            float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));

            // We are not using anisotropic GGX
            float2 alpha2d = roughness.xx;

            float3 v_h = normalize(float3(alpha2d.x * V_local.x, alpha2d.y * V_local.y, V_local.z));

            float lensq = dot(v_h.xy, v_h.xy);
            float3 T1 = lensq > 0.0f ? float3(-v_h.y, v_h.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
            float3 T2 = cross(v_h, T1);

            float r = sqrt(u.x);
            float phi = TWO_PI * u.y;
            float t1 = r * cos(phi);
            float t2 = r * sin(phi);
            float s = 0.5f * (1.0f + v_h.z);
            t2 = lerp(sqrt(1.0f - t1 * t1), t2, s);

            float3 n_h = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * v_h;

            H_local = normalize(float3(alpha2d.x * n_h.x, alpha2d.y * n_h.y, max(0.0f, n_h.z)));
        }
        else 
        {
            // perfect mirror (roughness == 0.0f) has singularity when transform the view direction to hemisphere configturation 
            H_local = float3(0.0f, 0.0f, 1.0f);
        }

        // Apply relfect operator
        float3 L_local = reflect(-V_local, H_local);

        sample_dir = rotate_point(invert_rotation(rot_to_local), L_local);
    }
    #endif

    // Raytrace
    RadianceTraceResult trace_result = trace_radiance(position_ws, sample_dir, pc.has_prev_frame);

    //rw_origin_pos_texture[dispatch_id.xy] = position_ws;
    rw_hit_pos_texture[dispatch_id.xy] = float4(trace_result.position_ws, 1.0f);
    rw_hit_normal_texture[dispatch_id.xy] = normal_encode_oct_u32(trace_result.normal_ws);
    rw_hit_radiance_texture[dispatch_id.xy] = trace_result.radiance;
}
