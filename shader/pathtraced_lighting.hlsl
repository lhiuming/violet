#include "gbuffer.hlsl"
#include "scene_bindings.hlsl"
#include "frame_bindings.hlsl"
#include "brdf.hlsl"
#include "rand.hlsl"
#include "sampling.hlsl"
#include "util.hlsl"

#include "raytrace/geometry_ray.inc.hlsl"
#include "raytrace/shadow_ray.inc.hlsl"

/* 
This is a reference path tracer, not optimized for performance.
Reference: 
    - Ray Tracing Gems II, Chapter 14, Reference Path Tracer.
*/

#define MAX_BOUNCE 8
#define MAX_BOUNCE_WITH_LIGHTING MAX_BOUNCE

#define DIRECTI_LIGHTING 1
#define PRIMARY_RAY_JITTER 1
#define SPECULAR_SUPRESSION 1

#define WHITE_FURNACE 0
#define SPECULAR_ONLY 0
#define DIFFUSE_ONLY 0

RaytracingAccelerationStructure scene_tlas;
TextureCube<float4> skycube;
RWTexture2D<float4> rw_accumulated;
RWTexture2D<float3> rw_lighting;

struct PushConstants
{
    uint frame_index;
    uint accumulated_count;
    uint stop_accumuate;
};
[[vk::push_constant]]
PushConstants pc;

[shader("raygeneration")]
void main() {
    uint2 dispatch_id = DispatchRaysIndex().xy;

    if (bool(pc.stop_accumuate)) {
        rw_lighting[dispatch_id] = rw_accumulated[dispatch_id].xyz / pc.accumulated_count;
        return;
    }

    uint2 buffer_size;
    rw_accumulated.GetDimensions(buffer_size.x, buffer_size.y);

    uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

    // Primary ray direction
    float2 pix_coord = float2(dispatch_id) + 0.5f;
    #if PRIMARY_RAY_JITTER
    pix_coord += lerp(-0.5f.xx, 0.5f.xx, float2(lcg_rand(rng_state), lcg_rand(rng_state)));
    #endif
    float2 ndc = pix_coord / float2(buffer_size) * 2.0f - 1.0f;
    float4 view_dir_end_h = mul(view_params().inv_view_proj, float4(ndc, 1.0f, 1.0f));
    float3 view_dir_end = view_dir_end_h.xyz / view_dir_end_h.w;
    float3 view_dir = normalize(view_dir_end - view_params().view_pos);

    // Setup primary Ray
    RayDesc ray;
    ray.Origin = view_params().view_pos;
    ray.Direction = view_dir;
    ray.TMin = 0.0005f; // 0.5mm
    ray.TMax = 1000.0f;

    float3 throughput = 1.0f;
    float3 radiance = 0.0f;

    // Ray tracing loop
    float4 debug_color = 0.0f;
    for (int bounce = 0; bounce < MAX_BOUNCE; ++bounce) 
    {
        // Trace
        GeometryRayPayload hit = (GeometryRayPayload)0;
        hit.missed = false;
        TraceRay(scene_tlas,
                RAY_FLAG_FORCE_OPAQUE, // skip anyhit
                0xff, // uint InstanceInclusionMask,
                0, // uint RayContributionToHitGroupIndex,
                0, // uint MultiplierForGeometryContributionToHitGroupIndex,
                0, // uint MissShaderIndex,
                ray,
                hit
            );

        // Sky light
        if (hit.missed)
        {
            float3 skycolor = skycube.SampleLevel(sampler_linear_clamp, ray.Direction, 0).rgb;
            #if WHITE_FURNACE
            skycolor = 1.0f;
            #endif
            radiance += throughput * skycolor;
            break;
        }

        if (bounce == (MAX_BOUNCE - 1)) 
        {
            break;
        }

        #if WHITE_FURNACE
        hit.base_color = 1.0f;
        #endif

        // Material attribute decode
	    const float3 specular_color = get_specular_f0(hit.base_color, hit.metallic);
	    const float3 diffuse_color = get_diffuse_rho(hit.base_color, hit.metallic);
        #if SPECULAR_SUPRESSION
        float perceptual_roughness = max(hit.perceptual_roughness, 0.045f);
        const float roughness = perceptual_roughness * perceptual_roughness;
        #else
        const float roughness = hit.perceptual_roughness * hit.perceptual_roughness;
        #endif

        // Add Direct Lighting
        #if !WHITE_FURNACE && DIRECTI_LIGHTING
        float nol_sun = dot(hit.normal_ws, frame_params.sun_dir.xyz);
        if ((nol_sun > 0.0f) && (bounce < MAX_BOUNCE_WITH_LIGHTING))
        {
            //float leviation = 1.f/32.f;
            float leviation = 0.0f;

            RayDesc shadow_ray;
            shadow_ray.Origin = hit.position_ws + hit.normal_geo_ws * leviation;
            shadow_ray.Direction = frame_params.sun_dir.xyz;
            shadow_ray.TMin = 0.0005f; // 0.5mm
            shadow_ray.TMax = 1000.0f;

            ShadowRayPayload shadow = (ShadowRayPayload)0;
            shadow.missed = false;
            TraceRay(scene_tlas,
                RAY_FLAG_FORCE_OPAQUE // skip anyhit
                | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER // skip closesthit
                | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH // shadow
                , 
                0xff, // uint InstanceInclusionMask,
                0, // uint RayContributionToHitGroupIndex,
                0, // uint MultiplierForGeometryContributionToHitGroupIndex,
                1, // uint MissShaderIndex,
                shadow_ray,
                shadow
            );

            if (shadow.missed)
            {
                // Match with what eval_GGX_Lambertian is doing

                float3 v = -ray.Direction;
                float3 l = frame_params.sun_dir.xyz;
                float3 n = hit.normal_ws;
                float3 h = normalize(v + l);
                float NoL = saturate(nol_sun);
                float NoV = saturate(dot(n, v));
                float NoH = saturate(dot(n, h));
                float LoH = saturate(dot(l, h));

                float3 brdf = 0.0f;

                #if !SPECULAR_ONLY
                // Diffuse
                brdf += diffuse_color * Fd_Lambert();
                #endif

                #if !DIFFUSE_ONLY
                // Specualr
                float D = D_GGX(NoH, roughness);
                float3 F = F_Schlick(LoH, specular_color);
                float V = vis_smith_G2_height_correlated_GGX(NoL, NoV, roughness);
                brdf += min(D * V, F32_SIGNIFICANTLY_LARGE) * F;
                #endif

                radiance += (throughput * nol_sun) * brdf * frame_params.sun_inten.rgb;
            }
        }
        #endif

#if 0
        // Material debug
        debug_color.rgb = specular_color;
        debug_color.rgb = float3(hit.perceptual_roughness, hit.metallic, 0.0f);
        debug_color.a = 1.0f;
        break;
#endif

        // View direction in BRDF context
        const float3 V = -ray.Direction;

        // Pick BRDF lobe (importance sampling alomgs lobes with heuristics)
        // NOTE: we just add diffse and specular components, thus we multiply 1/pdf to throughput 
        // ref: RTGII, ch14
        float prop_specular;
        {
            float NoV = saturate(dot(hit.normal_ws, V));
            float f90_luminance = luminance(specular_color);
            float e_spec = F_Schlick_single(NoV, f90_luminance);
            float e_diff = luminance(diffuse_color);
            // NOTE: okay to divide by zero, since we clamp prop_specular below
            prop_specular = e_spec / (e_spec + e_diff);
            // remove extreme value of prop_specular to avoid undersampling specular/diffuse
            prop_specular = clamp(prop_specular, 0.1f, 0.9f);
#if SPECULAR_ONLY
            prop_specular = 1.0f;
#endif
#if DIFFUSE_ONLY
            prop_specular = 0.0f;
#endif
        }
        bool brdf_is_specular = lcg_rand(rng_state) < prop_specular;
        if (brdf_is_specular)
        {
            throughput /= prop_specular;
        } else {
            throughput /= (1.0 - prop_specular);
        }

        // Generate bounce sample direction, and calculate sample weight
        float3 brdf_NoL_over_pdf; // := BRDF() * NoL / pdf, also call `sample weight`
        float3 L_local; // bounced direction for next ray, in local space
        float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
        float4 rot_to_local = get_rotation_to_z_from(hit.normal_ws);
        if (brdf_is_specular) 
        {
            float3 V_local = rotate_point(rot_to_local, V);

            // Sample a half vector (microfacet normal) from the GGX distribution of visible normals (VNDF).
            // pdf_H = D_visible(H) = ( G1(V) * VoH * D(H) / NoV ) 
            // Note: this method allow V_local.z < 0 (viewing below the surface), but such
            // case is eliminated during hit shader (by flipping the normal).
            // ref: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h#L727
            // [Heitz 2014 "Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals", section 2]
            // TODO compare methods [Heitz 2018 "Sampling the GGX Distribution of Visible Normals"]
            float3 H_local;
            if (roughness != 0.0f)
            {
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

            // After relfect operator
            // pdf_L = pdf_H * Jacobian_refl(V, H) 
            //        = ( G1(V) * VoH * D(H) / NoV ) * ( 1 / (4 * VoH) )
            //        = G1(V) * D(H) / (4 * NoV)
            L_local = reflect(-V_local, H_local);

            float LoH = saturate(dot(L_local, H_local));
            float3 N_local = float3(0.0f, 0.0f, 1.0f);
            float NoL = max(F32_SIGNIFICANTLY_SMALL, dot(N_local, L_local)); // L_local.z
            float NoV = max(F32_SIGNIFICANTLY_SMALL, dot(N_local, V_local)); // V_local.z

            float3 F = F_Schlick(LoH, specular_color);

            float G2_over_G1 = smith_G2_over_G1_height_correlated_GGX(NoL, NoV, roughness);

            // Original formular:
            // pdf               = G1 * D / (4 * NoV)
            // brdf              = F * D * G2 / (4 * NoV * NoL);
            // brdf_NoL_over_pdf = bdrf * NoL / pdf
            //                   = F * D * G2 * NoL / (4 * NoV * NoL * pdf)
            //                   = F * G2 / G1
            brdf_NoL_over_pdf = F * G2_over_G1;
        }
        else
        {
            float pdf;
            L_local = sample_hemisphere_cosine(u, pdf);

            // Original formular:
            // pdf               = NoL / PI 
            // brdf              = diffuse_color / PI
            // brdf_NoL_over_pdf = brdf * NoL / pdf
            //                   = diffuse_color * NoL / (PI * pdf)
            //                   = diffuse_color
            brdf_NoL_over_pdf = diffuse_color;
        }
        ray.Direction = rotate_point(invert_rotation(rot_to_local), L_local);
        ray.Direction = normalize(ray.Direction);

        // Terminate poth with zero contribution
        if (max3(brdf_NoL_over_pdf) <= 0.0f)
        {
            break;
        }

        // Terminate invalid path (tracing down-ward due to normal mapping)
        if (dot(ray.Direction, hit.normal_geo_ws) <= 0.0f)
        {
            break;
        }

        throughput *= brdf_NoL_over_pdf;

        // Update ray origin for next bounce
        //ray.Origin += ray.Direction * hit.hit_t;
        ray.Origin = hit.position_ws;
    }

    // Update accumulation buffer
    float3 accu_radiance = radiance;
    if (pc.accumulated_count > 0)
    {
        accu_radiance += rw_accumulated[dispatch_id].xyz;
    }
    rw_accumulated[dispatch_id.xy] = float4(accu_radiance, 1.0f);

    // Write output for this frame
    float3 avg_radiance = accu_radiance / (pc.accumulated_count + 1);
    rw_lighting[dispatch_id.xy] = avg_radiance;

#if 0
    if (debug_color.a > 0.0f)
    rw_lighting[dispatch_id.xy] = debug_color;
#endif
}