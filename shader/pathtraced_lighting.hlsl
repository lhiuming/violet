#include "gbuffer.hlsl"
#include "scene_bindings.hlsl"
#include "brdf.hlsl"

/* 
This is a reference path tracer, not optimized for performance.
Reference: 
    - Ray Tracing Gems II, Chapter 14, Reference Path Tracer.
*/

#define MAX_BOUNCE 8
#define MAX_BOUNCE_WITH_LIGHTING MAX_BOUNCE

#define PRIMARY_RAY_JITTER 1
#define WHITE_FURNACE 0

RaytracingAccelerationStructure scene_tlas;
TextureCube<float4> skycube;
RWTexture2D<float4> rw_accumulated;
RWTexture2D<float4> rw_lighting;

struct PushConstants
{
    uint frame_index;
    uint accumulated_count;
};
[[vk::push_constant]]
PushConstants pc;


// ----------------
// PNG for sampling

// from: Ray Tracing Gems II, Chapter 14
uint jenkins_hash(uint x) {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    return x;
}

// from: Ray Tracing Gems II, Chapter 14
uint init_rng(uint2 pixel_coords, uint2 resolution, uint frame) {
    uint rng_state = dot(pixel_coords, uint2(1, resolution.x)) ^ jenkins_hash(frame);
    return jenkins_hash(rng_state);
}

float uint_to_float(uint x) {
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

uint xorshift(inout uint x /* rng state */) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

float rand(inout uint rng_state) {
    return uint_to_float(xorshift(rng_state));
}

// -------------
// BRDF sampling

// modified: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h
float3 sample_hemisphere_cosine(float2 u, out float pdf)
{
    float sin_theta = sqrt(u.x);
    float cos_theta = sqrt(1.0f - u.x); 

    pdf = cos_theta * ONE_OVER_PI;

    float phi = TWO_PI * u.y;
    float sin_phi, cos_phi;
    sincos(phi, sin_phi, cos_phi);

    return float3(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    );
}

// ----------
// Quaternion

// modified: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h
float4 get_rotation_to_z_from(float3 v) {
    if (v.z < -0.99999f)
        return float4(1.0f, 0.0f, 0.0f, 0.0f);
    return normalize(float4(v.y, -v.x, 0.0f, 1.0f + v.z));
}

// modified: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h
float4 invert_rotation(float4 q)
{
    return float4(-q.xyz, q.w);
}

// modified: https://github.com/boksajak/referencePT/blob/master/shaders/brdf.h
float3 rotate_point(float4 q, float3 v) {
    float3 axis = q.xyz;
	return 2.0f * dot(axis, v) * axis + (q.w * q.w - dot(axis, axis)) * v + 2.0f * q.w * cross(axis, v);
}


// Miscs
float luminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}
float max3(float3 v) {
    return max(max(v.x, v.y), v.z);
}

struct Payload {
    bool missed;
    float hit_t;
    float3 normal_ws;
    float3 normal_geo_ws;
    float3 position_ws;
    float3 base_color;
    float metallic;
    float perceptual_roughness;
    uint mesh_index;
    uint triangle_index;
};

[shader("raygeneration")]
void raygen() {
    uint2 dispatch_id = DispatchRaysIndex().xy;

    uint2 buffer_size;
    rw_accumulated.GetDimensions(buffer_size.x, buffer_size.y);

    uint rng_state = init_rng(dispatch_id.xy, buffer_size, pc.frame_index);

    // Primary ray direction
    float2 pix_coord = float2(dispatch_id) + 0.5f;
    #if PRIMARY_RAY_JITTER
    pix_coord += lerp(-0.5f.xx, 0.5f.xx, float2(rand(rng_state), rand(rng_state)));
    #endif
    float2 ndc = pix_coord / float2(buffer_size) * 2.0f - 1.0f;
    float4 view_dir_end_h = mul(view_params.inv_view_proj, float4(ndc, 1.0f, 1.0f));
    float3 view_dir_end = view_dir_end_h.xyz / view_dir_end_h.w;
    float3 view_dir = normalize(view_dir_end - view_params.view_pos);

    // TODO Frame Parameters
    float exposure = 20;
    float3 light_color = float3(0.7f, 0.7f, 0.6f) * PI * exposure;

    // Setup primary Ray
    RayDesc ray;
    ray.Origin = view_params.view_pos;
    ray.Direction = view_dir;
    ray.TMin = 0.0005f; // 0.5mm
    ray.TMax = 100.0f;

    float3 throughput = 1.0f;
    float3 radiance = 0.0f;

    // Ray tracing loop

    float3 debug_color = 0.0f;
    for (int bounce = 0; bounce < MAX_BOUNCE; ++bounce) 
    {
        debug_color.r = float(bounce) / MAX_BOUNCE;

        // Trace
        Payload hit = (Payload)0;
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

        // Add Direct Lighting
        #if !WHITE_FURNACE
        float nol_sun = dot(hit.normal_ws, view_params.sun_dir);
        if ((nol_sun > 0.0f) && (bounce < MAX_BOUNCE_WITH_LIGHTING))
        {
            RayDesc shadow_ray;
            shadow_ray.Origin = hit.position_ws;
            shadow_ray.Direction = view_params.sun_dir;
            shadow_ray.TMin = 0.0005f; // 0.5mm
            shadow_ray.TMax = 100.0f;

            Payload shadow = (Payload)0;
            shadow.missed = false;
            TraceRay(scene_tlas,
                RAY_FLAG_FORCE_OPAQUE // skip anyhit
                | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER // skip closesthit
                | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH // shadow
                , 
                0xff, // uint InstanceInclusionMask,
                0, // uint RayContributionToHitGroupIndex,
                0, // uint MultiplierForGeometryContributionToHitGroupIndex,
                0, // uint MissShaderIndex,
                shadow_ray,
                shadow
            );

            if (shadow.missed)
            {
                // TODO specular
                float3 brdf = hit.base_color * ONE_OVER_PI;
                radiance += (throughput * nol_sun) * brdf * light_color;
            }
        }
        #endif

        // Material attribute decode
	    const float3 specular_color = lerp(float3(0.04f, 0.04f, 0.04f), hit.base_color, hit.metallic);
	    const float3 diffuse_color = hit.base_color* (1.0f - hit.metallic);

        // Pick brdf lobe (importance sampling the lobe with heuristics)
        // ref: RTGII, ch14
        float prop_specular;
        {
            float3 income_dir = -ray.Direction;
            float NoV = max(0.0f, dot(hit.normal_ws, income_dir));
            float f90_luminance = luminance(specular_color); // TODO mul by throughput?
            float e_spec = F_Schlick(NoV, f90_luminance.xxx).x;
            float e_diff = luminance(diffuse_color);
            prop_specular = e_spec / max(0.0000001f, e_spec + e_diff);
            // remove extreme value of prop_specular to avoid undersampling specular/diffuse
            prop_specular = clamp(prop_specular, 0.1f, 0.9f);
        }
        bool brdf_is_specular = rand(rng_state) < prop_specular;
        if (brdf_is_specular)
        {
            throughput /= prop_specular;
        } else {
            throughput /= (1.0 - prop_specular);
        }

        // Generate bounce sample direction 
        float3 brdf_NoL_over_pdf;
        float3 new_dir;
        float2 u = float2(rand(rng_state), rand(rng_state));
        float4 rot_to_local = get_rotation_to_z_from(hit.normal_ws);
        if (brdf_is_specular) 
        {
            float3 prev_dir_local = rotate_point(rot_to_local, ray.Direction);

            float3 H_local = 0.0f; // TODO
            float3 L_local = reflect(prev_dir_local, H_local);
            // TODO 
            brdf_NoL_over_pdf = 0.0f;
        }
        else
        {
            float pdf;
            float3 dir_local = sample_hemisphere_cosine(u, pdf);
            new_dir = rotate_point(invert_rotation(rot_to_local), dir_local);
#if NOT_OPTIMIZE
            float3 brdf = diffuse_color * ONE_OVER_PI;
            float NoL = dot(hit.normal_ws, new_dir);
            brdf_NoL_over_pdf = brdf * NoL / pdf;
#else
            // pdf = NoL / PI 
            brdf_NoL_over_pdf = diffuse_color;
#endif
        }
        ray.Direction = new_dir;

        // Terminate poth with zero contribution
        if (max3(brdf_NoL_over_pdf) <= 0.0f)
        {
            break;
        }

        // Terminate invalid path
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
    if (pc.accumulated_count > 1)
    {
        accu_radiance += rw_accumulated[dispatch_id].xyz;
    }
    rw_accumulated[dispatch_id.xy] = float4(accu_radiance, 1.0f);

    // Write output for this frame
    float3 avg_radiance = accu_radiance / pc.accumulated_count;
    rw_lighting[dispatch_id.xy] = float4(avg_radiance, 1.0f);

#if 0
    rw_lighting[dispatch_id.xy] = float4(debug_color, 1.0f);
#endif
}


// ----------------
// Geometry loading

float2 load_float2(uint attr_offset, uint vert_id) {
    uint2 words = uint2(
        vertex_buffer[attr_offset + vert_id * 2 + 0],
		vertex_buffer[attr_offset + vert_id * 2 + 1]
        );
	return asfloat(words);
}

float3 load_float3(uint attr_offset, uint vert_id) {
    uint3 words = uint3(
        vertex_buffer[attr_offset + vert_id * 3 + 0],
		vertex_buffer[attr_offset + vert_id * 3 + 1],
		vertex_buffer[attr_offset + vert_id * 3 + 2]
        );
	return asfloat(words);
}

float4 load_float4(uint attr_offset, uint vert_id) {
    uint4 words = uint4(
        vertex_buffer[attr_offset + vert_id * 4 + 0],
		vertex_buffer[attr_offset + vert_id * 4 + 1],
		vertex_buffer[attr_offset + vert_id * 4 + 2],
		vertex_buffer[attr_offset + vert_id * 4 + 3]
        );
	return asfloat(words);
}

// NOTE: bary.x is weight for vertex_1, bary.y is weight for vertex_2
#define INTERPOLATE_MESH_ATTR(type, attr_offset, indicies, bary) \
(load_##type(attr_offset, indicies.x) * (1.0f - bary.x - bary.y) \
+load_##type(attr_offset, indicies.y) * bary.x \
+load_##type(attr_offset, indicies.z) * bary.y)

struct Attribute
{
    float2 bary;
};

[shader("closesthit")]
void closesthit(inout Payload payload, in Attribute attr) {
    uint mesh_index = GeometryIndex(); // index of geometry in BLAS; we are using only one blas for all loaded mesh
    uint triangle_index = PrimitiveIndex();

    MeshParams mesh = mesh_params[mesh_index];
    uint3 indicies = uint3(
        index_buffer[mesh.index_offset + triangle_index * 3 + 0],
        index_buffer[mesh.index_offset + triangle_index * 3 + 1],
        index_buffer[mesh.index_offset + triangle_index * 3 + 2]);

    float2 uv = INTERPOLATE_MESH_ATTR(float2, mesh.texcoords_offset, indicies, attr.bary);
    float3 pos = INTERPOLATE_MESH_ATTR(float3, mesh.positions_offset, indicies, attr.bary);
    float3 normal = INTERPOLATE_MESH_ATTR(float3, mesh.normals_offset, indicies, attr.bary);
    float4 tangent = INTERPOLATE_MESH_ATTR(float4, mesh.tangents_offset, indicies, attr.bary);
    // TODO calculate before interpolation?
	float3 bitangent = normalize(tangent.w * cross(normal, tangent.xyz));

    MaterialParams mat = material_params[mesh.material_index];
	float4 base_color = bindless_textures[mat.base_color_index].SampleLevel(sampler_linear_clamp, uv, 0);
	float4 metal_rough = bindless_textures[mat.metallic_roughness_index].SampleLevel(sampler_linear_clamp, uv, 0);
    float4 normal_map = bindless_textures[mat.normal_index].SampleLevel(sampler_linear_clamp, uv, 0);

	// normal mapping
	float3 normal_ts = normal_map.xyz * 2.0f - 1.0f;
	float3 normal_ws = normalize( normal_ts.x * tangent.xyz + normal_ts.y * bitangent + normal_ts.z * normal );

    // World position
    float3 position_ws = mul(WorldToObject3x4(), float4(pos, 1.0f));
    //position_ws = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    // Geometry normal
    float3 pos0 = load_float3(mesh.positions_offset, indicies.x);
    float3 pos1 = load_float3(mesh.positions_offset, indicies.y);
    float3 pos2 = load_float3(mesh.positions_offset, indicies.z);
    float3 normal_geo = normalize(cross(pos1 - pos0, pos2 - pos0));

    // Two-side geometry: flip the geometry attribute if ray hit from back face
    if (dot(normal_geo, WorldRayDirection()) > 0.0f)
    {
        normal_geo = -normal_geo;
        normal_ws = -normal_ws;
    }

    payload.missed = false;
    payload.hit_t = RayTCurrent();
    payload.normal_ws = normal_ws;
    payload.normal_geo_ws = normal_geo;
    payload.position_ws = position_ws;
#if WHITE_FURNACE
    payload.base_color = 1.0f;
#else
    payload.base_color = base_color.xyz;
#endif
    payload.metallic = metal_rough.b;
    payload.perceptual_roughness = metal_rough.g;
    payload.mesh_index = mesh_index;
    payload.triangle_index = triangle_index;
}

[shader("miss")]
void miss(inout Payload payload)
{
    payload.missed = true;
}