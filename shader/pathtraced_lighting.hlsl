#include "gbuffer.hlsl"
#include "scene_bindings.hlsl"

/* 
This is a reference path tracer, not optimized for performance.
Reference: 
    - Ray Tracing Gems II, Chapter 14, Reference Path Tracer.
*/

RaytracingAccelerationStructure scene_tlas;
Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
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

#define PI 3.14159265359f
#define TWO_PI (2.0f * PI)
#define ONE_OVER_PI (1.0f / PI)

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
float4 get_rotation_to_z(float3 v) {
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



struct Payload {
    bool missed;
    float hit_t;
    float3 normal_ws;
    float3 geometry_normal_ws;
    float3 position_ws;
};

[shader("raygeneration")]
void raygen() {
    uint2 dispatch_id = DispatchRaysIndex().xy;
    float depth = gbuffer_depth[dispatch_id.xy];

    // early out if no geometry
    if (depth == 0)
        return;

    uint4 gbuffer_enc = gbuffer_color[dispatch_id.xy];
    GBuffer gbuffer = decode_gbuffer(gbuffer_enc);

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

	// world position reconstruction from depth buffer
    float depth_error = 1.f / 16777216.f; // 24 bit unorm depth 
    float2 screen_pos = (dispatch_id.xy + 0.5f) / float2(buffer_size);
	float4 position_ws_h = mul(view_params.inv_view_proj, float4(screen_pos * 2.0f - 1.0f, depth + depth_error, 1.0f));
	float3 position_ws = position_ws_h.xyz / position_ws_h.w;

    float3 radiance = 0.0f;

    // TODO
    float3 light_color = float3(0.7f, 0.7f, 0.6f) * PI;

    // Add Direct Lighting
    {
        RayDesc shadow_ray;
        shadow_ray.Origin = position_ws;
        shadow_ray.Direction = view_params.sun_dir;
        shadow_ray.TMin = 0.0005f; // 0.5mm
        shadow_ray.TMax = 100.0f;

        Payload shadow = (Payload)0;
        TraceRay(scene_tlas,
            RAY_FLAG_FORCE_OPAQUE // skip anyhit
            | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 
            0xff, // uint InstanceInclusionMask,
            0, // uint RayContributionToHitGroupIndex,
            0, // uint MultiplierForGeometryContributionToHitGroupIndex,
            0, // uint MissShaderIndex,
            shadow_ray,
            shadow
        );

        if (shadow.missed)
        {
            float nol = dot(gbuffer.normal, view_params.sun_dir);
            //float3 brdf = gbuffer.color * ONE_OVER_PI;
            float3 brdf = 1.0 * ONE_OVER_PI;
            radiance += nol * brdf * light_color;
        }
     }


    uint rng_state = init_rng(dispatch_id.xy, buffer_size, pc.frame_index);

    RayDesc ray;
    ray.Origin = position_ws;
    ray.TMin = 0.0005f; // 0.5mm
    ray.TMax = 100.0f;
    float3 normal_ws = gbuffer.normal;
    float3 normal_geo = gbuffer.normal; // TODO
    float3 throughput = 1.0f;

    // Ray tracing loop
    const int MAX_BOUNCE = 8;
    for (int bounce = 0; bounce < MAX_BOUNCE; ++bounce) 
    {
        // Generate sample direction 
        float pdf;
        float2 u = float2(rand(rng_state), rand(rng_state));
        float3 dir_local = sample_hemisphere_cosine(u, pdf);
        float4 rot_q = get_rotation_to_z(normal_ws);
        ray.Direction = rotate_point(invert_rotation(rot_q), dir_local);

        // Terminate invalid path
        if (dot(ray.Direction, normal_geo) <= 0.0f)
        {
            break;
        }

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
            //skycolor = lerp(skycolor, 1.0f, 0.999999);
            radiance += throughput * skycolor;
            break;
        }

        if (bounce == (MAX_BOUNCE - 1)) {
            break;
        }

        // Evaluate BRDF
        float3 diffuse_color = 1.0f;
        float3 brdf = diffuse_color * ONE_OVER_PI;
        throughput *= brdf / pdf;

        // Add Direct Lighting
        float nol_sun = dot(hit.normal_ws, view_params.sun_dir) > 0.0;
        if (nol_sun > 0.0f)
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

            if (shadow.missed && (bounce == 0))
            {
                radiance += nol_sun * brdf * light_color;
            }
        }

        // Update position for next bounce;
        ray.Origin += ray.Direction * hit.hit_t;
        //ray.Origin = payload.position_ws;

        normal_ws = hit.normal_ws;
        normal_geo = hit.geometry_normal_ws;
    }

    float3 accu_radiance = radiance;
    if (pc.accumulated_count > 1)
    {
        accu_radiance += rw_accumulated[dispatch_id].xyz;
    }
    rw_accumulated[dispatch_id.xy] = float4(accu_radiance, 1.0f);

    float3 avg_radiance = accu_radiance / pc.accumulated_count;
    rw_lighting[dispatch_id.xy] = float4(avg_radiance, 1.0f);

#if 0
    rw_lighting[dispatch_id.xy] = float4(debug, 1.0f);
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

#define INTERPOLATE_MESH_ATTR(type, attr_offset, indicies, bary) \
(load_##type(attr_offset, indicies.x) * bary.x \
+load_##type(attr_offset, indicies.y) * bary.y \
+load_##type(attr_offset, indicies.z) * (1.0f - bary.x - bary.y))

struct Attribute
{
    float2 bary;
};

[shader("closesthit")]
void closesthit(inout Payload payload, in Attribute attr) {
    uint mesh_index = GeometryIndex(); // index of geometry in BLAS; we are using only one blas for all loaded mesh
    uint triangle_index = PrimitiveIndex();

    MeshParams mesh = mesh_params[mesh_index];
    uint index0 = index_buffer[triangle_index * 3];
    uint index1 = index_buffer[triangle_index * 3 + 1];
    uint index2 = index_buffer[triangle_index * 3 + 2];
    uint3 indicies = uint3(index0, index1, index2);

    float2 uv = INTERPOLATE_MESH_ATTR(float2, mesh.texcoords_offset, indicies, attr.bary);
    float3 pos = INTERPOLATE_MESH_ATTR(float3, mesh.positions_offset, indicies, attr.bary);
    float3 normal = INTERPOLATE_MESH_ATTR(float3, mesh.normals_offset, indicies, attr.bary);
    float4 tangent = INTERPOLATE_MESH_ATTR(float4, mesh.tangents_offset, indicies, attr.bary);
	float3 bitangent = normalize(tangent.w * cross(normal, tangent.xyz));

    MaterialParams mat = material_params[mesh.material_index];
	float4 base_color = bindless_textures[mat.base_color_index].SampleLevel(sampler_linear_clamp, uv, 0);
    float4 normal_map = bindless_textures[mat.normal_index].SampleLevel(sampler_linear_clamp, uv, 0);

	// normal mapping
	float3 normal_ts = normal_map.xyz * 2.0f - 1.0f;
	float3 normal_ws = normalize( normal_ts.x * tangent.xyz + normal_ts.y * bitangent + normal_ts.z * normal );

    // Position
    //float3 position_ws = mul(ObjectToWorld3x4(), float4(pos, 1.0f));
    float3 position_ws = pos;

    // Geometry normal
    float3 pos0 = load_float3(mesh.positions_offset, indicies.x);
    float3 pos1 = load_float3(mesh.positions_offset, indicies.x);
    float3 pos2 = load_float3(mesh.positions_offset, indicies.x);
    float3 normal_geo = normalize(cross(pos1 - pos0, pos2 - pos0));

    payload.missed = false;
    payload.hit_t = RayTCurrent();
    payload.normal_ws = normal_ws;
    payload.geometry_normal_ws = normal_geo;
    payload.position_ws = position_ws;
}

[shader("miss")]
void miss(inout Payload payload)
{
    payload.missed = true;
    /*
    float3 dir_ws = WorldRayDirection();
    float3 sky = skycube.SampleLevel(sampler_linear_clamp, dir_ws, 0).xyz;
    payload.radiance = sky;
    */
}