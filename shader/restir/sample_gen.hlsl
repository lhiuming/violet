#include "../brdf.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"
#include "../scene_bindings.hlsl"
#include "../util.hlsl"

#include "reservoir.hlsl"

RaytracingAccelerationStructure scene_tlas;
Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
TextureCube<float4> skycube;
Texture2D<float3> prev_color;
Texture2D<float> prev_depth;
StructuredBuffer<Reservoir> prev_reservoir_temporal_buffer;
RWStructuredBuffer<Reservoir> rw_reservoir_temporal_buffer;
RWTexture2D<float4> rw_debug_texture;

struct PushConstants
{
    uint frame_index;
    uint has_prev_frame;
//    uint accumulated_count;
};
[[vk::push_constant]]
PushConstants pc;

struct PayloadSecondary {
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
    float depth = gbuffer_depth[dispatch_id.xy];

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    // early out if no geometry
    if (has_no_geometry_via_depth(depth))
    {
        // Set null reservoir to avoid stale sample during temporal accumulation
        uint buffer_index = buffer_size.x * dispatch_id.y + dispatch_id.x;
        rw_reservoir_temporal_buffer[buffer_index] = null_reservoir();
        return;
    }

    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);

	// world position reconstruction from depth buffer
    float depth_error = 1.f / 16777216.f; // 24 bit unorm depth 
    float2 screen_pos = (dispatch_id.xy + 0.5f) / float2(buffer_size);
	float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_pos * 2.0f - 1.0f, depth + depth_error, 1.0f));
	float3 position_ws = position_ws_h.xyz / position_ws_h.w;
    
    uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

    // Generate Sample Point

    // uniform hemisphere sampling
    // TODO blue noise
    float3 sample_dir;
    {
        float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
        float3 L_local = sample_hemisphere_uniform(u);
        float4 rot_from_local = get_rotation_to_z_from(gbuffer.normal);
        sample_dir = rotate_point(invert_rotation(rot_from_local), L_local);
    }

    PayloadSecondary payload;
    payload.missed = false;
    RayDesc ray;
    ray.Origin = position_ws;
    ray.Direction = sample_dir;
    ray.TMin = 0.0005f; // 0.5mm
    ray.TMax = 1000.0f;
    TraceRay(scene_tlas,
            RAY_FLAG_FORCE_OPAQUE // skip anyhit
            ,
            0xff, // uint InstanceInclusionMask,
            0, // uint RayContributionToHitGroupIndex,
            0, // uint MultiplierForGeometryContributionToHitGroupIndex,
            0, // uint MissShaderIndex,
            ray, // RayDesc Ray,
            payload // inout payload_t Payload
        );

    // Compute Radiance for the sample point

    float3 debug_color = 0.0f;

    float3 radiance = 0.0f;
    bool recursive = false;
    if (payload.missed) {
        radiance = skycube.SampleLevel(sampler_linear_clamp, ray.Direction, 0.0f).rgb;

        // Construct a hit point at skybox if miss
        payload.position_ws = position_ws + ray.Direction * ray.TMax;
        payload.normal_ws = -ray.Direction;
    }
    else 
    {
        radiance = 0.0;

        // try to read from prev frame color
        // TODO rejection and stuff
        if (pc.has_prev_frame) {
            float4 prev_hpos = mul(frame_params.prev_view_proj, float4(payload.position_ws, 1.0f));
            float2 ndc = prev_hpos.xy / prev_hpos.w;
            float reproj_depth = prev_hpos.z / prev_hpos.w;

            bool in_view = all(abs(ndc.xy) < 1.0);
            if (in_view)
            {
                // nearest neighbor sampling
	            float2 prev_screen_uv = ndc.xy * 0.5f + 0.5f;
                uint2 prev_pixel_pos = uint2(floor(prev_screen_uv * buffer_size));

                float prev_depth_value = prev_depth[prev_pixel_pos];
                float DEPTH_TOLERANCE = 0.0005f; // TODO compare in world space?
                if (abs(reproj_depth - prev_depth_value) < DEPTH_TOLERANCE) {
                    radiance = prev_color[prev_pixel_pos].rgb;
                }
            }
        } 

        // TODO radiance cache
    }

    const RestirSample new_sample = make_restir_sample(
        position_ws,
        gbuffer.normal,
        payload.position_ws,
        payload.normal_ws,
        radiance
    );

    // Reservoir Temporal Resampling

    Reservoir reservoir;
    float new_target_pdf = luminance(new_sample.hit_radiance);
    float new_w = new_target_pdf * TWO_PI; // source_pdf = 1 / TWO_PI;
    if (pc.has_prev_frame)
    {
        // read reservoir from prev frame
        // TODO reproject
        uint2 prev_pos = dispatch_id.xy;
        if (0)
        {
            // permutation sampling
            // TODO add some per-frame jitter
            prev_pos.xy ^= 3;
        }
        prev_pos = clamp(prev_pos, uint2(0, 0), buffer_size - uint2(1, 1));
        uint buffer_index = buffer_size.x * prev_pos.y + prev_pos.x;
        reservoir = prev_reservoir_temporal_buffer[buffer_index];

        // Bound the temporal information to avoid stale sample
        if (1)
        {
            const uint M_MAX = 20; // [Benedikt 2020]
            //const uint M_MAX = 30; // [Ouyang 2021]
            //const uint M_MAX = 10; // [h3r2tic 2022]
            reservoir.M = min(reservoir.M, M_MAX);
        }

        float reservoir_target_pdf = luminance(reservoir.z.hit_radiance);
        float w_sum = reservoir.W * reservoir_target_pdf * float(reservoir.M);

        // update reservoir with new sample
        w_sum += new_w;
        reservoir.M += 1;
        float chance = new_w / w_sum;
        if (lcg_rand(rng_state) < chance) {
            reservoir.z = new_sample;
            reservoir_target_pdf = new_target_pdf;
        }

        // update the W
        if (reservoir_target_pdf > 0.0) // avoid deviding zero
        {
            reservoir.W = w_sum / ( reservoir_target_pdf * float(reservoir.M) );
        }
        else 
        {
            // if reservoir_target_pdf is 0, it may not come from new_target_pdf (chance will be 0 or NaN in that case), therefore previously target_pdf (and w_sum) must be 0. Then new_target_pdf must be 0 too (otherwise chance is 1).
            reservoir.W = 1.0f / float(reservoir.M);
        }
    } 
    else 
    {
        // New reservoir 
        uint M = 1;
        uint W = TWO_PI;  // := w / (target_pdf * M); 
        reservoir = init_reservoir(new_sample, M, W);
    }

    // store updated reservoir
    uint buffer_index = buffer_size.x * dispatch_id.y + dispatch_id.x;
    rw_reservoir_temporal_buffer[buffer_index] = reservoir;

#if 0
    float3 selected_dir = normalize(reservoir.z.hit_pos - position_ws);
    float NoL = saturate(dot(gbuffer.normal, selected_dir));
    float brdf = ONE_OVER_PI;
    float3 RIS_estimator_diffuse = reservoir.z.hit_radiance * brdf * NoL * reservoir.W ;
    rw_debug_texture[dispatch_id] = float4(RIS_estimator_diffuse, 1.0f);
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
void closesthit(inout PayloadSecondary payload, in Attribute attr) {
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
void miss(inout PayloadSecondary payload)
{
    payload.missed = true;
}