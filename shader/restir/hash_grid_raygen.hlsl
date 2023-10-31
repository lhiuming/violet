#include "../constants.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"

#include "hash_grid.inc.hlsl"

// TODO remove the rw prefix (shader reflection should check `Sampled` from SPIR-V, and warn if not bound correctly)
RWStructuredBuffer<HashGridCell> rw_hash_grid_storage_buffer;
#define RAYTRACE_INC_HASH_GRID_STORATE_BUFFER rw_hash_grid_storage_buffer
#include "raytrace.inc.hlsl"

StructuredBuffer<HashGridQuery> hash_grid_query_buffer;
StructuredBuffer<uint> compact_query_counter_buffer;
StructuredBuffer<uint> compact_query_index_buffer;
StructuredBuffer<uint> compact_query_cell_addr_buffer;

RWBuffer<uint> rw_hash_grid_decay_buffer; // R8_UINT

[[vk::push_constant]]
struct PC {
    uint frame_hash;
} pc;

[shader("raygeneration")]
void main()
{
    uint dispatch_id = DispatchRaysIndex().x;

    // TODO indirect trace?
    if (dispatch_id >= compact_query_counter_buffer[0])
    {
        return;
    }

    uint query_index = compact_query_index_buffer[dispatch_id];
    uint cell_addr = compact_query_cell_addr_buffer[dispatch_id];

    uint lcg_state = jenkins_hash(dispatch_id ^ pc.frame_hash);

    HashGridQuery query = hash_grid_query_buffer[query_index];
    float3 normal = query.hit_normal();

    // Genreate new sample for the query
    float3 ray_dir;
    {
        float2 u = float2(lcg_rand(lcg_state), lcg_rand(lcg_state));
        float ray_pdf;
        float3 ray_dir_local = sample_hemisphere_cosine(u, ray_pdf);
        float4 rot_from_local = invert_rotation(get_rotation_to_z_from(normal));
        ray_dir = rotate_point(rot_from_local, ray_dir_local);
    }

    // Trace
    // TODO maybe sample current frame for indirect diffuse?
    RadianceTraceResult trace_result = trace_radiance(query.hit_position, ray_dir, true);

    // Calculate lighting (without diffuse_rho)
    // original formular: 
    //   surface_radiance = trace_result.radiance * (brdf * NoL / ray_pdf)
    // but: 
    //   ray_pdf := NoL * ONE_OVER_PI == NoL * brdf
    float3 radiance = trace_result.radiance;

    // Update hash grid cell
    // DEBUG
    #if 0
    radiance = min(radiance, 0.0f);
    radiance += float3(0, 1, 0);
    #endif

    // Store radiance 
    // TODO temporal filtering
    float3 prev_radiance = rw_hash_grid_storage_buffer[cell_addr].radiance;
    float blend_factor = 1.0f;
    float3 blend_radiance = radiance * blend_factor + prev_radiance * (1.0 - blend_factor);
    rw_hash_grid_storage_buffer[cell_addr].radiance = blend_radiance;

    // Refresh cell decay
    rw_hash_grid_decay_buffer[cell_addr] = 10;
}