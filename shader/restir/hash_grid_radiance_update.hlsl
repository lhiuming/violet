#include "../constants.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"

#include "hash_grid.inc.hlsl"

RWStructuredBuffer<HashGridCell> rw_hash_grid_storage_buffer;
#define RAYTRACE_INC_HASH_GRID_STORATE_BUFFER rw_hash_grid_storage_buffer
#include "raytrace.inc.hlsl"

StructuredBuffer<HashGridQuery> hash_grid_query_buffer;
StructuredBuffer<uint> hash_grid_query_counter_buffer;

RWBuffer<uint> rw_hash_grid_decay_buffer; // R8_UINT

[[vk::push_constant]]
struct PC {
    uint frame_hash;
} pc;

[shader("raygeneration")]
void main()
{
    uint dispatch_id = DispatchRaysIndex().x;

    // TODO indirect dispatch
    if (dispatch_id >= hash_grid_query_counter_buffer[0])
    {
        return;
    }

    HashGridQuery query = hash_grid_query_buffer[dispatch_id];

    // Find or allocate a cell
    // Nothing to do if the hash grid is full
    uint cell_addr;
    if ( !hash_grid_find_or_insert(rw_hash_grid_storage_buffer, query.cell_hash, query.cell_checksum, cell_addr) )
    {
        return;
    }

    uint lcg_state = jenkins_hash(dispatch_id ^ pc.frame_hash);
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

    // TODO average and pack to r11g11b10 in later pass
    float weight = 1.0f;
    uint3 radiance_u = HashGridCell::radiance_scale(radiance, weight);
    InterlockedAdd(rw_hash_grid_storage_buffer[cell_addr].radiance_acc_r, radiance_u.r);
    InterlockedAdd(rw_hash_grid_storage_buffer[cell_addr].radiance_acc_g, radiance_u.g);
    InterlockedAdd(rw_hash_grid_storage_buffer[cell_addr].radiance_acc_b, radiance_u.b);
    InterlockedAdd(rw_hash_grid_storage_buffer[cell_addr].weight_acc, uint(weight));

    // Refresh cell decay
    rw_hash_grid_decay_buffer[cell_addr] = 10;
}