#include "../frame_bindings.hlsl"
#include "../util.hlsl"

#include "hash_grid.inc.hlsl"

Texture2D<float> gbuffer_depth;
StructuredBuffer<HashGridCell> hash_grid_storage_buffer;
RWTexture2D<float3> rw_color;

[[vk::push_constant]]
struct PC
{
    uint color_code;
} pc;

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    float depth = gbuffer_depth[dispatch_id.xy];
    if (has_no_geometry_via_depth(depth))
    {
        rw_color[dispatch_id] = 0;
        return;
    }

    uint2 buffer_size;
    gbuffer_depth.GetDimensions(buffer_size.x, buffer_size.y);

    float3 position = cs_depth_to_position(dispatch_id.xy, buffer_size, depth);
    float3 view_dir = normalize(position - frame_params.view_pos.xyz);

    // TODO search with all lod (may be up to 12)
    VertexDescriptor vert = VertexDescriptor::create(position, view_dir);
    HashGridCell cell;
    uint addr;
    if ( hash_grid_find(hash_grid_storage_buffer, vert.hash(), vert.checksum(), cell, addr) )
    {
        float3 color;
        if (bool(pc.color_code))
        {
            uint code = vert.checksum(); // none zero, good for color code
            color.r = (code & 0xFF) / 255.0;
            color.g = ((code >> 8) & 0xFF) / 255.0;
            color.b = (((code >> 16) ^ (code >> 24)) & 0xFF) / 255.0;
        }
        else
        {
            color = cell.radiance;
        }
        rw_color[dispatch_id] = color;
    }
    else
    {
        float3 null_color;
        if (bool(pc.color_code))
        {
            null_color = 0.0;
        }
        else
        {
            null_color = float3(1, 0, 1);
        }
        rw_color[dispatch_id] = null_color;
    }
}