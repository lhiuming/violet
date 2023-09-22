#include "../frame_bindings.hlsl"
#include "../util.hlsl"
#include "reservoir.hlsl"

StructuredBuffer<RestirSample> new_sample_buffer;
Texture2D<float> depth_buffer;
RWTexture2D<float4> rw_ao_history_texture; 
RWTexture2D<float> rw_ao_texture; 

struct PushConstants
{
    uint has_new_sample;
    uint has_history;
    float radius_ws;
};
[[vk::push_constant]]
PushConstants pc;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID) {
    uint2 buffer_size;
    depth_buffer.GetDimensions(buffer_size.x, buffer_size.y);
    
    float depth = depth_buffer[dispatch_id.xy];
    
    // early out
    if (has_no_geometry_via_depth(depth))
        {
        rw_ao_texture[dispatch_id] = 0.0f;
        if (!bool(pc.has_history))
            {
            rw_ao_history_texture[dispatch_id] = 0.0f;
        }
        return;
    }
    
    // Read history
    float4 history;
    if (bool(pc.has_history)) 
    {
        // TODO reprojection
        history = rw_ao_history_texture[dispatch_id.xy];
    }
    else 
    {
        history = 0.0f;
    }
    
    // Calculate AO
    float ao;
    if (bool(pc.has_new_sample))
    {
        RestirSample z;
        {
            uint index = dispatch_id.x + dispatch_id.y * buffer_size.x;
            z = new_sample_buffer[index];
        }
        float3 position_ws = cs_depth_to_position(dispatch_id, buffer_size, depth);
        float3 sample_offset = z.hit_pos - position_ws;
        float sample_distance = length(sample_offset);
        ao = select(sample_distance < pc.radius_ws, 0.0, 1.0);
        
        // Temporal filter
        if (bool(pc.has_history)) 
        {
            ao = lerp(history.x, ao, 1.0 / 16.0);
        }
        history.x = ao;
        rw_ao_history_texture[dispatch_id] = history;
    }
    else 
    {
        ao = history.x;
    }
    
    // Done
    rw_ao_texture[dispatch_id] = ao;
}