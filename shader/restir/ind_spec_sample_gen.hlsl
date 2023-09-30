#include "../brdf.hlsl"
#include "../enc.inc.hlsl"
#include "../frame_bindings.hlsl"
#include "../gbuffer.hlsl"
#include "../rand.hlsl"
#include "../sampling.hlsl"
#include "../util.hlsl"

#include "raytrace.inc.hlsl"
#include "reservoir.hlsl"

Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;

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

    GBuffer gbuffer = decode_gbuffer(gbuffer_color[dispatch_id.xy]);

    // world position reconstruction from depth buffer
    float depth_error = 1.f / 16777216.f; // 24 bit unorm depth
    float2 screen_pos = (dispatch_id.xy + 0.5f) / float2(buffer_size);
    float4 position_ws_h = mul(view_params().inv_view_proj, float4(screen_pos * 2.0f - 1.0f, depth + depth_error, 1.0f));
    float3 position_ws = position_ws_h.xyz / position_ws_h.w;

    // Specualr may need some better noise
    uint rng_state = lcg_init(dispatch_id.xy, buffer_size, pc.frame_index);

    // Generate Sample Point with uniform hemisphere sampling
    // TODO blue noise
    float3 sample_dir;
    {
        float2 u = float2(lcg_rand(rng_state), lcg_rand(rng_state));
        sample_dir = sample_hemisphere_uniform_with_normal(u, gbuffer.normal);
    }

    // Raytrace
    RadianceTraceResult trace_result = trace_radiance(position_ws, sample_dir, pc.has_prev_frame);

    //rw_origin_pos_texture[dispatch_id.xy] = position_ws;
    rw_hit_pos_texture[dispatch_id.xy] = float4(trace_result.position_ws, 1.0f);
    rw_hit_normal_texture[dispatch_id.xy] = normal_encode_oct_u32(trace_result.normal_ws);
    rw_hit_radiance_texture[dispatch_id.xy] = trace_result.radiance;
}
