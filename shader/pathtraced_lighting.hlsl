#include "gbuffer.hlsl"
#include "scene_bindings.hlsl"

RaytracingAccelerationStructure scene_tlas;
Texture2D<float> gbuffer_depth;
Texture2D<uint4> gbuffer_color;
TextureCube<float4> skycube;
RWTexture2D<float4> rw_lighting;

struct Payload {
    float3 radiance;
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

    float3 lighting = 0.0f;
    {
        float3 dir = view_params.sun_dir;

        Payload payload;
        payload.radiance = 0.0f;
        RayDesc ray;
        ray.Origin = position_ws;
        ray.Direction = dir;
        ray.TMin = 0.0005f; // 0.5mm
        ray.TMax = 100.0f;
        TraceRay(scene_tlas,
                RAY_FLAG_FORCE_OPAQUE, // skip anyhit
                0xff, // uint InstanceInclusionMask,
                0, // uint RayContributionToHitGroupIndex,
                0, // uint MultiplierForGeometryContributionToHitGroupIndex,
                0, // uint MissShaderIndex,
                ray, // RayDesc Ray,
                payload // inout payload_t Payload
            );

        float atten = saturate(dot(ray.Direction, gbuffer.normal));
        lighting += gbuffer.color.xyz * payload.radiance * atten;
    }

    rw_lighting[dispatch_id.xy] = float4(lighting, 1.0f);
}

struct Attribute
{
    float2 bary;
};

float2 load_uv(MeshParams mesh, uint vert_id) {
	return float2(
		asfloat(vertex_buffer[mesh.texcoords_offset + vert_id * 2 + 0]),
		asfloat(vertex_buffer[mesh.texcoords_offset + vert_id * 2 + 1]));
}

[shader("closesthit")]
void closesthit(inout Payload payload, in Attribute attr) {
    uint mesh_index = GeometryIndex(); // index of geometry in BLAS; we are using only one blas for all loaded mesh
    uint triangle_index = PrimitiveIndex();

    MeshParams mesh = mesh_params[mesh_index];
    uint index0 = index_buffer[triangle_index * 3];
    uint index1 = index_buffer[triangle_index * 3 + 1];
    uint index2 = index_buffer[triangle_index * 3 + 2];
    float2 uv0 = load_uv(mesh, index0);
    float2 uv1 = load_uv(mesh, index0);
    float2 uv2 = load_uv(mesh, index0);
    float2 uv = uv0 * attr.bary.x + uv1 * attr.bary.y + uv2 * (1.0f - attr.bary.x - attr.bary.y);

    MaterialParams mat = material_params[mesh.material_index];
	float4 base_color = bindless_textures[mat.base_color_index].SampleLevel(sampler_linear_clamp, uv, 0);

    payload.radiance = base_color.rgb * 3.f;
    payload.radiance+= 0.5f;
}

[shader("miss")]
void miss(inout Payload payload)
{
    float3 dir_ws = WorldRayDirection();
    float3 sky = skycube.SampleLevel(sampler_linear_clamp, dir_ws, 0).xyz;
    payload.radiance = sky;
}