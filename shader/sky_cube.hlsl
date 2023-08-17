#include "atmosphere_felix_westin.hlsl"
#include "frame_bindings.hlsl"

struct PushConstants
{
    float cube_width; 
};
[[vk::push_constant]]
PushConstants pc;

RWTexture2DArray<float3> rw_cube_texture;

[numthreads(8, 4, 1)]
void main(uint2 dispatch_thread_id: SV_DISPATCHTHREADID, uint3 group_id: SV_GROUPID) {
    uint face = group_id.z;

    // Convert uv-and-face to cubemap direction
    float2 uv = (dispatch_thread_id + 0.5f) / pc.cube_width.xx;
    float2 st = uv * 2.0f - 1.0f;
    float3 dir;
    // ref: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#_cube_map_face_selection_and_transformations
    switch (face) {
        case 0: dir = float3( 1, -st.y, -st.x); break;
        case 1: dir = float3(-1, -st.y,  st.x); break;
        case 2: dir = float3( st.x,  1,  st.y); break;
        case 3: dir = float3( st.x, -1, -st.y); break;
        case 4: dir = float3( st.x, -st.y,  1); break;
        case 5: dir = float3(-st.x, -st.y, -1); break;
    }
    dir = normalize(dir);

    // Compute sky color for direction
	float3 ray_start = float3(0, 0, 0); // TODO Update this?
	float3 ray_dir = dir;
	float ray_len = 100000.0f;

	float3 light_dir = frame_params.sun_dir.xyz;
	float3 light_color = frame_params.sun_inten.rgb;

    // TODO store transmittance in cube for latter lighting calculation?
	float3 _transmittance;
	float3 color = IntegrateScattering(ray_start, ray_dir, ray_len, light_dir, light_color, _transmittance);

    rw_cube_texture[uint3(dispatch_thread_id, face)] = color;
}
