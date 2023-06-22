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

    float2 uv = (dispatch_thread_id + 0.5f) / pc.cube_width.xx;
    float3 dir;
    // TODO gen by AI; review it!
    switch (face) {
        case 0: dir = float3( 1,  uv.x,  uv.y); break;
        case 1: dir = float3(-1,  uv.x, -uv.y); break;
        case 2: dir = float3( uv.x,  1,  uv.y); break;
        case 3: dir = float3( uv.x, -1, -uv.y); break;
        case 4: dir = float3( uv.x,  uv.y,  1); break;
        case 5: dir = float3(-uv.x,  uv.y, -1); break;
    }

    float3 color = dir * 0.5f + 0.5f;
    rw_cube_texture[uint3(dispatch_thread_id, face)] = color;
}
