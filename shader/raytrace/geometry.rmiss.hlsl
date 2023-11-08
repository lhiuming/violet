#include "geometry_ray.inc.hlsl"

[shader("miss")]
void main(inout GeometryRayPayload payload)
{
    #if SHRINK_PAYLOAD
    payload.hit_t = -1.0;
    #else
    payload.missed = true;
    #endif
}