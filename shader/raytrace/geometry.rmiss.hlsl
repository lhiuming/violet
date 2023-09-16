#include "geometry_ray.inc.hlsl"

[shader("miss")]
void main(inout GeometryRayPayload payload)
{
    payload.missed = true;
}