#include "geometry_ray.hlsli"

[shader("miss")]
void main(inout GeometryRayPayload payload)
{
    payload.missed = true;
}