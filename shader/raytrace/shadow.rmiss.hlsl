#include "shadow_ray.hlsli"

[shader("miss")]
void main(inout ShadowRayPayload payload)
{
    payload.missed = true;
}