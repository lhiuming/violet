#include "shadow_ray.inc.hlsl"

[shader("miss")]
void main(inout ShadowRayPayload payload)
{
    payload.missed = true;
}