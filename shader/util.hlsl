#pragma once

// component max
float component_max(float3 v)
{
    return max(max(v.x, v.y), v.z);
}

// max of three entries
float4 max3(float4 a, float4 b, float4 c)
{
    return max(max(a, b), c);
}

// min of three entries
float4 min3(float4 a, float4 b, float4 c)
{
    return min(min(a, b), c);
}

// Common luminance calculation for Rec.709 tristimulus values
float luminance(float3 col)
{
    return dot(col, float3(0.2126f, 0.7152f, 0.0722f));
}

// Check geometry by depth
bool has_no_geometry_via_depth(float depth_buffer_value)
{
    return depth_buffer_value == 0.0f;
}
