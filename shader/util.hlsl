#pragma once

// Common luminance calculation for Rec.709 tristimulus values
float luminance(float3 col) {
    return dot(col, float3(0.2126f, 0.7152f, 0.0722f));
}

// Check geometry by depth
bool has_no_geometry_via_depth(float depth_buffer_value) {
    return depth_buffer_value == 0.0f;
}
