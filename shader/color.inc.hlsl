/*
 * Color space conversions.
 */

 /// Encode color by luma and chroma, with very simple calculation.
 /// NOTE: Output can be nagative.
 /// ref: https://en.wikipedia.org/wiki/YCoCg
 float3 rgb_to_yCoCg(float3 col)
 {
    float y  = col.r * 0.25  + col.g * 0.5 + col.b * 0.25;
    float co = col.r * 0.5                 - col.b * 0.5;
    float cg = col.r * -0.25 + col.g * 0.5 - col.b * 0.25;
    return float3(y, co, cg);
 }

 float3 yCoCg_to_rgb(float3 col)
 {
    float y = col.r;
    float co = col.g;
    float cg = col.b;
    float tmp = y - cg;
    float r = tmp + co; // Y - Cg + Co
    float g = y + cg;
    float b = tmp - co; // Y - Cg - Co
    return float3(r, g, b);
 }