#include "../gbuffer.hlsl"
#include "../util.hlsl"

Texture2D<float> depth_texture;
GBUFFER_TEXTURE_TYPE gbuffer_texture;

Texture2D<float3> diff_texture;
Texture2D<float3> spec_texture;
Texture2D<float2> variance_texture;

RWTexture2D<float3> rw_diff_texture;
RWTexture2D<float3> rw_spec_texture;
RWTexture2D<float2> rw_variance_texture;

[[vk::push_constant]]
struct PC 
{
    int step_size;
} pc;

struct EdgeStoppingFunc
{
    float depth_center;
    float fwidth_z;
    float3 normal_center;
    float2 lumi_center;
    float2 lumi_std;

    // luminance: x diffuse, y: specular
    static EdgeStoppingFunc create(float depth, float fwidth_z, float3 normal, float2 lumi, float2 lumi_var)
    {
        EdgeStoppingFunc ret;
        ret.depth_center = depth;
        ret.fwidth_z = fwidth_z;
        ret.normal_center = normal;
        ret.lumi_center = lumi;
        ret.lumi_std = sqrt(lumi_var); // TODO merge with rcp?
        return ret;
    }

    // weight for diffuse component (x) and specular components (y)
    float2 weight(float depth, float3 normal, float2 lumi)
    {
        // Tweaking
        const float sigma_z = 1.0; //default: 1
        const float sigma_n = 128.0; // default: 128
        const float sigma_l = 4.0; // default: 4
        const float epsilon = 1e-8;

        float weight_mono = 1.0;

        // depth
        float dist_z = abs(depth - depth_center);
        weight_mono *= exp( - dist_z * rcp(sigma_z * fwidth_z + epsilon) );

        // normal 
        float n_dot = saturate(dot(normal_center, normal));
        weight_mono *= pow(n_dot, sigma_n);

        // luminance 
        float2 dist_l = abs(lumi.xy - lumi_center.xy);
        float2 weight = weight_mono * exp( - dist_l * rcp(sigma_l * lumi_std.xy + epsilon));

        return weight;
    }
};

[numthreads(8, 8, 1)]
void main(uint2 dispatch_id: SV_DispatchThreadID)
{
    uint2 buffer_size;
    depth_texture.GetDimensions(buffer_size.x, buffer_size.y);

    float depth_center = depth_texture[dispatch_id];
    GBuffer gbuffer = load_gbuffer(gbuffer_texture, dispatch_id);
    float3 diff_center = diff_texture[dispatch_id];
    float3 spec_center = spec_texture[dispatch_id];
    float2 lumi_center = float2(luminance(diff_center), luminance(spec_center));
    // TODO Gaussian filter on variance
    float2 variance_center = variance_texture[dispatch_id];
    const EdgeStoppingFunc edge_stopping = EdgeStoppingFunc::create(depth_center, gbuffer.fwidth_z, gbuffer.normal, lumi_center, variance_center);

    // A-Trous: 5x5 taps, with increaing step size in each pass
    const int2 pix_coord_center = int2(dispatch_id);
    const float kernel_weights[3] = { 1.0, 2.0/3.0, 1.0/6.0 }; // normalized such that center weight (at [0]*[0]) is 1.0. originally { 3.0/8.0, 1.0/4.0, 1.0/16.0 };
    float3 diff_sum = diff_center;
    float3 spec_sum = spec_center;
    float2 variance_sum = variance_center;
    float2 weight_sum = 1.0; // kernel_weights[0] * kernel_weights[0]
    for (int y = -2; y <= 2 ; y++)
    {
        float kernel_weight_y = kernel_weights[abs(y)];
        for (int x = -2; x <= 2 ; x++)
        {
            int2 pix_coord_unclamped = pix_coord_center + int2(x, y) * pc.step_size;
            int2 pix_coord = clamp(pix_coord_unclamped, int2(0, 0), int2(buffer_size) - 1);
            bool coord_valid = all(pix_coord_unclamped == pix_coord);
            bool not_center = (x != 0) && (y != 0);
            if (coord_valid && not_center)
            {
                float2 weight = kernel_weight_y * kernel_weights[abs(x)];

                float3 diff = diff_texture[pix_coord];
                float3 spec = spec_texture[pix_coord];

                // Apply edge stoping function
                float depth = depth_texture[pix_coord];
                GBuffer gbuffer = load_gbuffer(gbuffer_texture, pix_coord);
                float3 normal = gbuffer.normal;
                float2 lumi = float2(luminance(diff), luminance(spec));
                weight *= edge_stopping.weight(depth, normal, lumi);
                float2 weight_var = weight * weight;

                diff_sum += weight.x * diff;
                spec_sum += weight.y * spec;
                variance_sum += weight_var * variance_texture[pix_coord];

                weight_sum += weight;
            }
        }
    }

    float3 diff = diff_sum / weight_sum.x;
    float3 spec = spec_sum / weight_sum.y;
    float2 variance = variance_sum / (weight_sum * weight_sum);

    rw_diff_texture[dispatch_id] = diff;
    rw_spec_texture[dispatch_id] = spec;
    rw_variance_texture[dispatch_id] = variance;
}