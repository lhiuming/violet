#ifndef DENOISE_SVGF_INCLUDED
#define DENOISE_SVGF_INCLUDED

#define HISTORY_LEN_UNIT (1.0 / 255.0) // R8_UNORM

struct EdgeStoppingFunc
{
    float depth_center;
    float fwidth_z;
    float3 normal_center;
    float2 lumi_center;
    float2 lumi_std;

    // Config
    bool use_lumi;
    float sigma_z;
    float sigma_n;

    // luminance: x diffuse, y: specular
    static EdgeStoppingFunc create(float depth, float fwidth_z, float3 normal)
    {
        EdgeStoppingFunc ret;
        ret.depth_center = depth;
        ret.fwidth_z = fwidth_z;
        ret.normal_center = normal;
        ret.use_lumi = false;
        ret.sigma_z = 1.0;
        ret.sigma_n = 128.0;
        return ret;
    }

    // Default: 1.0, higher -> relaxed
    EdgeStoppingFunc sigma_depth(float s)
    {
        EdgeStoppingFunc ret = this;
        ret.sigma_z = s;
        return ret;
    }

    // Default: 128.0, lower -> relaxed
    EdgeStoppingFunc sigma_normal(float s)
    {
        EdgeStoppingFunc ret = this;
        ret.sigma_n = s;
        return ret;
    }

    EdgeStoppingFunc enable_lumi(float2 lumi, float2 lumi_var)
    {
        EdgeStoppingFunc ret = this;
        ret.use_lumi = true;
        ret.lumi_center = lumi;
        ret.lumi_std = sqrt(lumi_var);
        return ret;
    }

    // weight for diffuse component (x) and specular components (y)
    float2 weight(float tap_dist, float depth, float3 normal, float2 lumi)
    {
        // Tweaking
        const float sigma_l = 4.0; // default: 4 or 10
        const float epsilon = 1e-8;

        // depth
        float dist_z = abs(depth - depth_center);
        float weight_z_log = - dist_z  / ((sigma_z * fwidth_z) * tap_dist + epsilon);

        // normal 
        float n_dot = saturate(dot(normal_center, normal));
        float weight_n = pow(n_dot, sigma_n);

        // luminance 
        float2 weight_l_log = 0.0;
        if (use_lumi)
        {
            float2 dist_l = abs(lumi.xy - lumi_center.xy);
            weight_l_log = - dist_l * rcp(sigma_l * lumi_std.xy + epsilon);
        }

        // Compose
        float2 weight = weight_n * exp( weight_l_log + weight_z_log ); 
        return weight;
    }
};

#endif