#pragma once
#include "constants.hlsl"

// Fresnel, assuming f90 = 1.0
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float3 F_Schlick(float u, float3 f0) {
	// NOTE: Replacing f90 with 1.0 in F_Schlick_f90, and some refactoring
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

// Fresnel for single channel, assuming f90 = 1.0
float F_Schlick_single(float u, float f0) {
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

// Fresnel with f90
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float3 F_Schlick_f90(float u, float3 f0, float f90) {
    return f0 + (f90.xxx - f0) * pow(1.0 - u, 5.0);
}

// Fresnel with f90 for single channel
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float F_Schlick_f90_single(float u, float f0, float f90) {
	return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float D_GGX(float NoH, float roughness) {
    float a = NoH * roughness;
    float k = roughness / (1.0 - NoH * NoH + a * a);
    return k * k * (1.0 / PI);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float Fd_Lambert() {
    return 1.0 / PI;
}

// Disney BRDF
// https://google.github.io/filament/Filament.md.html#materialsystem/brdf
float Fd_Burley(float NoV, float NoL, float LoH, float roughness) {
    float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    float lightScatter = F_Schlick_f90_single(NoL, 1.0, f90);
    float viewScatter = F_Schlick_f90_single(NoV, 1.0, f90);
    return lightScatter * viewScatter * (1.0 / PI);
}


// Lambda term in Smith masking function for isotropic GGX NDF.
// [Heitz 2014 "Understanding the Masking-Shadowing Function", p84]
// [Boksansky 2021 "Crash Course in BRDF Implementation", p13]
float smith_lambda_GGX(float cos_theta, float roughness)
{
#if 1
    // Reference
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    float tan_theta = sin_theta / cos_theta;
    float a = rcp(roughness * tan_theta); // Heitz
          a = cos_theta * rcp(roughness * sin_theta); // Jakub
          a = cos_theta * rcp( roughness * sqrt(1.0f - cos_theta*cos_theta) ); // Jakub
    return ( -1.0f + sqrt( 1.0f + rcp(a*a) ) ) / 2.0f;
#else
    // Simplified
    float r2 = roughness * roughness;
    float c = cos_theta;
    return ( -1.0f + sqrt( (-r2 * c + c) * c + r2 ) / c ) / 2.0f;
#endif
}

// Smith masking-shadowing function for isotropic GGX NDF.
// Note that binary masking by HoV>0 and HoL>0 is ignoreed (left to calling site).
// [Heitz 2014 "Understanding the Masking-Shadowing Function", p91]
// [Boksansky 2021 "Crash Course in BRDF Implementation", p14]
//   Note that this blog seems to mistabke N for H when discussing Smith model.
float smith_G2_height_correlated_GGX(float NoL, float NoV, float roughness)
{
#if 1
    // Reference
    float lambda_L = smith_lambda_GGX(NoL, roughness);
    float lambda_V = smith_lambda_GGX(NoV, roughness);
    return 1.0f / (1.0f + lambda_L + lambda_V);
#else
    // Optimized
    float r2 = roughness * roughness;
    float k_L = sqrt((-r2 * NoL + NoL) * NoL + r2) / NoL; // 2 fma, 1 sqrt, 1 div
    float k_V = sqrt((-r2 * NoV + NoV) * NoV + r2) / NoV;
    return 2.0f / ( k_L + k_V );
#endif
}

// Visibility with Smith masking-shadowing function for isotropic GGX NDF.
// Vis := G / (4*NoL*NoV).
// Divide G by 4*NoL*NoV from the full microfacet BRDF to cancel out some terms.
// Note that binary masking by HoV>0 and HoL>0 is ignoreed (left to calling site).
// [Lagarde 2014, "Moving Frostbite to Physically Based Rendering", p12]
float vis_smith_G2_height_correlated_GGX(float NoL, float NoV, float roughness)
{
#if 0
    // Reference
    float G = smith_G2_height_correlated(HoL, HoV, roughness);
    float Vis = G / (4.0f * NoL * NoV);
    return Vis;
#else
    // Optimized
    float r2 = roughness * roughness;
    float u_L = NoV * sqrt((-r2 * NoL + NoL) * NoL + r2);
    float u_V = NoL * sqrt((-r2 * NoV + NoV) * NoV + r2);
    return 0.5f / (u_L + u_V);
#endif
}

// Smith G2/G1 for isotropic GGX, only used when generate sampling from GGX VNDF.
// Note that G1 is for Smith G1 masking function (G1(V)).
// Note that binary masking by HoL>0 is ignoreed (left to calling site).
// [Heitz 2014 "Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals", section 2]
float smith_G2_over_G1_height_correlated_GGX(float NoL, float NoV, float roughness)
{
#if 0
    // Reference 
    float G2 = smith_G2_height_correlated_GGX(NoL, NoV, roughness);
    float G1_V = rcp(1 + smith_lambda_GGX(NoV, roughness));
    return G1 / G1_V;
#elif 0
    // Reference, alternative
    // [Heitz 2015 "Implementing a Simple Anisotropic Rough Diffuse Material with Stochastic Evaluation", appendix A]
    float l_V = smith_lambda_GGX(NoV, roughness);
    float l_L = smith_lambda_GGX(NoL, roughness);
    return (1.0f + l_V) / (1.0f + l_V + l_L);
#else
    // Optimized
    float r2 = roughness * roughness;
    float k_V = sqrt( (-r2 * NoV + NoV) * NoV + r2 ) / NoV;
    float k_L = sqrt( (-r2 * NoL + NoL) * NoL + r2 ) / NoL;
    return (1.0f + k_V) / (k_V + k_L);
#endif
}

// Diffuse color for Metallic workflow 
float3 get_diffuse_rho(float3 base_color, float metallic) {
    return base_color.rgb * (1.0f - metallic);
}

// Specular color for Metallic workflow
float3 get_specular_f0(float3 base_color, float metallic) {
    return lerp((0.04f).xxx, base_color.rgb, metallic);
}

// Combined BRDF evaluation
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
// NOTE: ignoring the (1.0 - Fresnel) term compared to the glTF spec 
float3 eval_GGX_Lambertian(float3 v, float3 l, float3 n, float perceptual_roughness, float3 diffuse_rho, float3 specular_f0) {
	float3 h = normalize(v + l);

    float NoL = saturate(dot(n, l));
    float NoV = saturate(dot(n, v));
    float NoH = saturate(dot(n, h));
    float LoH = saturate(dot(l, h));

    // perceptually linear roughness to roughness
    float roughness = perceptual_roughness * perceptual_roughness;

    float D = D_GGX(NoH, roughness);
    float3 F = F_Schlick(LoH, specular_f0);
    float V = vis_smith_G2_height_correlated_GGX(NoL, NoV, roughness);

    // specular BRDF
    // NOTE: vis_smith_G2_height_correlated_GGX produce INF if both NoV and NoL is zero; to minimized bias and make reasoning easier, we clamp final scalar only.
    float3 Fr = min(D * V, F32_SIGNIFICANTLY_LARGE) * F;

    // diffuse BRDF
    float3 Fd = diffuse_rho * Fd_Lambert();

	return Fd + Fr;
}