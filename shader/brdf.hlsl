#pragma once

#define PI 3.14159265359
#define TWO_PI (2.0f * PI)
#define ONE_OVER_PI (1.0f / PI)

// Fresnel with f90
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float3 F_Schlick_with_f90(float u, float3 f0, float f90) {
    return f0 + (f90.xxx - f0) * pow(1.0 - u, 5.0);
}

// Fresnel 
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float3 F_Schlick(float u, float3 f0) {
	// Replace f90 with 1.0 in F_Schlick, and some refactoring
    float f = pow(1.0 - u, 5.0);
    return f + f0 * (1.0 - f);
}

// Fresnel with single channel
// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float F_Schlick_single(float u, float f0, float f90) {
	return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float D_GGX(float NoH, float roughness) {
    float a = NoH * roughness;
    float k = roughness / (1.0 - NoH * NoH + a * a);
    return k * k * (1.0 / PI);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
    float a2 = roughness * roughness;
    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
float Fd_Lambert() {
    return 1.0 / PI;
}

// Disney BRDF
// https://google.github.io/filament/Filament.md.html#materialsystem/brdf
float Fd_Burley(float NoV, float NoL, float LoH, float roughness) {
    float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    float lightScatter = F_Schlick_single(NoL, 1.0, f90);
    float viewScatter = F_Schlick_single(NoV, 1.0, f90);
    return lightScatter * viewScatter * (1.0 / PI);
}