#ifndef RESTIR_CONFIG_INCLUDED
#define RESTIR_CONFIG_INCLUDED

// Indirect Specular supression 
#define IND_SPEC_R_MIN 0.002

#define DEMODULATE_INDIRECT_SPECULAR_FOR_DENOISER 1
// NOTE: feels too high, but lower case banding on metal surface; use pre-integrated LUT instead
#define DEMODULATE_INDIRECT_SPECULAR_RG_INTERGRAL_MIN 1e-2

#endif