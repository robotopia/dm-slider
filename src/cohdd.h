#ifndef __COHDD_H__
#define __COHDD_H__

#include <math_constants.h>

#define DMCONST          4148.808  /* MHz² pc⁻¹ cm³ s */
#define LIGHTSPEED  299792458.0    /* m/s */

#define TAPER_NONE   0
#define TAPER_HANN   1
#define TAPER_WELCH  2

__host__ __device__ __inline__
float calcDM( float dt_sec, float flo_MHz, float fhi_MHz )
{
    return dt_sec/(DMCONST*(1.0/(flo_MHz*flo_MHz) - 1.0/(fhi_MHz*fhi_MHz)));
}

__host__ __device__ __inline__
float calcTimeDelay( float DM, float flo_MHz, float fhi_MHz )
{
    return DMCONST*DM*(1.0/(flo_MHz*flo_MHz) - 1.0/(fhi_MHz*fhi_MHz));
}

__host__ __device__ __inline__
float taperHann( float f, float bw )
{
    float s = sin( CUDART_PI_F*f/bw );
    return s*s;
}

__host__ __device__ __inline__
float taperWelch( float f, float bw )
{
    float s = 2.0*f/bw - 1.0;
    return 1.0 - s*s;
}

__host__ __device__ __inline__
float taperFunc( float f, float bw, int taperType )
{
    float res;
    switch (taperType)
    {
        case TAPER_HANN:
            res = taperHann( f, bw );
            break;
        case TAPER_WELCH:
            res = taperWelch( f, bw );
            break;
        default:
            res = 1.0;
            break;
    }
    return res;
}

#endif
