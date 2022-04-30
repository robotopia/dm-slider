#ifndef __COHDD_H__
#define __COHDD_H__

#define DMCONST          4148.808  /* MHz² pc⁻¹ cm³ s */
#define LIGHTSPEED  299792458.0    /* m/s */

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

#endif
