/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 * Modifications Copyright (C) 2018 University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library - INT-Q extension
 * Title:        arm_int2_to_int16_reordered_no_shift.c
 * Description:  Converts the elements of the INT2 vector
 *               to reordered int16 vector without left-shift
 *
 * $Date:       09. July 2018
 * $Authors:    Manuele Rusci - manuele.rusci@unibo.it
 *              Alessandro Capotondi - alessandro.capotondi@unibo.it
 *              Francesco Conti - f.conti@unibo.it
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**    
 * @ingroup groupSupport    
 */

/**    
 * @addtogroup nndata_convert    
 * @{    
 */

/**    
 * @brief Converts the elements of the INT2 vector
 * to reordered INT16 vector without left-shift
 * @param[in]       *pSrc points to the INT2 input vector    
 * @param[out]      *pDst points to the INT16 output vector   
 * @param[in]       blockSize length of the input vector    
 * @return none.
 *
 */

void arm_int2_to_int16_reordered_no_shift(const int8_t * pSrc, int16_t * pDst, uint32_t blockSize)
{
    const int8_t *pIn = pSrc;     /* Src pointer */
    uint32_t  blkCnt;           /* loop counter */

#ifndef ARM_MATH_CM0_FAMILY
    int32_t     in;
    int32_t     in1, in2, in3, in4, in5, in6, in7, in8;

    /* Run the below code for Cortex-M4 and Cortex-M3 */

    /*loop Unrolling */
    blkCnt = blockSize >> 4u;	//number of iteration on 16-elements blocks

    /* First part of the processing with loop unrolling.  Compute 16 outputs at a time. */
    while (blkCnt > 0u)
    {
        /* convert from INT2 to INT16 and then store the results in the destination buffer */
        in = *__SIMD32(pIn)++;

        //4 input channel
        in8 = __SXTB16(__ROR(__SXTB16( in <<  6) , 6 ) );
        in7 = __SXTB16(__ROR(__SXTB16( in <<  4) , 6 ) ) ;
        in6 = __SXTB16(__ROR(__SXTB16( in <<  2) , 6 ) ) ;
        in5 = __SXTB16(__ROR(__SXTB16( in      ) , 6 ) );


        in4 = __SXTB16(__ROR(__SXTB16( in >>  2) , 6 ) );
        in3 = __SXTB16(__ROR(__SXTB16( in >>  4) , 6 ) ) ;
        in2 = __SXTB16(__ROR(__SXTB16( in >>  6) , 6 ) ) ;
        in1 = __SXTB16(__ROR(__SXTB16( in >>  8) , 6 ) );


#ifndef ARM_MATH_BIG_ENDIAN
        *__SIMD32(pDst)++ = in8;
        *__SIMD32(pDst)++ = in7;
        *__SIMD32(pDst)++ = in6;
        *__SIMD32(pDst)++ = in5;
        *__SIMD32(pDst)++ = in4;
        *__SIMD32(pDst)++ = in3;
        *__SIMD32(pDst)++ = in2;
        *__SIMD32(pDst)++ = in1;
#else 
        *__SIMD32(pDst)++ = in1;
        *__SIMD32(pDst)++ = in2;
        *__SIMD32(pDst)++ = in3;
        *__SIMD32(pDst)++ = in4;
        *__SIMD32(pDst)++ = in5;
        *__SIMD32(pDst)++ = in6;
        *__SIMD32(pDst)++ = in7;
        *__SIMD32(pDst)++ = in8;
#endif

        /* Decrement the loop counter */
        blkCnt--;
    }

    /* If the blockSize is not a multiple of 16, compute any remaining output samples here.    
     ** No loop unrolling is used. */
    blkCnt = blockSize % 0x16u;

#else

    /* Run the below code for Cortex-M0 */

    /* Loop over blockSize number of values */
    blkCnt = blockSize;

#endif                          /* #ifndef ARM_MATH_CM0_FAMILY */

    if(blkCnt)
    {
        int32_t in_32 = *((int32_t *)  pIn);
        
        while (blkCnt > 0u)
        {
            /* convert from INT4 to INT16 and then store the results in the destination buffer */
            *pDst++ = (int16_t) __SXTB16(__ROR(__SXTB16( in_32 <<  6) , 6 ) );
            in_32 = __ROR(in_32, 2);

            /* Decrement the loop counter */
            blkCnt--;
        }
    }

}

/**    
 * @}
 */
