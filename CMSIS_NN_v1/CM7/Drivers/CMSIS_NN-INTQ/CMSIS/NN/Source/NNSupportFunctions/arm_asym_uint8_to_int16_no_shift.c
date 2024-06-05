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
 * Title:        arm_asym_uint8_to_int16_no_shift.c
 * Description:  Converts the elements of the Asymmetric UINT8 vector
 *               to INT16 vector without left-shift
 *
 * $Date:        26. February 2019
 * $Authors:     Alessandro Capotondi - alessandro.capotondi@unibo.it
 *               Manuele Rusci - manuele.rusci@unibo.it
 *
 * Target Processor:  Cortex-M cores
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
 * @brief Converts the elements of the Asymmetric UINT8 vector
 * to int16 vector without left-shift
 * @param[in] *pSrc       points to the uint8_t input vector
 * @param[in] offset      quantization offset (asymmetry)
 * @param[out] *pDst      points to the int16_t output vector
 * @param[in] blockSize   length of the input vector
 * @return none.
 */

void arm_asym_uint8_to_int16_no_shift(
		const uint8_t * pSrc,
		const uint8_t offset,
		int16_t * pDst,
		uint32_t blockSize)
{
    const uint8_t *pIn = pSrc;     /* Src pointer */
    uint32_t  blkCnt;           /* loop counter */
    int16_t offsets[2] = {offset, offset};
    const int16_t *offset_ptr = offsets;

#ifndef ARM_MATH_CM0_FAMILY
    int32_t   in;
    int32_t   in1, in2;
    int32_t   out1, out2;
    int32_t   offset_vect = *__SIMD32(offset_ptr);

    /*loop Unrolling */
    blkCnt = blockSize >> 2u;

    /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.    
     ** a second loop below computes the remaining 1 to 3 samples. */
    while (blkCnt > 0u)
    {
        /* C = (q15_t) A << 8 */
        /* convert from q7 to q15 and then store the results in the destination buffer */
        in = *__SIMD32(pIn)++;

        /* rotatate in by 8 and extend two q7_t values to q15_t values */
        in1 = __UXTB16(__ROR(in, 8));

        /* extend remainig two q7_t values to q15_t values */
        in2 = __UXTB16(in);

        if(offset){
        	in1 = __SSUB16(in1, offset_vect);
        	in2 = __SSUB16(in2, offset_vect);
        }

#ifndef ARM_MATH_BIG_ENDIAN

        out2 = __PKHTB(in1, in2, 16);
        out1 = __PKHBT(in2, in1, 16);

#else

        out1 = __PKHTB(in1, in2, 16);
        out2 = __PKHBT(in2, in1, 16);

#endif

        *__SIMD32(pDst)++ = out1;
        *__SIMD32(pDst)++ = out2;

        /* Decrement the loop counter */
        blkCnt--;
    }

    /* If the blockSize is not a multiple of 4, compute any remaining output samples here.    
     ** No loop unrolling is used. */
    blkCnt = blockSize % 0x4u;

#else
#error "Cortex-M0 is not supported"
#endif                          /* #ifndef ARM_MATH_CM0_FAMILY */

    while (blkCnt > 0u)
    {
    	*pDst++ = ((int16_t) * pIn++)-offset;

        /* Decrement the loop counter */
        blkCnt--;
    }

}

/**    
 * @}
 */
