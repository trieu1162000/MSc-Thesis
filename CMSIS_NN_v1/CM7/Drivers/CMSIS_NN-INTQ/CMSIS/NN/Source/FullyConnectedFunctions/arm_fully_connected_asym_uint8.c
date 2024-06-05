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
 * Title:        arm_fully_connected_uint8_asym.c
 * Description:  Fully connected uint8 asymmetric convolution
 *
 * $Date:        26. February 2019
 * $Authors:     Alessandro Capotondi - alessandro.capotondi@unibo.it
 *               Manuele Rusci - manuele.rusci@unibo.it
 *
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */
#include "stdint.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup FC
 * @{
 */

  /**
   * @brief uint8 asymmetric opt fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       z_wt        weights offset
   * @param[in]       z_in        input offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * vec_buffer size: dim_vec
   *
   * This basic function is designed to work with regular weight
   * matrix without interleaving.
   *
   */

arm_status
arm_fully_connected_asym_uint8(const uint8_t * pV,
                           	   	   const uint8_t * pM,
								   const uint16_t dim_vec,
								   const uint16_t num_of_rows,
								   const uint8_t z_wt,
								   const uint8_t z_in,
								   const uint8_t z_out,
								   const int32_t m_zero,
								   const uint16_t n_zero,
								   const int32_t * bias,
								   uint8_t * pOut,
								   int16_t * vec_buffer)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    const uint8_t *pB = pM;
    const uint8_t *pB2;
    uint8_t       *pO = pOut;
    const int32_t *pBias = bias;
    int16_t    *pA;
    uint16_t  rowCnt = num_of_rows >> 1;
    int16_t VzA[2] = {z_wt,z_wt};
	const int16_t *pzA = VzA;
	int32_t inzA = *__SIMD32(pzA);

    /* expand the vector into the buffer */
    arm_asym_uint8_to_int16_reordered_no_shift(pV, z_in, vec_buffer, dim_vec);

    while (rowCnt)
    {
    	int32_t   sum =  *pBias++;
    	int32_t   sum2 = *pBias++;
        uint16_t  colCnt = dim_vec >> 2;

        pA = vec_buffer;
        pB2 = pB + dim_vec;

        while (colCnt)
        {
        	int32_t     inV, inM11, inM12, inM21, inM22;
            pB = (uint8_t *) read_and_pad_reordered_uint8((void *)pB, &inM11, &inM12);
            pB2 = (uint8_t *) read_and_pad_reordered_uint8((void *)pB2, &inM21, &inM22);

            inV = *__SIMD32(pA)++;
            inM11 = __SSUB16(inM11, inzA);
            inM21 = __SSUB16(inM21, inzA);
            sum = __SMLAD(inV, inM11, sum);
            sum2 = __SMLAD(inV, inM21, sum2);

            inV = *__SIMD32(pA)++;
            inM12 = __SSUB16(inM12, inzA);
            inM22 = __SSUB16(inM22, inzA);
            sum = __SMLAD(inV, inM12, sum);
            sum2 = __SMLAD(inV, inM22, sum2);

            colCnt--;
        }
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
        	int16_t   inV  = (int16_t) *pA++;
        	int16_t   inM  = (int16_t) *pB++;
        	int16_t   inM2 = (int16_t) *pB2++;

            inM = inM - VzA[0];
            inM2 = inM2 - VzA[0];
            sum += inV * inM;
            sum2 += inV * inM2;
            colCnt--;
        }                       /* while over colCnt */

		sum  = ((__HI_SMULL(sum,m_zero)) >> n_zero) + z_out;
		sum2 = ((__HI_SMULL(sum2,m_zero)) >> n_zero) + z_out;
		*pO++ = (uint8_t) __USAT(sum, 8);
		*pO++ = (uint8_t) __USAT(sum2, 8);

        /* adjust the pointers and counters */
        pB += dim_vec;
        rowCnt--;
    }

    /* left-over part of the rows */
    rowCnt = num_of_rows & 0x1;

    while (rowCnt)
    {
        uint16_t  colCnt = dim_vec >> 2;
        int32_t   sum =  *pBias++;

        pA = vec_buffer;

        while (colCnt)
        {
        	int32_t     inV1, inV2, inM11, inM12;

            pB = (uint8_t *) read_and_pad_reordered_uint8((void *)pB, &inM11, &inM12);
            inM11 = __SSUB16(inM11, inzA);
            inM12 = __SSUB16(inM12, inzA);

            inV1 = *__SIMD32(pA)++;
            sum = __SMLAD(inV1, inM11, sum);

            inV2 = *__SIMD32(pA)++;
            sum = __SMLAD(inV2, inM12, sum);

            colCnt--;
        }

        /* left-over of the vector */
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
        	int16_t   inV  = (int16_t) *pA++;
        	int16_t   inM  = (int16_t) *pB++;

            inM = inM - VzA[0];
            sum += inV * inM;
            colCnt--;
        }

		sum  = ((__HI_SMULL(sum,m_zero)) >> n_zero) + z_out;
		*pO++ = (uint8_t) __USAT(sum, 8);

        rowCnt--;
    }

#else
	#error "Cortex-M0 and Cortex-M3 not supported"
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
#endif                          /* ARM_MATH_DSP */

    /* Return to ARM_MATH_SUCCESS */
    return (ARM_MATH_SUCCESS);

}

/**
 * @} end of FC group
 */
