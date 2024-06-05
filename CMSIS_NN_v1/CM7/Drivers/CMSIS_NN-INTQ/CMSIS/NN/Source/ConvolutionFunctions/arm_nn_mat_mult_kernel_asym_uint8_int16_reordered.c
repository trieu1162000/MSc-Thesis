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
 * Title:        arm_nn_mat_mult_kernel_asym_uint8_int16_reordered.c
 * Description:  Matrix-multiplication function for
 *               Asymmetric UINT8 x INT16 convolution with reordered columns
 *
 * $Date:        26. February 2019
 * $Authors:     Alessandro Capotondi - alessandro.capotondi@unibo.it
 *               Manuele Rusci - manuele.rusci@unibo.it
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_math.h"

  /**
   * @brief Matrix-multiplication function for
   *        Asymmetric UINT8 x INT16 convolution with reordered columns
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       z_wt        weights offset
   * @param[in]       z_in        input offset
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t *arm_nn_mat_mult_kernel_asym_uint8_int16_reordered(const uint8_t * pA,
                                                  const int16_t * pInBuffer,
												  const uint8_t z_a,
												  const uint8_t z_b,
												  const uint8_t z_out,
												  const int32_t m_zero,
												  const uint16_t n_zero,
                                                  const uint16_t ch_im_out,
                                                  const uint16_t numCol_A,
                                                  const int32_t * bias,
												  uint8_t * pOut)
{

#if defined (ARM_MATH_DSP)
    /* set up the second output pointers */
	uint8_t     *pOut2 = pOut + ch_im_out;
    int       i;
    int16_t VzA[2] = {z_a,z_a};
	const int16_t *pzA = VzA;
	int32_t inzA = *__SIMD32(pzA);

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        /* setup pointers for B */
        const int16_t *pB = pInBuffer;
        const int16_t *pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const uint8_t *pA2 = pA + numCol_A;

        int32_t     sum =  bias[i];
        int32_t     sum2 = bias[i];
        int32_t     sum3 = bias[i + 1];
        int32_t     sum4 = bias[i + 1];

        uint16_t  colCnt = numCol_A >> 2;

        /* accumulate over the vector */
        while (colCnt)
        {
        	int32_t inA11, inA12, inA21, inA22;
        	int32_t inB1 = *__SIMD32(pB)++;
        	int32_t inB2 = *__SIMD32(pB2)++;

            pA = (uint8_t *) read_and_pad_reordered_uint8((void *)pA, &inA11, &inA12);
            pA2 = (uint8_t *) read_and_pad_reordered_uint8((void *)pA2, &inA21, &inA22);

			inA11 = __SSUB16(inA11, inzA);
			inA12 = __SSUB16(inA12, inzA);
			inA21 = __SSUB16(inA21, inzA);
			inA22 = __SSUB16(inA22, inzA);

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);
            sum3 = __SMLAD(inA21, inB1, sum3);
            sum4 = __SMLAD(inA21, inB2, sum4);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);
            sum3 = __SMLAD(inA22, inB1, sum3);
            sum4 = __SMLAD(inA22, inB2, sum4);

            colCnt--;
        } /* while over colCnt */


        //*** TO BE TESTED ***
        colCnt = numCol_A & 0x3;
        while (colCnt)
        {
            int16_t   inA1 = (int16_t)*pA++;
            int16_t   inB1 = *pB++;
            int16_t   inA2 = (int16_t)*pA2++;
            int16_t   inB2 = *pB2++;

			inA1 = inA1 - VzA[0];
			inA2 = inA2 - VzA[0];

            sum += inA1 * inB1;
            sum2 += inA1 * inB2;
            sum3 += inA2 * inB1;
            sum4 += inA2 * inB2;
            colCnt--;
        } /* while over colCnt */

		sum  = ((__HI_SMULL(sum,m_zero)) >> n_zero) + z_out;
		sum2 = ((__HI_SMULL(sum2,m_zero)) >> n_zero) + z_out;
		sum3 = ((__HI_SMULL(sum3,m_zero)) >> n_zero) + z_out;
		sum4 = ((__HI_SMULL(sum4,m_zero)) >> n_zero) + z_out;

		*pOut++ = (uint8_t) __USAT(sum, 8);
		*pOut++ = (uint8_t) __USAT(sum3, 8);
		*pOut2++ = (uint8_t) __USAT(sum2, 8);
		*pOut2++ = (uint8_t) __USAT(sum4, 8);

        /* skip the row computed with A2 */
        pA += numCol_A;
    }                           /* for over ch_im_out */

    pOut += ch_im_out;

#else
	#error "Cortex-M0 and Cortex-M3 not supported"
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
#endif                          /* ARM_MATH_DSP */

    /* return the new output pointer with offset */
    return pOut;
}
