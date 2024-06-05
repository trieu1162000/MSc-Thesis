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
 * Title:        arm_nn_mat_mult_kernel_int4_int16_reordered.c
 * Description:  Matrix-multiplication function for
 *               INT4 x INT16 convolution with reordered column
 *
 * $Date:        09. July 2018
 * $Authors:	 Manuele Rusci - manuele.rusci@unibo.it
 *				 Alessandro Capotondi - alessandro.capotondi@unibo.it
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_math.h"

#define INT4_SIZE(x)	((x)>>1)

  /**
   * @brief Threshold-based INT4 quantization
   * @param[in]       input     integer results of the convolution
   * @param[in]       pThr   	pointer to the threshold array, implicit size of 2^4 = 16  
   * @return     The function returns the quantized value
   *
   */

static int8_t int4_quant(int16_t input, const int16_t * pThr)
{
	int8_t ret = 0;
	if(input <= pThr[7] )
	{
		if( input <= pThr[3])
		{
			if( input <= pThr[1])
			{
				if( input <= pThr[0])
					ret = -8;
				else 
					ret = -7;
			}
			else
			{
				if( input <= pThr[2])
					ret = -6;
				else 
					ret = -5;
			}
		}
		else
		{
			if( input <= pThr[5])
			{
				if( input <= pThr[4])
					ret = -4;
				else
					ret = -3;
			}
			else
			{
				if( input <= pThr[6])
					ret = -2;
				else
					ret = -1;
			}
		}
	}
	else
	{
		if( input <= pThr[11])
		{
			if( input <= pThr[9])
			{
				if( input <= pThr[8])
					ret = 0;
				else
					ret = 1;
			}
			else
			{
				if( input <= pThr[10])
					ret = 2;
				else
					ret = 3;
			}
		}
		else
		{
			if( input <= pThr[13])
			{
				if( input <= pThr[12])
					ret = 4;
				else
					ret = 5;
			}
			else
			{
				if( input <= pThr[14])
					ret = 6;
				else
					ret = 7;
			}
		}
	}
	return ret;
}


  /**
   * @brief Matrix-multiplication function for INT4 x INT16
   * convolution with reordered column
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always consists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       pThreshold  pointer to the threshold array
   * @param[in,out]   pOut        pointer to output
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

int8_t     *arm_nn_mat_mult_kernel_int4_int16_reordered(const int8_t * pA,			// weight buffer
                                                  const int16_t * pInBuffer,		// input reordered buffer
                                                  const uint16_t ch_im_out,			// output channel dim
                                                  const uint16_t numCol_A,			// receptive field dim
												  const int16_t * pThreshold,		// pointer to the threshold array
												  int8_t * pOut)					// output buffer
{

#if defined (ARM_MATH_DSP)
    /* set up the second output pointers */
	int8_t res1, res2;
    int8_t     *pOut2 = pOut + INT4_SIZE(ch_im_out);
    int       i;

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        /* setup pointers for B */
        const int16_t *pB = pInBuffer;
        const int16_t *pB2 = pB + numCol_A;

        /* align the second pointer for A */
        const int8_t *pA2 = pA + INT4_SIZE( numCol_A );	

        /* init the sum as zeros */
        int32_t     sum =  0;
        int32_t     sum2 = 0;
        int32_t     sum3 = 0;
        int32_t     sum4 = 0;

        uint16_t  colCnt = numCol_A >> 3;	//number of 8xINT4 vectors

        /* accumulate over the vector */
        while (colCnt)
        {
            int32_t     inA11, inA12, inA13, inA14;
			int32_t		inA21, inA22, inA23, inA24;

            pA = (int8_t *) read_and_pad_reordered_int4((void *)pA, &inA11, &inA12, &inA13, &inA14);
            pA2 = (int8_t *) read_and_pad_reordered_int4((void *)pA2, &inA21, &inA22, &inA23, &inA24);

            int32_t     inB1 = *__SIMD32(pB)++;
            int32_t     inB2 = *__SIMD32(pB2)++;

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

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            sum = __SMLAD(inA13, inB1, sum);
            sum2 = __SMLAD(inA13, inB2, sum2);
            sum3 = __SMLAD(inA23, inB1, sum3);
            sum4 = __SMLAD(inA23, inB2, sum4);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            sum = __SMLAD(inA14, inB1, sum);
            sum2 = __SMLAD(inA14, inB2, sum2);
            sum3 = __SMLAD(inA24, inB1, sum3);
            sum4 = __SMLAD(inA24, inB2, sum4);

            colCnt--;
        }                       

        //FIXME fix the handling of remaining cols
#if 0
        // check for remaining cols
        colCnt = numCol_A & 0x7;
        if(colCnt)
        {
        	int32_t inA1_32 = *((int32_t *)  pA);
        	int32_t inA2_32 = *((int32_t *)  pA2);

       		while (colCnt)
	        {
	        	//read and sign-extend INT4 LSB 
	            int16_t     inA1 = __SXTB16(__ROR(__SXTB16( ( inA1_32 <<  4) , 4 ) ) );
	            int16_t     inB1 = *pB++;
	            int16_t     inA2 = __SXTB16(__ROR(__SXTB16( ( inA2_32 <<  4) , 4 ) ) );
	            int16_t     inB2 = *pB2++;

	            //convolve
	            sum += inA1 * inB1;
	            sum2 += inA1 * inB2;
	            sum3 += inA2 * inB1;
	            sum4 += inA2 * inB2;

	            //shift next value to LSB
	            inA1_32 = __ROR(inA1_32, 4);
        		inA2_32 = __ROR(inA2_32, 4);

        		//decrease counter
	            colCnt--;
	        }
        }
#endif



        // quantization of convolution accumulators and results compression
        res1 = int4_quant((int16_t) sum , &pThreshold[i<<4]		);	
		res2 = int4_quant((int16_t) sum3, &pThreshold[(i+1)<<4]	);
        *pOut++  = ( res1 & 0x0F ) | (( res2 << 4 ) & 0xF0 );
        res1 = int4_quant((int16_t) sum2 , &pThreshold[i<<4]);
		res2 = int4_quant((int16_t) sum4 , &pThreshold[(i+1)<<4]);
        *pOut2++ = ( res1 & 0x0F ) | (( res2 << 4 ) & 0xF0 );


        /* skip the row computed with A2 */
        pA += INT4_SIZE(numCol_A);
    }                          

    // increment of output pointer given the INT4 precision
    pOut += INT4_SIZE(ch_im_out);

    /* return the new output pointer with offset */
    return pOut;
#else
    /* To be completed */
    #error "INT4 Convolution Operation not supported"

    return NULL;
#endif                          /* ARM_MATH_DSP */
}
