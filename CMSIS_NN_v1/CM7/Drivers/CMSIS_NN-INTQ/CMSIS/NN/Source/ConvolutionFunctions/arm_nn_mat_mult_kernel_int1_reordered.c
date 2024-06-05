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
 * Title:        arm_nn_mat_mult_kernel_int1_reordered.c
 * Description:  Matrix-multiplication function for binary convolution
 *               with reordered columns
 *
 * $Date:        09. July 2018
 * $Authors:     Manuele Rusci - manuele.rusci@unibo.it
 *               Alessandro Capotondi - alessandro.capotondi@unibo.it
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_math.h"


#define BIN_SIZE(x)	((x)>>3)        // size in byte
#define BIN_SIZE_INT32(x) ((x)>>5)  // size in 32-bit words

#define BIT_POS(x) ((x) & 0x1f)     // bit position in a 32-bit space
#define BIT_ADDR(x) ((x)>>5)        // bit address in a 32-bit space

  /**
   * @brief SW-emulated Bit-counting operator
   * @param[in]  input     bit array operand
   * @return     The function returns the bit count
   *
   */
static inline int32_t popcnt32(uint32_t v) {
  int c ;
  v = v - ((v >> 1) & 0x55555555U);
  v = (v & 0x33333333U) + ((v >> 2) & 0x33333333U);
  v = (v + (v >> 4)) & 0x0f0f0f0fU;
  c = (v * 0x01010101U) >> 24;
  return c;
}

  /**
   * @brief Matrix-multiplication function for convolution with reordered columns
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

uint32_t *arm_nn_mat_mult_kernel_int1_reordered( const uint32_t * pA, // weight buffer
                                                 const uint32_t * pInBuffer, // input reordered buffer
                                                 const uint16_t ch_im_out, // output channel dim
                                                 const uint32_t numCol_A, // receptive field dim
												 const int16_t * pThreshold, // pointer to the threshold array
												 uint32_t * pOut) // output buffer
{

#if defined (ARM_MATH_DSP)
	/* set up the second output pointers */
	int       i_feat_in, i;

	//number of 32-bit features chuncks within the input and oputput activation maps
	int n_input_block = BIN_SIZE_INT32(numCol_A);
	int n_output_block = BIN_SIZE_INT32(ch_im_out);
	const uint32_t * pIn  = pInBuffer;
	const uint32_t * pIn2 = pInBuffer + n_input_block;

	int16_t accum, accum2;
	uint32_t * pOut2 =  pOut + n_output_block;

	// cleaning output space
	for(i = 0; i < n_output_block ; i++){
	  pOut[i] = 0;
	  pOut2[i] = 0;
	}

	// xnor popcnt convolution
	for( i = 0; i < ch_im_out; i++){
		accum = 0;
		accum2 = 0;
		for(i_feat_in=0; i_feat_in<n_input_block; i_feat_in++) {
			accum += (int16_t) popcnt32(~(pIn[i_feat_in]^pA[i_feat_in]));
			accum2 += (int16_t) popcnt32(~(pIn2[i_feat_in]^pA[i_feat_in]));
		}
		pA += n_input_block;

		//thresholding and compression
		if(accum >= pThreshold[i])
			pOut[BIT_ADDR(i)] |= (1<<BIT_POS(i));
		if(accum2 >= pThreshold[i])
			pOut2[BIT_ADDR(i)] |= (1<<BIT_POS(i));
	}

	// skip one coloumn because of the two performed convolution
	pOut += BIN_SIZE_INT32(2*ch_im_out);

	/* return the new output pointer with offset */
	return pOut;

#else

    /* To be Completed */
    #error "Binary Convolution Operation not supported"

    return NULL;
#endif                          /* ARM_MATH_DSP */
}
