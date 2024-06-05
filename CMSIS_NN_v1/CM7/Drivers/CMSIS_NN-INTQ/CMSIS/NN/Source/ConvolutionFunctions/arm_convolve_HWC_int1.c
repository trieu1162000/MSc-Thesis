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
 * Title:        arm_convolve_HWC_int1.c
 * Description:  INT1 (binary) convolution
 *
 * $Date:        09. July 2018
 * $Authors:     Manuele Rusci - manuele.rusci@unibo.it
 *               Alessandro Capotondi - alessandro.capotondi@unibo.it
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

#define BIN_SIZE(x) ((x)>>3)
#define BIN_SIZE_INT32(x) ((x)>>5)


/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

  /**
   * @brief INT1 (binary) convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input 
   * @param[in]       pThreshold  pointer to threshold array
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return          The function returns either
   */


arm_status
arm_convolve_HWC_int1(const uint32_t * Im_in,
                          const uint16_t dim_im_in,
                          const uint16_t ch_im_in,
                          const uint32_t * wt,
                          const uint16_t ch_im_out,
                          const uint16_t dim_kernel,
                          const uint16_t padding,
                          const uint16_t stride,
						  uint8_t * Im_out,
                          const uint16_t dim_im_out, 
						  uint32_t * bufferA,
						  const int16_t * pThreshold,
						  int8_t * bufferB)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_out_y, i_out_x, i_ker_y, i_ker_x;

    /*
     *  Here we use bufferA as int16_t internally as computation are done with int16_t level
     *  im2col are done to output in int16_t format from int8_t input
     */

    uint32_t  *pOut = (uint32_t* )Im_out;

    if (ch_im_in % 32 != 0 || ch_im_out % 32 != 0)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    int16_t n_ifeat_block = BIN_SIZE_INT32(ch_im_in);
    int16_t i_feat;
    int idx = 0;

    uint32_t * ptr_input;


    /*
     *  Here we split the entire matrix into three regions depending on the padding situation
     *    Top: i_out_y from 0 to padding - 1
     * Middle: i_out_y from padding to dim_im_out-padding-1
     * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
     */

    /* top part */
    for (i_out_y = 0; i_out_y < padding; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function to read input data*/
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
          					if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
          					{
            						for(i_feat=0; i_feat<n_ifeat_block; i_feat++)
                        {
            							bufferA[idx++] = 0;
            						}
          					} 
                    else 
                    {
            						ptr_input = ((uint32_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * n_ifeat_block); //offset multiple of 32 bits
            						for(i_feat=0; i_feat<n_ifeat_block; i_feat++)
                        {
            							bufferA[idx++] = ptr_input[i_feat];
            						}
          					}

                }
            }

            if(idx == 2*n_ifeat_block*dim_kernel*dim_kernel ){
            	idx = 0;
            	pOut = arm_nn_mat_mult_kernel_int1_reordered(wt,
                                                          bufferA,
                                                          ch_im_out,
                                                          ch_im_in * dim_kernel * dim_kernel,
  												                                pThreshold,
  												                                pOut);
            }

        }
    }

    /* middle part, here we also divide the x into left, mid and right */
    for (; i_out_y < dim_im_out - padding; i_out_y++)
    {

        /* left part */
        for (i_out_x = 0; i_out_x < padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
        				for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
        				{
          					if (i_ker_x < 0 || i_ker_x >= dim_im_in)
          					{
            						for(i_feat=0; i_feat<n_ifeat_block; i_feat++)
                        {
            							 bufferA[idx++] = 0;
            						}
          					} 
                    else 
                    {
            						ptr_input = ((uint32_t *) Im_in  + (i_ker_y * dim_im_in + i_ker_x) * n_ifeat_block);
            						for(i_feat=0; i_feat<n_ifeat_block; i_feat++)
                        {
            							 bufferA[idx++] = ptr_input[i_feat];
            						}
          					}
        				}
            }

            if(idx == 2*n_ifeat_block*dim_kernel*dim_kernel )
            {
              	idx = 0;
              	pOut = arm_nn_mat_mult_kernel_int1_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
                                														pThreshold,
                                														pOut);
            }
        }

        /* mid part */
        for (; i_out_x < dim_im_out - padding; i_out_x++)
        {
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
        				ptr_input = ((uint32_t *) Im_in  + (i_ker_y*dim_im_in+i_out_x*stride-padding)*n_ifeat_block);
        				for(i_feat=0; i_feat<n_ifeat_block* dim_kernel; i_feat++)
                {
        				  	bufferA[idx++] = ptr_input[i_feat];
        				}
            }

            if(idx == 2*n_ifeat_block*dim_kernel*dim_kernel )
            {
              	idx = 0;
             	  pOut = arm_nn_mat_mult_kernel_int1_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
                                														pThreshold,
                                														pOut);
            }
        }


        /* right part */
        for (; i_out_x < dim_im_out; i_out_x++)
        {
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
        				for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
        				{
          					if (i_ker_x < 0 || i_ker_x >= dim_im_in)
          					{
            						for(i_feat=0; i_feat<n_ifeat_block; i_feat++)
                        {
            						    bufferA[idx++] = 0;
            						}
          					} 
                    else 
                    {
          						  ptr_input = ((uint32_t *) Im_in  + (i_ker_y * dim_im_in + i_ker_x) * n_ifeat_block);
          						  for(i_feat=0; i_feat<n_ifeat_block; i_feat++)
                        {
          							    bufferA[idx++] = ptr_input[i_feat];
          						  }
          					}
        				}
            }

            if(idx == 2*n_ifeat_block*dim_kernel*dim_kernel )
            {
              	idx = 0;
              	pOut = arm_nn_mat_mult_kernel_int1_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
                                														pThreshold,
                                														pOut);
            }
        }
    }

    /* bottom part */
    for (; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
    				for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
    				{
      					if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
      					{
        						for(i_feat=0; i_feat<n_ifeat_block; i_feat++)
                    {
        						    bufferA[idx++] = 0;
        						}
      					} 
                else 
                {
        						ptr_input = ((uint32_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * n_ifeat_block);
        						for(i_feat=0; i_feat<n_ifeat_block; i_feat++)
                    {
        						    bufferA[idx++] = ptr_input[i_feat];
        						}
      					}
    				}
            }

            if(idx == 2*n_ifeat_block*dim_kernel*dim_kernel )
            {
              	idx = 0;
              	pOut = arm_nn_mat_mult_kernel_int1_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
  														                              pThreshold,
  														                              pOut);
            }
        }
    }


#else
    /* Implementation for Cortex-M0 and Cortex-M3: TO BE COMPLETED */
	#error "Symmetric int1 Convolution Layer not supported (yet) on this device"
#endif                          /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNConv group
 */
