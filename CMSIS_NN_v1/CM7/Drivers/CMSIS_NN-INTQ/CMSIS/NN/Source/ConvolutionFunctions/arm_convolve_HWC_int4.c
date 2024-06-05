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
 * Title:        arm_convolve_HWC_int4.c
 * Description:  INT4 convolution
 *
 * $Date:        09. July 2018
 * $Authors:     Manuele Rusci - manuele.rusci@unibo.it
 *               Alessandro Capotondi - alessandro.capotondi@unibo.its
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

  /**
   * @brief INT4 convolution function
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
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
   *
   * bufferB size: 0
   *
   * <b>Input dimension constraints:</b>
   *
   * ch_im_in is multiple of 8    ( because of the 32-bit read)
   *
   * ch_im_out is multiple of 2    ( because 2x2 mat_mult kernel )
   *
   * The im2col converts the INT4 tensor input into INT16 column, which is stored in
   * bufferA. There is reordering happening during this im2col process with
   * arm_int4_to_int16_reordered_no_shift. 
   *
   * The computation kernel arm_nn_mat_mult_kernel_int4_int16_reordered does the
   * GEMM computation with the reordered columns.
   *
   * To speed-up the determination of the padding condition, we split the
   * computation into 3x3 parts, i.e., {top, mid, bottom} X {left, mid, right}.
   * This reduces the total number of boundary condition checks and improves
   * the data copying performance.
   */

arm_status
arm_convolve_HWC_int4(const int8_t * Im_in,
                         const uint16_t dim_im_in,
                         const uint16_t ch_im_in,
                         const int8_t * wt,
                         const uint16_t ch_im_out,
                         const uint16_t dim_kernel,
                         const uint16_t padding,
                         const uint16_t stride,
                         int8_t * Im_out,
                         const uint16_t dim_im_out, 
                         int16_t * bufferA,
						 const int16_t * pThreshold,
                         int8_t * bufferB)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_out_y, i_out_x, i_ker_y, i_ker_x;

    /*
     *  Here we use bufferA as int16_t internally as computation are done with int16_t level
     *  im2col are done to output in int16_t format from INT4 input
     */
    int16_t    *pBuffer = bufferA;
    int8_t     *pOut = Im_out;

    if (ch_im_in % 8 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

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
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t)*ch_im_in);
                    } else
                    {
                        arm_int4_to_int16_reordered_no_shift
                            ((int8_t *) Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1), pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in ;
                }
            }


            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int4_int16_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
															pThreshold,
															pOut);	
                /* counter reset */
                pBuffer = bufferA;
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
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(int16_t)*ch_im_in);
                    } else
                    {
                        arm_int4_to_int16_reordered_no_shift
                            ((int8_t *) Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1 ),
                            		pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int4_int16_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
															pThreshold,
															pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }


        /* mid part */
        for (; i_out_x < dim_im_out - padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                arm_int4_to_int16_reordered_no_shift(
                		(int8_t *) Im_in + (((i_ker_y * dim_im_in + i_out_x *stride - padding) * ch_im_in) >> 1 ),
                				pBuffer, ch_im_in * dim_kernel);
                pBuffer += ch_im_in * dim_kernel;
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int4_int16_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
															pThreshold,
															pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }



        /* right part */
        for (; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(int16_t)*ch_im_in);
                    } else
                    {
                        arm_int4_to_int16_reordered_no_shift
                            ((int8_t *) Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1 ),
                            		pBuffer,
                            		ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int4_int16_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
															pThreshold,
															pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }



    }

    for (; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(int16_t)*ch_im_in);
                    } else
                    {
                        arm_int4_to_int16_reordered_no_shift
                            ((int8_t *) Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1 ),
                            		pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int4_int16_reordered(wt,
                                                            bufferA,
                                                            ch_im_out,
                                                            ch_im_in * dim_kernel * dim_kernel,
															pThreshold,
															pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }


#else
    /* Implementation for Cortex-M0 and Cortex-M3: TO BE COMPLETED */
    #error "Symmetric int4 Convolution Layer not supported (yet) on this device"
#endif                          /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNConv group
 */
