/*
 * ov7670Reg.h
 *
 *  Created on: 2017/08/25
 *      Author: take-iwiw
 *
 *  Modified on: 2020/09/17
 *  Modified by: Enrique Phan
 *  Github: PHANzgz
 *
 */

#ifndef OV7670_OV7670REG_H_
#define OV7670_OV7670REG_H_

#define REG_EOT 0xFF

const uint8_t OV7670_reg[][2] = {

		/* Color mode and resolution settings */
		  {0x12,             0x14},         // QVGA, RGB
		//{0x12,             0xCU},         // QCIF (176*144), RGB
		  {0x8C,             0x00},         // RGB444 Disable
		  {0x40,             0xD0},         // RGB565
		  {0x3A,             0xCU},         // UYVY
		  {0x3D,             0x80},         // gamma enable, UV auto adjust, UYVY
		  {0xB0,             0x84},         // Important!
		  /* Clock settings */
		  {0x0C,             0x04},         // DCW enable
		  {0x3E,             0x19},         // manual scaling, pclk/=2
		  {0x70,             0x3A},         // scaling_xsc
		  {0x71,             0x35},         // scaling_ysc
		  {0x72,             0x11},         // down sample by 2
		  {0x73,             0xF1},         // DSP clock /= 2
		  /* Windowing */
		  {0x17,             0x16},         // HSTART
		  {0x18,             0x04},         // HSTOP
		  {0x32,             0x80},         // HREF
		  {0x19,             0x03},         // VSTART =  14 ( = 3 * 4 + 2)
		  {0x1A,             0x7B},         // VSTOP  = 494 ( = 123 * 4 + 2)
		  {0x03,             0x0A},         // VREF (VSTART_LOW = 2, VSTOP_LOW = 2)
		  /* Color matrix coefficient */
		#if 0
		  {0x4F,             0xB3},
		  {0x50,             0xB3},
		  {0x51,             0x00},
		  {0x52,             0x3D},
		  {0x53,             0xA7},
		  {0x54,             0xE4},
		  {0x58,             0x9E},
		#else
		  {0x4F,             0x80},
		  {0x50,             0x80},
		  {0x51,             0x00},
		  {0x52,             0x22},
		  {0x53,             0x5E},
		  {0x54,             0x80},
		  {0x58,             0x9E},
		#endif
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////Chỉnh màu ở đây/////////////////////////////////////////////
		{0x13,             0xE7},
		{0x5F, 0xFF}, // Phạm vi gain màu xanh dương (Blue)
		{0x60, 0x9F}, // Phạm vi gain màu đỏ (Red)
		{0x61, 0x7F}, // Phạm vi gain màu xanh lá (Green)
		{0x41,             0x38},         // edge enhancement, de-noise, AWG gain enabled
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
		  /* gamma curve */
		#if 1
		  {0x7B,             16},
		  {0x7C,             30},
		  {0x7D,             53},
		  {0x7E,             90},
		  {0x7F,             105},
		  {0x80,             118},
		  {0x81,             130},
		  {0x82,             140},
		  {0x83,             150},
		  {0x84,             160},
		  {0x85,             180},
		  {0x86,             195},
		  {0x87,             215},
		  {0x88,             230},
		  {0x89,             244},
		  {0x7A,             16},
		#else
		  /* gamma = 1 */
		  {0x7B,             4},
		  {0x7C,             8},
		  {0x7D,             16},
		  {0x7E,             32},
		  {0x7F,             40},
		  {0x80,             48},
		  {0x81,             56},
		  {0x82,             64},
		  {0x83,             72},
		  {0x84,             80},
		  {0x85,             96},
		  {0x86,             112},
		  {0x87,             144},
		  {0x88,             176},
		  {0x89,             208},
		  {0x7A,             64},
		#endif
		  /* FPS */
		  //{0x6B,             0x4A},         // PLL  x4
		  {0x11,             0x00},         // Pre-scalar = 1/1
		  /* Others */
		  {0x1E,             0x31},         // Mirror flip
		//{0x42,            0x08},         // Test screen with color bars
		  {0xFF,             0xFF},

		  {REG_EOT, REG_EOT},
	};


#endif /* OV7670_OV7670REG_H_ */
