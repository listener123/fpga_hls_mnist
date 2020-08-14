#pragma once
#include <ap_int.h>
//初始化网络配置
#define BATCH_SIZE 1

#define config_pad_1_f_H 28
#define config_pad_1_f_W  28
#define config_pad_1_f_in_ch  1
#define config_pad_1_S  1
#define config_pad_1_w_k  3

#define config_conv_1_f_H  30
#define config_conv_1_f_W  30
#define config_conv_1_f_in_ch  1
#define config_conv_1_f_out_ch  16
#define config_conv_1_S  1
#define config_conv_1_w_k  3

#define config_pool_1_f_H  28
#define config_pool_1_f_W  28
#define config_pool_1_f_in_ch  16
#define config_pool_1_S  2
#define config_pool_1_w_k  2

#define config_pad_2_f_H  14
#define config_pad_2_f_W  14
#define config_pad_2_f_in_ch  16
#define config_pad_2_S  1
#define config_pad_2_w_k  3

#define config_conv_2_f_H  16
#define config_conv_2_f_W  16
#define config_conv_2_f_in_ch  16
#define config_conv_2_f_out_ch  32
#define config_conv_2_S  1
#define config_conv_2_w_k  3

#define config_pool_2_f_H  14
#define config_pool_2_f_W  14
#define config_pool_2_f_in_ch  32
#define config_pool_2_S  2
#define config_pool_2_w_k  2

#define config_fc_1_f_H  7
#define config_fc_1_f_W  7
#define config_fc_1_f_in_ch  32
#define config_fc_1_f_out_ch  128
#define config_fc_1_S  1
#define config_fc_1_w_k  7


#define config_fc_2_f_H  1
#define config_fc_2_f_W  1
#define config_fc_2_f_in_ch  128
#define config_fc_2_f_out_ch  10
#define config_fc_2_S  1
#define config_fc_2_w_k  1

// config data
struct config
{
	//特征图参数
	ap_uint<16> f_W;
	ap_uint<16> f_H;
	ap_uint<16> f_in_ch;
	ap_uint<16> f_out_ch;

	//权值参数（有冗余，暂保留）
	ap_uint<16> w_k;
	ap_uint<16> w_in_ch;
	ap_uint<16> w_out_ch;

	//步长
	ap_uint<16> S;

	//批处理
	ap_uint<16> batch;

	//四个维度分块
	ap_uint<16> Tw;
	ap_uint<16> Th;
	ap_uint<16> Tin;
	ap_uint<16> Tout;
};
config set_layer(ap_uint<16> f_W,ap_uint<16> f_H,ap_uint<16> f_in_ch,ap_uint<16> f_out_ch,ap_uint<16> w_k,ap_uint<16> S,ap_uint<16> batch);
//offset is used to move to next layer
void load_conv_weight(float *in, float* out,int w_in,int w_out,int w_k,int offset);

void load_conv_bias(float *in, float* out,int w_in,int offset);

void clean_temp(float * in,int length);

void conv_bias_relu(float* in, float* weight, float* bias, float* out, config c,int relu_en);

//S=1 K为奇数的版本
void padding(float *in, float *out,config c,int value);

void pool(float* in, float* out, config c,int type);

void lenet(float in[],float W[],float bias[],float out[]);

