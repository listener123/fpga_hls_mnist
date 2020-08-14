#include "top.h"
#include <iostream>
config set_layer(ap_uint<16> f_W,ap_uint<16> f_H,ap_uint<16> f_in_ch,ap_uint<16> f_out_ch,ap_uint<16> w_k,ap_uint<16> S,ap_uint<16> batch)
{
	config c;
	c.f_W=f_W;
	c.f_H=f_H;
	c.f_in_ch=f_in_ch;
	c.f_out_ch=f_out_ch;

	//权值参数（有冗余，暂保留）
	c.w_k=w_k;
	c.w_in_ch= f_in_ch;
	c.w_out_ch= f_out_ch;

	//步长
	c.S=S;

	//批处理
	c.batch=batch;

	//四个维度分块
	c.Tw=1;
	c.Th=1;
	c.Tin=1;
	c.Tout=1;
	return c;
}
//offset is used to move to next layer
void load_conv_weight(float *in, float* out,int w_in,int w_out,int w_k,int offset)
{
	for (int i = 0; i < w_k;i++)
	{
		for (int j = 0; j < w_k;j++)
		{
			for (int ti = 0; ti < w_in;ti++)
			{
				for (int to = 0; to < w_out;to++)
				{
					out[to * w_in * w_k * w_k + ti * w_k * w_k + i * w_k + j] = in[offset+i * w_out * w_in * w_k + j * w_out * w_in + ti * w_out + to];
				}
			}
		}
	}
}
void load_conv_bias(float *in, float* out,int w_in,int offset)
{

	for (int ti = 0; ti < w_in;ti++)
	{
		out[ti]=in[offset+ti];
	}

}
void clean_temp(float * in,int length)
{
	for (int i = 0; i < length;i++)
	{
		in[i]=0;
	}
}
void conv_bias_relu(float* in, float* weight, float* bias, float* out, config c,int relu_en)
{
	//in[batch*f_in_ch*f_W*f_H]

	//weight[w_out_ch*w_in_ch*w_k*w_k]

	//out[batch*f_out_ch*out_w*out_h]

	int out_w = (c.f_W - c.w_k) / c.S + 1;
	int out_h = (c.f_H - c.w_k) / c.S + 1;
	for (int batch = 0; batch < c.batch; batch++)
	{
		for (int to = 0; to < c.f_out_ch; to++)
		{
			for (int tw = 0; tw < out_w; tw++)
			{
				for (int th = 0; th < out_h; th++)
				{
					float sum = 0;
					for (int ii = 0; ii < c.w_k; ii++)
					{
						for (int jj = 0; jj < c.w_k; jj++)
						{
							int h = th * c.S + jj;
							int w = tw * c.S + ii;
							if ((h >= 0 )&&( w >= 0 )&&( h < c.f_H) &&( w < c.f_W))
							{
								for (int ti = 0; ti < c.f_in_ch; ti++)
								{
									float tp;
									tp = in[batch * c.f_in_ch * c.f_W * c.f_H + ti * c.f_W * c.f_H + w * c.f_H + h] *
										weight[to * c.f_in_ch * c.w_k * c.w_k + ti * c.w_k * c.w_k + ii * c.w_k + jj];
									//if(relu_en==2)std:: cout<< tp << " "<<std::endl;
									sum += tp;
								}
							}
						}
					}


					sum += bias[to];
					if (relu_en & (sum < 0))
						sum = 0;

					out[batch*c.f_out_ch*out_w*out_h+to * out_w * out_h+tw * out_h + th] = sum;

				}
			}
		}
	}
}
//S=1 K为奇数的版本
void padding(float *in, float *out,config c,int value)
{
	//in[batch*f_in_ch*f_W*f_H]

	//out[batch*f_in_ch*out_w*out_h]
	int p = (c.w_k - 1) / 2;//    p=((s-1)x-s+k)/2
	int out_w = c.f_W + 2 * p;
	int out_h = c.f_H + 2 * p;
	for (int batch=0;batch<c.batch;batch++)
	{
		for (int ch = 0; ch < c.f_in_ch; ch++)
		{
			for (int i = 0; i < out_w; i++)
			{
				for (int j=0;j<p;j++)
				{
					out[batch * c.f_in_ch * out_w * out_h+ch * out_w * out_h+i* out_h+j] = value;
				}

				for (int j = p; j < out_h - p; j++)
				{
					if (i < p || i >=out_w - p) out[batch * c.f_in_ch * out_w * out_h + ch * out_w * out_h + i * out_h + j] = value;
					else out[batch * c.f_in_ch * out_w * out_h + ch * out_w * out_h + i * out_h + j] = in[batch * c.f_in_ch * c.f_W * c.f_H + ch * c.f_W * c.f_H + (i - p) * c.f_H + j - p];
				}

				for (int j = out_h-p; j < out_h; j++)
				{
					out[batch * c.f_in_ch * out_w * out_h + ch * out_w * out_h + i * out_h + j] = value;
				}

			}
		}
	}
}
void pool(float* in, float* out, config c,int type)
{
	//in[batch*f_in_ch*f_W*f_H]

	//out[batch*f_in_ch*out_w*out_h]
	//type0 max type1 average
	//compute
	int out_w = (c.f_W - c.w_k) / c.S + 1;
	int out_h = (c.f_H - c.w_k) / c.S + 1;

	for (int batch = 0; batch < c.batch; batch++)
	{
		for (int row = 0; row < out_w; row += c.Tw)
		{
			for (int col = 0; col < out_h; col += c.Th)
			{

				for (int ti = 0; ti < c.f_in_ch; ti += c.Tin)
					{

						for (int i = 0; i < c.w_k; i++)
						{
							for (int j = 0; j < c.w_k; j++)
							{
								for (int trr = row; trr < row + c.Tw; trr++)
								{
									for (int tcc = col; tcc < col + c.Th; tcc++)
									{

										for (int tii = ti; tii < ti + c.Tin; tii++)
											{
												if (type == 1) //average pool
												{
													out[batch * out_h * out_w * c.f_in_ch + tii * out_h * out_w + trr * out_h + tcc] +=
														in[batch * c.f_H * c.f_H * c.f_in_ch + tii * c.f_H * c.f_W + (c.S * trr + i) * c.f_H + c.S * tcc + j] / (c.w_k * c.w_k);
												}
												if (type == 0) //max pool
												{
													float a = out[batch * out_h * out_w * c.f_in_ch + tii * out_h * out_w + trr * out_h + tcc];
													float b = in[batch * c.f_H * c.f_H * c.f_in_ch + tii * c.f_H * c.f_W + (c.S * trr + i) * c.f_H + c.S * tcc + j];
													if (b > a) out[batch * out_h * out_w * c.f_in_ch + tii * out_h * out_w + trr * out_h + tcc] = b;
												}
											}

									}
								}
							}
						}
					}

			}
		}
	}
}

void lenet(float in[],float W[],float bias[],float out[])
{
#pragma HLS INTERFACE m_axi depth=1*10 port=out offset=slave bundle=DATA_OUT
#pragma HLS INTERFACE m_axi depth=4294967295 port=bias offset=slave bundle=BIAS
#pragma HLS INTERFACE m_axi depth=4294967295 port=W offset=slave bundle=WEIGHT
#pragma HLS INTERFACE m_axi depth=28*28*1 port=in offset=slave bundle=DATA_IN
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL

	float temp_weight[200704];//128*32*7*7 max memory in lenet
	float temp_bias[128];//128 max memory in lenet
	float temp_feature1[12544];//28*28*1 max memory in lenet
	float temp_feature2[12544];
	int offset_weight=0;
	int offset_bias=0;
	//initialize
	clean_temp(temp_feature1,16*28*28);
	clean_temp(temp_feature2,16*28*28);
	clean_temp(temp_weight,128*32*7*7);
	clean_temp(temp_bias,128);

	config config_pad_1=set_layer(config_pad_1_f_H, config_pad_1_f_W, config_pad_1_f_in_ch, config_pad_1_f_in_ch, config_pad_1_w_k, config_pad_1_S,BATCH_SIZE);
	config config_conv_1 = set_layer(config_conv_1_f_H, config_conv_1_f_W, config_conv_1_f_in_ch, config_conv_1_f_out_ch, config_conv_1_w_k, config_conv_1_S, BATCH_SIZE);
	config config_pool_1 = set_layer(config_pool_1_f_H, config_pool_1_f_W, config_pool_1_f_in_ch, config_pool_1_f_in_ch, config_pool_1_w_k, config_pool_1_S, BATCH_SIZE);

	config config_pad_2 = set_layer(config_pad_2_f_H, config_pad_2_f_W, config_pad_2_f_in_ch, config_pad_2_f_in_ch, config_pad_2_w_k, config_pad_2_S, BATCH_SIZE);
	config config_conv_2 = set_layer(config_conv_2_f_H, config_conv_2_f_W, config_conv_2_f_in_ch, config_conv_2_f_out_ch, config_conv_2_w_k, config_conv_2_S, BATCH_SIZE);
	config config_pool_2 = set_layer(config_pool_2_f_H, config_pool_2_f_W, config_pool_2_f_in_ch, config_pool_2_f_in_ch, config_pool_2_w_k, config_pool_2_S, BATCH_SIZE);

	config config_fc_1 = set_layer(config_fc_1_f_H, config_fc_1_f_W, config_fc_1_f_in_ch, config_fc_1_f_out_ch, config_fc_1_w_k, config_fc_1_S, BATCH_SIZE);
	config config_fc_2 = set_layer(config_fc_2_f_H, config_fc_2_f_W, config_fc_2_f_in_ch, config_fc_2_f_out_ch, config_fc_2_w_k, config_fc_2_S, BATCH_SIZE);

	//pad1
	padding(in,temp_feature1, config_pad_1, 0);

	//conv1
	load_conv_weight(W, temp_weight, config_conv_1_f_in_ch, config_conv_1_f_out_ch, config_conv_1_w_k, offset_weight);
	load_conv_bias(bias,temp_bias,config_conv_1_f_out_ch,offset_weight);
	conv_bias_relu(temp_feature1,temp_weight,temp_bias,temp_feature2,config_conv_1,1);

	clean_temp(temp_feature1,16*28*28);
	clean_temp(temp_weight,128*32*7*7);
	clean_temp(temp_bias,128);

	//pool1
	pool(temp_feature2,temp_feature1,config_pool_1,0);

	clean_temp(temp_feature2,16*28*28);

	//pad2
	padding(temp_feature1,temp_feature2, config_pad_2, 0);

	clean_temp(temp_feature1,16*28*28);

	//conv2
	offset_weight+=config_conv_1_f_in_ch*config_conv_1_f_out_ch*config_conv_1_w_k*config_conv_1_w_k;
	offset_bias+=config_conv_1_f_out_ch;
	load_conv_weight(W, temp_weight, config_conv_2_f_in_ch, config_conv_2_f_out_ch, config_conv_2_w_k, offset_weight);
	load_conv_bias(bias,temp_bias,config_conv_2_f_out_ch,offset_bias);
	conv_bias_relu(temp_feature2,temp_weight,temp_bias,temp_feature1,config_conv_2,1);
	/**/
	clean_temp(temp_feature2,16*28*28);
	clean_temp(temp_weight,128*32*7*7);
	clean_temp(temp_bias,128);

	//pool2
	pool(temp_feature1,temp_feature2,config_pool_2,0);

	clean_temp(temp_feature1,16*28*28);

	//fc1
	offset_weight+=config_conv_2_f_in_ch*config_conv_2_f_out_ch*config_conv_2_w_k*config_conv_2_w_k;
	offset_bias+=config_conv_2_f_out_ch;
	load_conv_weight(W, temp_weight, config_fc_1_f_in_ch, config_fc_1_f_out_ch, config_fc_1_w_k, offset_weight);
	load_conv_bias(bias,temp_bias,config_fc_1_f_out_ch,offset_bias);
	conv_bias_relu(temp_feature2,temp_weight,temp_bias,temp_feature1,config_fc_1,1);

	clean_temp(temp_feature2,16*28*28);
	clean_temp(temp_weight,128*32*7*7);
	clean_temp(temp_bias,128);

	//fc2
	offset_weight+=config_fc_1_f_in_ch*config_fc_1_f_out_ch*config_fc_1_w_k*config_fc_1_w_k;
	offset_bias+=config_fc_1_f_out_ch;
	load_conv_weight(W, temp_weight, config_fc_2_f_in_ch, config_fc_2_f_out_ch, config_fc_2_w_k, offset_weight);
	load_conv_bias(bias,temp_bias,config_fc_2_f_out_ch,offset_bias);
	conv_bias_relu(temp_feature1,temp_weight,temp_bias,out,config_fc_2,1);

}
