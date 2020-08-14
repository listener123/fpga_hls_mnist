#include "top.h"
#include "config.h"
#include <iostream>
#include <fstream>
void load_data(const char* path, char* ptr, unsigned int size)
{
	std::ifstream f(path, std::ios::in | std::ios::binary);
	if (!f)
	{
		std::cout << path<<" no such file,please check the file name!/n";
		exit(0);
	}

	f.read(ptr, size);
	f.close();
}

int main()
{
	//initialize feature
	float in[28 * 28];
	load_data("D:/fpga_exp/lenet/data/test_data.bin", (char*)in, sizeof(in));
	//initialize weight and bias
	float *weight_all=new float[1*16*3*3+16*32*3*3+32*128*7*7+128*10*1*1];
	float *bias_all=new float[16+32+128+10];

	float w1[3 * 3 * 1 * 16];
	load_data("D:/fpga_exp/lenet/data/W_conv1.bin", (char*)w1, sizeof(w1));
	float w2[3 * 3 * 16 * 32];
	load_data("D:/fpga_exp/lenet/data/W_conv2.bin", (char*)w2, sizeof(w2));
	float *w3=new float[7 * 7 * 32 * 128];
	load_data("D:/fpga_exp/lenet/data/W_fc1.bin", (char*)w3, 7 * 7 * 32 * 128 * 4);
	float w4[1 * 1 * 128 * 10];
	load_data("D:/fpga_exp/lenet/data/W_fc2.bin", (char*)w4, sizeof(w4));

	for(int i=0;i<3*3*1*16;i++) weight_all[i]=w1[i];
	for(int i=0;i<3*3*16*32;i++) weight_all[3*3*1*16+i]=w2[i];
	for(int i=0;i<7*7*32*128;i++) weight_all[3*3*1*16+3*3*16*32+i]=w3[i];
	for(int i=0;i<1*1*128*10;i++) weight_all[3*3*1*16+3*3*16*32+7*7*32*128+i]=w4[i];

	float b1[16];
	load_data("D:/fpga_exp/lenet/data/b_conv1.bin", (char*)b1, sizeof(b1));
	float b2[32];
	load_data("D:/fpga_exp/lenet/data/b_conv2.bin", (char*)b2, sizeof(b2));
	float b3[128];
	load_data("D:/fpga_exp/lenet/data/b_fc1.bin", (char*)b3, sizeof(b3));
	float b4[10];
	load_data("D:/fpga_exp/lenet/data/b_fc2.bin", (char*)b4, sizeof(b4));

	for(int i=0;i<16;i++) bias_all[i]=b1[i];
	for(int i=0;i<32;i++) bias_all[16+i]=b2[i];
	for(int i=0;i<128;i++) bias_all[16+32+i]=b3[i];
	for(int i=0;i<10;i++) bias_all[16+32+128+i]=b4[i];

	//result
	float out[10*BATCH_SIZE]={0};
	lenet(in,weight_all,bias_all,out);
	for (int i=0;i<10;i++) std::cout<<out[i]<<" ";
	std::cout<<std::endl;
	return 0;
}
