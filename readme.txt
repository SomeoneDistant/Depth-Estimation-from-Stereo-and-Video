This data is used for research purposes only and may not be commercially distributed. If this data is used in an academic publication, please cite one of the following papers:

[1] Guofeng Zhang, Jiaya Jia, Tien-Tsin Wong and Hujun Bao. Consistent Depth Maps Recovery from a Video Sequence. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 31(6):974-988, 2009.
[2] Guofeng Zhang, Jiaya Jia, Tien-Tsin Wong and Hujun Bao. Recovering Consistent Video Depth Maps via Bundle Optimization. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2008(oral).

This data includes a sequence of 141 frames. The image resolution is 960*540. 

The camera parameters for each frame is included in "cameras.txt". The format is:
#Frame Number 
#Camera Parameters of each frame: K   R   T ------  X_{world} = R * X_{camera} + T
K (intrinsic matrix): the first 3*3 matrix 
R (camera rotation): the second 3*3 matrix 
T (camera position):  the final line 


"src" directory contains all source images.
"depth" directory contains all depth maps.


Each depth map is stored as a raw float file. Note that the depth value is stored by inverse depth (disparity), i.e. 1/z. Note the (0,0) coordinate of an image is located in the top left corner.

For example, about the 0th frame of depth map file "_depth0.raw", you can read it using the following C++ code:

#include "stdio.h"

int main()
{
	int iWidth = 960;
	int iHeight = 540;
	FILE* fp = fopen("_depth0.raw", "rb");

	float * data = new float[iWidth*iHeight];

	fread(data, sizeof(float), iWidth*iHeight, fp);

	//The depth value of pixel (i,j)
	int i = 426, j = 391;
	float z = 1.0 / data[i+j*iWidth];

	printf("z:%f\n",z);

	return 0;
}