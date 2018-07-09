// cv_ceshi1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace cv;
using namespace std;


double point2Line(Point p1, Point lp1, Point lp2)
{
	double a, b, c, dis;
	// 化简两点式为一般式
	// 两点式公式为(y - y1)/(x - x1) = (y2 - y1)/ (x2 - x1)
	// 化简为一般式为(y2 - y1)x + (x1 - x2)y + (x2y1 - x1y2) = 0
	// A = y2 - y1
	// B = x1 - x2
	// C = x2y1 - x1y2
	a = lp2.y - lp1.y;
	b = lp1.x - lp2.x;
	c = lp2.x * lp1.y - lp1.x * lp2.y;
	// 距离公式为d = |A*x0 + B*y0 + C|/√(A^2 + B^2)
	dis = abs(a * p1.x + b * p1.y + c) / sqrt(a * a + b * b);
	return dis;
};


bool Point_sameleval(Point a,Point b,double d_x)
{
	if ((abs(a.y - b.y) < 20) && (abs(a.x - b.x - d_x)) < 15)
		return true;
	else
	{
		return false;
	}
}
double real_length(Point3d a, Point3d b)
{
	double d_x = a.x - b.x;
	double d_y = a.y - b.y;
	double d_z = a.z - b.z;
	return sqrt(pow(d_x, 2) + pow(d_y, 2) + pow(d_z, 2));
}
double my_real_length(Point a, Point b)
{
	double d_x = a.x - b.x;
	double d_y = a.y - b.y;
	return sqrt(pow(d_x, 2) + pow(d_y, 2) );
}
Point mid_point(Point a, Point b)
{
	Point mid;
	mid.x = (a.x + b.x) / 2;
	mid.y = (a.y + b.y) / 2;
	return mid;
}

vector<Point> my_select(vector<Point> input_array)
{
	//默认0为已知端点
	vector<Point> selected_point;
	double max_line=0;
	int index = 0;
	for (int i = 0; i<input_array.size();i++)
	{
		if (max_line<my_real_length(input_array[0], input_array[i]))
		{
			max_line = my_real_length(input_array[0], input_array[i]);
			index = i;
		}
	}
	selected_point.push_back(input_array[index]);
	selected_point.push_back(input_array[0]);
	input_array.erase(input_array.begin() + index);
	input_array.erase(input_array.begin());
	while (input_array.size() != 0)
	{
		double dis=0;
		int index_dis = 0;
		bool flag = false;
		for (int j = 0; j < selected_point.size() - 1; j++)
		{
			for (int i = 0; i < input_array.size();)
			{
				
				double dis1 = point2Line(input_array[i], selected_point[j], selected_point[selected_point.size() - 1]);
				if (dis1 < 10)
				{
					cout << input_array[i] << endl;
					input_array.erase(input_array.begin() + i);
				}
				else if (dis1 > dis)
				{
					dis = dis1;
					index_dis = i;
					i++;
					flag = true;
				}
				else
					i++;
			}

		}
		if (flag)
		{
			selected_point.push_back(input_array[index_dis]);
			input_array.erase(input_array.begin() + index_dis);
			flag = false;
		}
	}
	return selected_point;
}
int Otsu2(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	int x = 0, y = 0;
	int pixelCount[256];
	float pixelPro[256];
	int i, j, pixelSum = width * height, threshold = 0;

	//初始化
	for (i = 0; i < 256; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//统计灰度级中每个像素在整幅图像中的个数
	for (i = y; i < height; i++)
	{
		for (j = x; j <width; j++)
		{
			pixelCount[src.at<uchar>(i, j)]++;
		}
	}


	//计算每个像素在整幅图像中的比例
	for (i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
	}

	//经典ostu算法,得到前景和背景的分割
	//遍历灰度级[0,255],计算出方差最大的灰度值,为最佳阈值
	float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
	for (i = 0; i < 256; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;

		for (j = 0; j < 256; j++)
		{
			if (j <= i) //背景部分
			{
				//以i为阈值分类，第一类总的概率
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else       //前景部分
			{
				//以i为阈值分类，第二类总的概率
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}

		u0 = u0tmp / w0;        //第一类的平均灰度
		u1 = u1tmp / w1;        //第二类的平均灰度
		u = u0tmp + u1tmp;      //整幅图像的平均灰度
		//计算类间方差
		deltaTmp = w0 * (u0 - u)*(u0 - u) + w1 * (u1 - u)*(u1 - u);
		//找出最大类间方差以及对应的阈值
		if (deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	//返回最佳阈值;
	return threshold;
}
void get_contours(vector<vector<Point>>&contours, double minarea, double maxarea)
{
	//double maxarea = 0;
	int maxAreaIdx = 0;
	int whRatio = 4;
	for (int i = 0; i<contours.size();)
	{

		double tmparea = fabs(contourArea(contours[i]));


		Rect aRect = boundingRect(contours[i]);

		if ((tmparea>maxarea) || (tmparea < minarea) ||
			((aRect.width / aRect.height)>whRatio) ||
			((aRect.height / aRect.width)>whRatio))
		{
			//删除面积小于设定值的轮廓  
			contours.erase(contours.begin() + i);
			//cout << "delete  area" << endl;
			//continue;
		}
		else
		{
			i++;
		}
	}
}

vector<vector<Point> > getContours(Mat src, double minarea, double maxarea, int canny_1, int canny_2)
{
	Mat  dst, canny_output;
	/// Load source image and convert it to gray  
	//src = imread(Imgname, 0);
	if (!src.data)
	{
		cout << "read data error!" << std::endl;
	}
	blur(src, src, Size(3, 3));

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Canny(src, canny_output, canny_1, canny_2, 3);
	//imshow("canny",canny_output);
	Mat dst1 = Mat::zeros(src.size(), CV_8UC1);
	Mat dst2 = Mat::zeros(src.size(), CV_8UC1);

	findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	get_contours(contours, minarea, maxarea);
	vector<vector<Point>> ploy(contours.size());
	vector<vector<Point>> hull(contours.size());

	for (int i = 0; i<contours.size(); i++)
	{
		convexHull(contours[i], hull[i], false);
		approxPolyDP(hull[i], ploy[i], arcLength(hull[i], 1) * 0.1, true);
		//convexHull(contours[i], hull[i], false);
		drawContours(dst1, ploy, i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(dst2, hull, i, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());

	}
		//imshow("ploy", dst1);
		//imshow("contours", dst2);

	//imshow("contours", dst1);
	//imshow("hull", dst2);
	return ploy;
}
/*
struct corner{
	Point corner_point;
	double dis_corner;

};
*/
struct obj_contours{

	vector<Point> corner;
	vector<Vec3d> corner_selected;

	vector<Point> contours;
	Moments mu;
	int detcet_num = 0;
	vector<Point3d> corner_world;
};

struct match_shape{
	obj_contours shape_L;
	obj_contours shape_R;
};
const int imageWidth = 680;                             //摄像头的分辨率    
const int imageHeight = 480;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域    
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表    
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q  
Mat xyz;              //三维坐标  

Point origin;         //鼠标按下的起始点  
Rect selection;      //定义矩形选框  
bool selectObject = false;    //是否选择对象  

int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
//StereoBM bm = StereoBM::create(16, 9);


//事先标定好的相机的参数
//fx 0 cx
//0 fy cy
//0 0  1


Mat cameraMatrixL = (Mat_<double>(3, 3) << 604.27364, 0, 226.33472,
	0, 585.81090, 213.80692,
	0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.00164, -0.00276, -0.01207, -0.01202, 0.00000);

Mat 
 = (Mat_<double>(3, 3) << 598.43642, 0, 277.65691,
	0, 586.80375, 237.30513,
	0, 0, 1);

Mat distCoeffR = (Mat_<double>(5, 1) << 0.08600, -0.72086, 0.00248, 0.00326, 0.00000);

Mat T = (Mat_<double>(3, 1) << -96.95129, 4.08323, 5.93219);//T平移向量   
Mat rec = (Mat_<double>(3, 1) << 0.06487, 0.02678, -0.00027);//rec旋转向量  
Mat R;//R 旋转矩阵 


int main()
{

	VideoCapture cam1;
	VideoCapture cam2;
	cam1.open(1);
	cam2.open(2);




	Rodrigues(rec, R); //Rodrigues变换
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);


Mat rgbImageL, rgbImageR;
while (cam2.read(rgbImageL) && cam1.read(rgbImageR))
	{


		//cout << rgbImageL.cols << rgbImageL.rows;
		cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
		cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

		//imshow("ImageL Before Rectify", grayImageL);
		//imshow("ImageR Before Rectify", grayImageR);


		remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
		remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);


		Mat rgbRectifyImageL, rgbRectifyImageR;
		cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
		cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);

		//单独显示
		//rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
		//rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
		//imshow("ImageL After Rectify", rgbRectifyImageL);
		//imshow("ImageR After Rectify", rgbRectifyImageR);


		Mat copy_L = rgbRectifyImageL.clone();
		Mat copy_R = rgbRectifyImageR.clone();

		moveWindow("frame_R", rgbImageL.cols, 0);
		moveWindow("frame_L", 0, 0);
		vector<Point2d> corners_1;
		vector<Point2d> corners_2;

		int canny_1 = Otsu2(grayImageL);
		int canny_2 = Otsu2(grayImageR);


		//计算轮廓的质心

		vector<vector<Point>> contours_1;
		vector<obj_contours> obj_contours_1;
		contours_1 = getContours(rectifyImageL, 300, 10000, 0.5*canny_1, canny_1);
		for (int i = 0; i < contours_1.size(); i++)
		{
			obj_contours obj;
			obj.contours = contours_1[i];
			obj.corner = contours_1[i];
			obj.mu = moments(contours_1[i], false);
			obj_contours_1.push_back(obj);
		}
		cout << "左图检测轮廓个数：" << contours_1.size() << endl;
		


		//waitKey();
		vector<vector<Point>> contours_2;
		vector<obj_contours> obj_contours_2;
		contours_2 = getContours(rectifyImageR, 300, 10000, 0.5*canny_2, canny_2);
		for (int i = 0; i < contours_2.size(); i++)
		{
			obj_contours obj;
			obj.contours = contours_2[i];
			obj.corner=contours_2[i];
			obj.mu = moments(contours_2[i], false);//不变矩
			obj_contours_2.push_back(obj);
		}
		cout << "右图检测轮廓个数：" << contours_2.size() << endl;
		

		//drawContours(copy_1, obj_contours_1[0].contours, 0, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());

		for (int i = 0; i< obj_contours_1.size(); i++)
		{
			for each (Point var in obj_contours_1[i].corner)
			{
			//	circle(copy_L, var, 3, Scalar(255, 0, 0));//绘制角点
			}
			Point point_p(obj_contours_1[i].mu.m10 / obj_contours_1[i].mu.m00, obj_contours_1[i].mu.m01 / obj_contours_1[i].mu.m00);
			//circle(copy_L, Point2f(point_p), 2, Scalar(0, 0, 255));
			
			//char str[25];
			//_itoa(point_p.y, str, 10);
			//putText(copy_L, str, Point2f(obj_contours_1[i].mu.m10 / obj_contours_1[i].mu.m00, obj_contours_1[i].mu.m01 / obj_contours_1[i].mu.m00 + 10), 1, 1, Scalar(0, 255, 0));
			vector<vector<Point>> contours;
			contours.push_back(obj_contours_1[i].contours);
			//drawContours(copy_L, contours, 0, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());
			
		}

		for (int i = 0; i < obj_contours_2.size(); i++)
		{
			for each (Point var in obj_contours_2[i].corner)
			{
			//	circle(copy_R, var, 3, Scalar(255, 0, 0));
			}
			Point point_p(obj_contours_2[i].mu.m10 / obj_contours_2[i].mu.m00, obj_contours_2[i].mu.m01 / obj_contours_2[i].mu.m00);
			//circle(copy_R, point_p, 2, Scalar(0, 0, 255));
			//char str[25];
			//_itoa(point_p.y, str, 10);

			//putText(copy_R, str, Point2f(obj_contours_2[i].mu.m10 / obj_contours_2[i].mu.m00, obj_contours_2[i].mu.m01 / obj_contours_2[i].mu.m00 + 10), 1, 1, Scalar(0, 255, 0));
			vector<vector<Point>> contours;
			contours.push_back(obj_contours_2[i].contours);
			//drawContours(copy_R, contours, 0, Scalar(255, 255, 255), 1, 8, vector<Vec4i>(), 0, Point());		
		}
		/////////////////////////////////////
		vector<Scalar> my_color;
		my_color.push_back(Scalar(0, 0, 0));
		my_color.push_back(Scalar(255, 0, 0));
		my_color.push_back(Scalar(0, 255, 0));
		my_color.push_back(Scalar(0, 0, 255));
		vector<match_shape> M_matchshape;
		int shape_num = 0;
		for (int i = 0; i < obj_contours_1.size(); i++)
		{
			double min_match = 2;
			int index_match = 0;
			bool match_flag = false;
			for (int j = 0; j < obj_contours_2.size(); )
			{
				double match_value = matchShapes(obj_contours_1[i].contours, obj_contours_2[j].contours, CV_CONTOURS_MATCH_I3, 0.0);
				
				if ((match_value < 0.15)&&(match_value<min_match))
				{
					min_match = match_value;
					index_match = j;
					match_flag = true;
					j++;
				}
				else
				{
					j++;
				}
				
			}
			if (match_flag)
			{
				match_shape m_matchshape;
				m_matchshape.shape_L = obj_contours_1[i];
				m_matchshape.shape_R = obj_contours_2[index_match];
				M_matchshape.push_back(m_matchshape);
				obj_contours_2.erase(obj_contours_2.begin() + index_match);
			}

		}

		for (int i = 0; i < M_matchshape.size(); i++)
		{
			double L_y = M_matchshape[i].shape_L.mu.m01 / M_matchshape[i].shape_L.mu.m00;
			double R_y = M_matchshape[i].shape_R.mu.m01 / M_matchshape[i].shape_R.mu.m00;
			double L_x = M_matchshape[i].shape_L.mu.m10 / M_matchshape[i].shape_L.mu.m00;
			double R_x = M_matchshape[i].shape_R.mu.m10 / M_matchshape[i].shape_R.mu.m00;
			char str_L[25],str_R[25];
			_itoa(L_y, str_L, 10);
			_itoa(R_y, str_R, 10);
			string match = "macthed";
			circle(copy_L, Point(L_x, L_y),3,my_color[i]);
			circle(copy_R, Point(R_x, R_y), 3, my_color[i]);
			//putText(copy_L, str_L, Point(L_x, L_y),3,1,Scalar(0,0,255));
			//putText(copy_R, str_R, Point(R_x, R_y), 3, 1, Scalar(0, 0, 255));
			putText(copy_L, match, Point(L_x, L_y-20), 3, 1, my_color[i]);
			putText(copy_R, match, Point(R_x, R_y-20), 3, 1, my_color[i]);
			vector<vector<Point>> contours_L;
			contours_L.push_back(M_matchshape[i].shape_L.contours);
			vector<vector<Point>> contours_R;
			contours_R.push_back(M_matchshape[i].shape_R.contours);
			drawContours(copy_L, contours_L, 0, my_color[i], 1, 8, vector<Vec4i>(), 0, Point());
			drawContours(copy_R, contours_R, 0, my_color[i], 1, 8, vector<Vec4i>(), 0, Point());

		}


		//waitKey(); 


		
		for (int n = 0; n < M_matchshape.size();n++)
		{
			if (M_matchshape[n].shape_L.corner.size() == M_matchshape[n].shape_R.corner.size())
			{
				//质心x坐标距离
				double d_x = (M_matchshape[n].shape_L.mu.m10 / M_matchshape[n].shape_L.mu.m00) -
					(M_matchshape[n].shape_R.mu.m10 / M_matchshape[n].shape_R.mu.m00);
				for (int i = 0; i < M_matchshape[n].shape_L.corner.size(); i++)
				{
					for (int j = 0; j < M_matchshape[n].shape_R.corner.size(); j++)
					{
						if (Point_sameleval(M_matchshape[n].shape_L.corner[i], M_matchshape[n].shape_R.corner[j], d_x))
						{
							//M_matchshape[n].shape_R.corner.swap(M_matchshape[n].shape_R.corner.begin()+i, );
							
							M_matchshape[n].shape_L.corner_selected.push_back(Vec3d(M_matchshape[n].shape_L.corner[i].x, M_matchshape[n].shape_L.corner[i].y, M_matchshape[n].shape_L.corner[i].x - M_matchshape[n].shape_R.corner[j].x));
							//swap(M_matchshape[n].shape_R.corner[i], M_matchshape[n].shape_R.corner[j]);
							break;
						}
					}
				}

			}
		}

		/////////////////////
		for (int i = 0; i<M_matchshape.size();i++)
		{
			for (int j=0;j<M_matchshape[i].shape_L.corner_selected.size();j++)
			{
				Mat xyz = (Mat_<double>(4, 1) << M_matchshape[i].shape_L.corner_selected[j][0],
					M_matchshape[i].shape_L.corner_selected[j][1],
					M_matchshape[i].shape_L.corner_selected[j][2],
					1);
				Mat xyz_world  = Q*xyz;
				double x_world = xyz_world.at<double>(0, 0);
				double y_world = xyz_world.at<double>(1, 0);
				double z_world = xyz_world.at<double>(2, 0);
				double w	   = xyz_world.at<double>(3, 0);
				M_matchshape[i].shape_L.corner_world.push_back(Point3d(x_world / w, y_world / w, z_world / w));
			}
		}

		////////////////////////////////////////////

		for (int n=0;n<M_matchshape.size();n++)
		{
			for (int i = 0; i < M_matchshape[n].shape_L.corner_world.size(); i++)
			{
				for (int j = i + 1; j < M_matchshape[n].shape_L.corner_world.size(); j++)
				{
					if (abs(pointPolygonTest(M_matchshape[n].shape_L.contours,
						mid_point(Point(M_matchshape[n].shape_L.corner_selected[i][0], M_matchshape[n].shape_L.corner_selected[i][1]), Point(M_matchshape[n].shape_L.corner_selected[j][0], M_matchshape[n].shape_L.corner_selected[j][1])), true))
						 < 5)
					{
						double length = real_length(M_matchshape[n].shape_L.corner_world[i], M_matchshape[n].shape_L.corner_world[j]);
						
						char str[25];
						_itoa(length, str, 10);
						string show_str = "mm";
						show_str = str + show_str;
						putText(copy_L, show_str, mid_point(M_matchshape[n].shape_L.corner[i], M_matchshape[n].shape_L.corner[j]), 1, 1, Scalar(0, 0, 255));
					
					}
				}
			}
		}
		
		
		imshow("frame_L", copy_L);
		imshow("frame_R", copy_R);
		//waitKey();
		if (27 == waitKey(30))
		{
			return -1;
		}
		//namedWindow("disparity", CV_WINDOW_AUTOSIZE);


		//waitKey(30);
	}
return 0;
}

//save left,right camera picture 
/*
int main()
{
	VideoCapture cam1;
	VideoCapture cam2;
	int count = 0;
	string l = "left_test";
	string r = "right_test";
	string format = ".jpg";
	
	
	cam1.open(2);
	cam2.open(1);
		
	Mat frameL, frameR;
	if (!cam1.isOpened())// || !cam2.isOpened())
		cout << "cam1 err";
	if (!cam2.isOpened())// || !cam2.isOpened())
		cout << "cam2 err";
	while (1)
	{
		cam2.read(frameL);
		cam1.read(frameR);
		moveWindow("frameL", 0, 0);
		moveWindow("frameR", frameL.cols, 0);
		imshow("frameR", frameL);
		imshow("frameL", frameR);
		int c=waitKey(30);
		if (32 == c)//检测按键“空格”保存图片
		{
			char str[25];
			_itoa(count, str, 10);
			imwrite(r + str + format, frameR);
			imwrite(l + str + format, frameL);
			count++;
			cout <<"保存"<< count <<format<< endl;
		}
		else if(27 == c)//检测按键“esc”退出程序
		{
			break;
		}

	}
}
*/
