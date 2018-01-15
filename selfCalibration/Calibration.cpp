#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
using namespace std;
using namespace cv;
// ----------------------------------------------------------------
// 定义宏函数
// ----------------------------------------------------------------
#define COUT_VAR(x) cout << #x"=" << x << endl;
#define SHOW_IMG(x) namedWindow(#x);imshow(#x,x);waitKey(20);
// ----------------------------------------------------------------
// 定义对数函数
// ----------------------------------------------------------------
double log2( double n )  
{   
	return log( n ) / log( 2.0 );  
}
//----------------------------------------------------------
// Frobenius
//----------------------------------------------------------
double FNorm(Mat& M)
{
return (sqrt(sum((M*M.t()).diag())[0]));
}
// ----------------------------------------------------------------
// 图像转换函数
// ----------------------------------------------------------------
void transformImg(Mat& img,Mat& Tau,Mat& dst)
{	
	double cx=Tau.at<double>(0);
	double cy=Tau.at<double>(1);
	double fx=Tau.at<double>(2);
	double fy=Tau.at<double>(3);

	Mat cameraMatrix=(Mat_<double>(3, 3) << 
		fx, 0, cx,
		0,  fy, cy,
		0,  0, 1);

	Mat distCoeffs(4,1,CV_64FC1);

	distCoeffs.at<double>(0)=Tau.at<double>(4);
	distCoeffs.at<double>(1)=Tau.at<double>(5);
	distCoeffs.at<double>(2)=Tau.at<double>(6);
	distCoeffs.at<double>(3)=Tau.at<double>(7);

	cv::undistort(img,dst,cameraMatrix,distCoeffs);
}

double getF(Mat& img)
{
	Mat tmp=img/norm(img);
	Mat A, w, u, vt;
	SVD::compute(img, w, u, vt,SVD::NO_UV);
	return sum(w)[0];
}

void getJacobian(Mat& img,Mat& Tau,Mat& J_vec)
{
	int p=8;
	double epsilon[8]={1, // x
					   1, // y
					   1, // fx
					   1, // fy
					   2e-1, // k1
					   2e-1, // k2
					   5e-1, // k3
					   5e-1  // k4
	};
	Mat dst;
	Mat Ip,Im,Tau_p,Tau_m,diff;
	int n=img.cols;
	int m=img.rows;
	J_vec=Mat::zeros(n*m,p,CV_64FC1);
	for(int i=0;i<p;i++)
	{
		Tau_p=Tau.clone();
		Tau_m=Tau.clone();
		Tau_p.at<double>(i)+=epsilon[i];
		Tau_m.at<double>(i)-=epsilon[i];
		transformImg(img,Tau_p,Ip);
		transformImg(img,Tau_m,Im);	
		diff=Ip-Im;
		cv::GaussianBlur(diff,diff,Size(3,3),2);
		
		diff/=2*epsilon[i];
		diff=diff.reshape(1,m*n);		
		
		diff.copyTo(J_vec.col(i));
	}
}

//--------------------------------
// 符号函数
//--------------------------------
double sign(double X)
{
	double res=0;
	if(X>0){res=1;}
	if(X<0){res=-1;}
	return res;
}
//--------------------------------
double S_mu(double d,double mu)
{
	double res;
	res=sign(d)*MAX(fabs(d)-mu,0);
	return res;
}
//----------------------------------------------------------
//----------------------------------------------------------
Mat Smu(Mat& x,double mu)
{
	Mat res(x.size(),CV_64FC1);
	for(int i=0;i<x.rows;i++)
	{
		for(int j=0;j<x.cols;j++)
		{
			res.at<double>(i,j)=S_mu(x.at<double>(i,j),mu);
		}
	}
	return res;
}

Mat ALM(Mat& I,Mat& Tau_0,double Lambda)
{
	int maxcicles=100;
	int cicle_counter1=0;
	int cicle_counter2=0;
	bool converged1=0;
	bool converged2=0;
	Mat I_tau;
	Mat Tau=Tau_0.clone();
	Mat deltaTau_prev;
	Mat deltaTau;

	double F_prev=DBL_MAX;

	int n=I.cols;
	int m=I.rows;
	int p=8;

	double mu;
	double rho=1.5; 
	Mat J_vec;

	while(!converged1)
	{
		transformImg(I,Tau,I_tau);
		getJacobian(I,Tau,J_vec);

		Mat E=Mat::zeros(m, n,CV_64FC1);
		Mat A=Mat::zeros(m, n,CV_64FC1);

		mu=1.25/norm(I_tau);


		deltaTau=Mat::zeros(p, 1,CV_64FC1);
		deltaTau_prev=Mat::zeros(p, 1,CV_64FC1);

		Mat Y=I_tau.clone();
		Mat I0;

		cicle_counter2=0;
		converged2=0;

		Mat J_vec_inv=J_vec.inv(DECOMP_SVD);

		while(!converged2)
		{
			Mat t1=J_vec*deltaTau;
			Mat tmp;
			Mat U, Sigma,V;
			SVD::compute(I_tau+t1.reshape(1,m)-E+Y/mu,Sigma,U,V);
			Sigma=Smu(Sigma,1.0/mu);

			Mat W=Mat::zeros(Sigma.rows,Sigma.rows,CV_64FC1);
			Sigma.copyTo(W.diag());

			I0=U*W*V;
			tmp=I_tau+t1.reshape(1,m)-I0+Y/mu;
			E=Smu(tmp,Lambda/mu);

			tmp=(-I_tau+I0+E-Y/mu);
			deltaTau=J_vec_inv*tmp.reshape(1,m*n);

			tmp=J_vec*deltaTau;			
			Y+=mu*(I_tau+tmp.reshape(1,m)-I0-E);

			mu*=rho;

			cicle_counter2++;
			if(cicle_counter2>maxcicles){break;}

			double m,M;
			cv::minMaxLoc(abs(deltaTau_prev-deltaTau),&m,&M);	
			if(cicle_counter2>1 && M<1e-3)
			{
				converged2=true;
			}
			deltaTau_prev=deltaTau.clone();
		}
		for(int i=0;i<p;i++)
		{
			Tau.at<double>(i)+=deltaTau.at<double>(i);
		}

		Mat dst;
		transformImg(I,Tau,dst);

		double F=getF(dst);
		double perf=F_prev-F;
		F_prev=F;
		COUT_VAR(F);

		// ----------------------------------
		dst.convertTo(dst,CV_8UC1,255);
		imshow("矫正图",dst);
		cvWaitKey(10);
		// ----------------------------------

		if(perf<=0){converged1=true;}

		cicle_counter1++;
		if(cicle_counter1>maxcicles){break;}

	}
	return Tau;
}
//----------------------------------------------------------
//----------------------------------------------------------
vector<Point2f> pt;
void pp_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if(event == CV_EVENT_LBUTTONDOWN)
	{
		pt.push_back(Point2f(x,y));
	}
}
//----------------------------------------------------------
// 绘制区域
//----------------------------------------------------------
void DrawRegion(Mat& img,vector<Point2f>& region,Scalar color)
{
	for(int i = 0; i < 4; i++ )
	{
		line(img, region[i], region[(i+1)%4], color, 1, CV_AA);
	}
}

//----------------------------------------------------------
// 主函数
//----------------------------------------------------------
int main(int argc, char* argv[])
{
	// 定义原始图像及处理图像
	Mat img_c,img;
	// 读取原始图像
	img_c=imread("D:/opencv/pic/test2.jpg",1);
	resize(img_c,img_c,Size(200,200));
	//显示原始图像
	imshow("原始图", img_c);
	waitKey(10);

	cv::cvtColor(img_c,img,cv::COLOR_BGR2GRAY);
	img.convertTo(img,CV_64FC1,1.0/255.0);

	double theta_opt=0;
	double t_opt=0;
	Mat dst;

	Mat Tau_0(8,1,CV_64FC1);
	Tau_0.at<double>(0)=img.cols/2;
	Tau_0.at<double>(1)=img.rows/2;
	Tau_0.at<double>(2)=img.cols;
	Tau_0.at<double>(3)=img.rows;
	Tau_0.at<double>(4)=0;
	Tau_0.at<double>(5)=0;
	Tau_0.at<double>(6)=0;
	Tau_0.at<double>(7)=0;

	Mat Tau=ALM(img,Tau_0,0.5);
	//imshow("规范化", img_c);
	//  cx,cy,fx,fy,k1,k2,p1,p2
	COUT_VAR(Tau);
	cvWaitKey(0);
	destroyAllWindows();
	return 0;
}
