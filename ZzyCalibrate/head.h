#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<string>
#include<vector>
class CameraCalibrator
{
private:
	//世界坐标
	std::vector < std::vector<cv::Point3f >> objectPoints;
	//图像坐标
	std::vector <std::vector<cv::Point2f>> imagePoints;
	//输出矩阵
	cv::Mat camerMatirx;
	cv::Mat disCoeffs;
	//标记
	int flag;
	//去畸变参数
	cv::Mat map1, map2;
	//是否去畸变
	bool mustInitUndistort;

	///保存点数据
	void addPoints(const std::vector<cv::Point2f>&imageConers, const std::vector<cv::Point3f>&objectConers)
	{
		imagePoints.push_back(imageConers);
		objectPoints.push_back(objectConers);
	}
public:
	CameraCalibrator() :flag(0), mustInitUndistort(true) {}
	//打开棋盘图片，提取角点
	int addChessboardPoints(const std::vector<std::string>&filelist, cv::Size &boardSize)
	{
		std::vector<cv::Point2f>imageConers;
		std::vector<cv::Point3f>objectConers;
		//输入角点的世界坐标
		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				objectConers.push_back(cv::Point3f(i, j, 0.0f));
			}
		}
		//计算角点在图像中的坐标
		cv::Mat image;
		int success = 0;
		for (int i = 0; i < filelist.size(); i++)
		{
			image = cv::imread(filelist[i], 0);
			//找到角点坐标
			bool found = cv::findChessboardCorners(image, boardSize, imageConers);
			cv::cornerSubPix(image,
				imageConers,
				cv::Size(5, 5),
				cv::Size(-1, -1),
				cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
					30, 0.1));
			if (imageConers.size() == boardSize.area())
			{
				addPoints(imageConers, objectConers);
				success++;
			}
			//画出角点
			cv::drawChessboardCorners(image, boardSize, imageConers, found);
			cv::imshow("Corners on Chessboard", image);
			cv::waitKey(100);
		}
		return success;
	}

	//相机标定
	double calibrate(cv::Size&imageSize)
	{
		mustInitUndistort = true;
		std::vector<cv::Mat>rvecs, tvecs;
		//相机标定
		return cv::calibrateCamera(objectPoints, imagePoints, imageSize,
			camerMatirx, disCoeffs, rvecs, tvecs, flag);
	}
	///去畸变
	cv::Mat remap(const cv::Mat &image)
	{
		cv::Mat undistorted;
		if (mustInitUndistort)
		{
			//计算畸变参数
			cv::initUndistortRectifyMap(camerMatirx, disCoeffs,
				cv::Mat(), cv::Mat(), image.size(), CV_32FC1, map1, map2);
			mustInitUndistort = false;
		}
		//应用映射函数
		cv::remap(image, undistorted, map1, map2, cv::INTER_LINEAR);
		return undistorted;
	}
	//常成员函数，获得相机内参数矩阵、投影矩阵数据
	cv::Mat getCameraMatrix() const { return camerMatirx; }
	cv::Mat getDistCoeffs()   const { return disCoeffs; }
};