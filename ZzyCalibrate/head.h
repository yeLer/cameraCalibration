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
	//��������
	std::vector < std::vector<cv::Point3f >> objectPoints;
	//ͼ������
	std::vector <std::vector<cv::Point2f>> imagePoints;
	//�������
	cv::Mat camerMatirx;
	cv::Mat disCoeffs;
	//���
	int flag;
	//ȥ�������
	cv::Mat map1, map2;
	//�Ƿ�ȥ����
	bool mustInitUndistort;

	///���������
	void addPoints(const std::vector<cv::Point2f>&imageConers, const std::vector<cv::Point3f>&objectConers)
	{
		imagePoints.push_back(imageConers);
		objectPoints.push_back(objectConers);
	}
public:
	CameraCalibrator() :flag(0), mustInitUndistort(true) {}
	//������ͼƬ����ȡ�ǵ�
	int addChessboardPoints(const std::vector<std::string>&filelist, cv::Size &boardSize)
	{
		std::vector<cv::Point2f>imageConers;
		std::vector<cv::Point3f>objectConers;
		//����ǵ����������
		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				objectConers.push_back(cv::Point3f(i, j, 0.0f));
			}
		}
		//����ǵ���ͼ���е�����
		cv::Mat image;
		int success = 0;
		for (int i = 0; i < filelist.size(); i++)
		{
			image = cv::imread(filelist[i], 0);
			//�ҵ��ǵ�����
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
			//�����ǵ�
			cv::drawChessboardCorners(image, boardSize, imageConers, found);
			cv::imshow("Corners on Chessboard", image);
			cv::waitKey(100);
		}
		return success;
	}

	//����궨
	double calibrate(cv::Size&imageSize)
	{
		mustInitUndistort = true;
		std::vector<cv::Mat>rvecs, tvecs;
		//����궨
		return cv::calibrateCamera(objectPoints, imagePoints, imageSize,
			camerMatirx, disCoeffs, rvecs, tvecs, flag);
	}
	///ȥ����
	cv::Mat remap(const cv::Mat &image)
	{
		cv::Mat undistorted;
		if (mustInitUndistort)
		{
			//����������
			cv::initUndistortRectifyMap(camerMatirx, disCoeffs,
				cv::Mat(), cv::Mat(), image.size(), CV_32FC1, map1, map2);
			mustInitUndistort = false;
		}
		//Ӧ��ӳ�亯��
		cv::remap(image, undistorted, map1, map2, cv::INTER_LINEAR);
		return undistorted;
	}
	//����Ա�������������ڲ�������ͶӰ��������
	cv::Mat getCameraMatrix() const { return camerMatirx; }
	cv::Mat getDistCoeffs()   const { return disCoeffs; }
};