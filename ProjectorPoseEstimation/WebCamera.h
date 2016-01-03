#ifndef WEBCAMERA_H
#define WEBCAMERA_H

#define LEAST_CAP (5) //キャリブレーションに必要な画像数

#include <opencv2\opencv.hpp>

class WebCamera
{
public:
	//解像度
	int width;
	int height;
	//保存関連
	int capture_num;
	std::string save_dir;

	cv::VideoCapture vc;
	cv::Mat frame;
	std::string winName;

	/***** キャリブレーション関係 *****/
	bool calib_flag;
	cv::Size checkerPattern;		// チェッカーパターンの交点の数
	float checkerSize;				// チェッカーパターンのマス目のサイズ(mm)

	std::vector<cv::Point3f> worldPoint;		// チェッカー交点座標と対応する世界座標の値を格納する行列

	// カメラ
	cv::Mat cam_K;					// 内部パラメータ行列
	cv::Mat cam_dist;				// レンズ歪み
	std::vector<cv::Mat> cam_R;		// 回転ベクトル
	std::vector<cv::Mat> cam_T;		// 平行移動ベクトル

	WebCamera(){};

	WebCamera(int _width, int _height, std::string _winName)
	{
		width = _width;
		height = _height;
		winName = _winName;
		vc = cv::VideoCapture(1);
		save_dir = "./capture/";
		capture_num = getStoredImage(save_dir);
		calib_flag = false;
		cv::Size captureSize(width, height);
		vc.set(CV_CAP_PROP_FRAME_WIDTH, captureSize.width);
		vc.set(CV_CAP_PROP_FRAME_HEIGHT, captureSize.height);
		cv::namedWindow(winName);


	};

	int getStoredImage(std::string dir)
	{
		HANDLE hFind;
		WIN32_FIND_DATA win32fd;
		int counter= 0;

		std::string target = dir + "*.jpg";
		hFind = FindFirstFile(target.data(), &win32fd);

		if (hFind == INVALID_HANDLE_VALUE) {
		}

		do {
			if (win32fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			} else {
				counter++;
				printf("%s\n", win32fd.cFileName);
			}
		} while (FindNextFile(hFind, &win32fd));

		FindClose(hFind);

		printf("counter: %i\n" , counter);
		return counter;
	}

	void idle(){
		vc >> frame;
		cv::imshow(winName, frame);
	}

	cv::Mat getFrame()
	{
		vc >> frame;
		return frame;
	}

	void capture()
	{
		std::string filename = save_dir + "cap"+ std::to_string(capture_num) + ".jpg";
		cv::imwrite(filename, frame);
		capture_num++;
	}

	/***** キャリブレーション関係 *****/
	void initCalibration(int _checkerRow, int _checkerCol, float _checkerSize)
	{
		checkerPattern = cv::Size(_checkerRow, _checkerCol);
		checkerSize = _checkerSize;
		// 世界座標におけるチェッカーパターンの交点座標を決定
		for( int i = 0; i < checkerPattern.area(); ++i ) {
			worldPoint.push_back( cv::Point3f(	static_cast<float>( i % checkerPattern.width * checkerSize ),
													static_cast<float>( i / checkerPattern.width * checkerSize ), 0.0 ) );
		}
	}

	void getCheckerCorners(std::vector<cv::Point2f> &imagePoint, const cv::Mat &image, cv::Mat &draw_image)
	{
		//交点検出
		bool detect = cv::findChessboardCorners(image, checkerPattern, imagePoint);

		//検出点の描画
		image.copyTo(draw_image);
		if(detect)
		{
			//サブピクセル精度
			cv::Mat gray;
			cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
			cv::cornerSubPix( gray, imagePoint, cv::Size( 11, 11 ), cv::Size( -1, -1 ), cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.001 ) );

			cv::drawChessboardCorners(draw_image, checkerPattern, imagePoint, true);
		}else
		{
			cv::drawChessboardCorners(draw_image, checkerPattern, imagePoint, false);
		}
	}

	void cameraCalibration()
	{
		if(!calib_flag)
		{
			if(capture_num < LEAST_CAP)
			{

				printf("画像数が足りません\n5枚以上撮影してください\n");
			}else
			{
				std::vector<std::vector<cv::Point3f>>	worldPoints; 
				std::vector<std::vector<cv::Point2f>>	cameraPoints;
				//①画像からチェッカ交点を取得
				for(int i = 1; i <= capture_num; i++)
				{
					std::vector<cv::Point2f> imagePoint;
					std::string filename = save_dir + "cap"+ std::to_string(i) + ".jpg";
					cv::Mat  image = cv::imread(filename);				//画像読み込み
					cv::Mat draw_image;											//描画用
					getCheckerCorners(imagePoint, image, draw_image);
					// 追加
					worldPoints.emplace_back(worldPoint);
					cameraPoints.emplace_back(imagePoint);
				}
				//②キャリブレーション実行
				double cam_error = cv::calibrateCamera(worldPoints, cameraPoints, cv::Size(width, height), cam_K, cam_dist, cam_R, cam_T, cv::CALIB_FIX_K3, 
																			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, DBL_EPSILON));
				//③結果の保存
				cv::FileStorage fs("Camera.xml", cv::FileStorage::WRITE);
				fs << "cam_K" << cam_K << "cam_dist" << cam_dist
				<< "cam_R0" << cam_R.at(0) << "cam_T0" << cam_T.at(0);
				fs.release();

				// 結果の表示
				std::cout << "***** Calibration results *****" << std::endl << std::endl;
				std::cout	<< "Camera Calibration results:" << std::endl
					<< " - Reprojection error: " << cam_error << std::endl
					<< " - K:\n" << cam_K << std::endl
					<< " - Distortion:" << cam_dist << std::endl 
					<< " - R0\n:" << cam_R.at(0) << std::endl
					<< " - T0:" << cam_T.at(0) << std::endl << std::endl; 

				calib_flag = true;
			}
		}else
		{
			printf("キャリブレーション済みです\n");
		}
	}

// キャリブレーション結果の読み込み(内部パラメータのみ)
void loadCalibParam(const std::string &fileName)
{
	// xmlファイルの読み込み
	cv::FileStorage cvfs(fileName, cv::FileStorage::READ);

	cvfs["cam_K"] >> cam_K;
	cvfs["cam_dist"] >> cam_dist;

	calib_flag = true;
}

	~WebCamera(){};
};
#endif