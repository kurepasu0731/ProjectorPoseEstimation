#ifndef PROJECTORESTIMATION_H
#define PROJECTORESTIMATION_H

#include <opencv2\opencv.hpp>
#include "WebCamera.h"

#include <Eigen/Dense>
#include "unsupported/Eigen/NonLinearOptimization"
#include "unsupported/Eigen/NumericalDiff"


using namespace cv;
using namespace Eigen;


class ProjectorEstimation
{
public:
	WebCamera camera;
	WebCamera projector;
	cv::Size checkerPattern;

	std::vector<cv::Point2f> projectorImageCorners; //プロジェクタ画像上の対応点座標
	std::vector<cv::Point2f> cameraImageCorners; //カメラ画像上の対応点座標

	//カメラ画像上のコーナー点
	std::vector<cv::Point2f> camcorners;
	//プロジェクタ画像上のコーナー点
	std::vector<cv::Point2f> projcorners;

	//3次元点(カメラ中心)LookUpテーブル
	//** index = カメラ画素(左上始まり)
	//** int image_x = i % CAMERA_WIDTH;
	//** int image_y = (int)(i / CAMERA_WIDTH);
	//** Point3f = カメラ画素の3次元座標(計測されていない場合は(-1, -1, -1))
	std::vector<cv::Point3f> reconstructPoints;

	//3 * 4形式ののプロジェクタ内部行列
	Mat projK;

	//動きベクトル
	Mat dR, dt;


	//コンストラクタ
	ProjectorEstimation(WebCamera _camera, WebCamera _projector, int _checkerRow, int _checkerCol, int _blockSize, cv::Size offset) //よこ×たて
	{
		camera = _camera;
		projector = _projector;
		checkerPattern = cv::Size(_checkerRow, _checkerCol);

		//後で使うプロジェクタの内部行列
		projK = (cv::Mat_<double>(3, 4) << projector.cam_K.at<double>(0,0),projector.cam_K.at<double>(0,1), projector.cam_K.at<double>(0,2), 0,
						            projector.cam_K.at<double>(1,0), projector.cam_K.at<double>(1,1), projector.cam_K.at<double>(1,2), 0,
									projector.cam_K.at<double>(2,0), projector.cam_K.at<double>(2,1), projector.cam_K.at<double>(2,2), 0);
		//動きベクトル用
		//一個前の推定結果と現推定結果の差分
		dR = cv::Mat::zeros(3,1,CV_64F);
		dt = cv::Mat::zeros(3,1,CV_64F);

		//プロジェクタ画像上の対応点初期化
		getProjectorImageCorners(projectorImageCorners, _checkerRow, _checkerCol, _blockSize, offset);
	};

	~ProjectorEstimation(){};

	//3次元復元結果読み込み
	void loadReconstructFile(const string& filename)
	{
		//3次元点(カメラ中心)LookUpテーブルのロード
		FileStorage fs(filename, FileStorage::READ);
		FileNode node(fs.fs, NULL);

		read(node["points"], reconstructPoints);

		std::cout << "3次元復元結果読み込み." << std::endl;
	}

	//コーナー検出によるプロジェクタ位置姿勢を推定
	bool findProjectorPose_Corner(const cv::Mat& camframe, const cv::Mat projframe, cv::Mat& initialR, cv::Mat& initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &draw_camimage, cv::Mat &draw_projimage)
	{
		//draw用(カメラ)
		draw_camimage = camframe.clone();

		//カメラ画像上のコーナー検出
		bool detect_cam = getCorners(camframe, camcorners, draw_camimage);
		//プロジェクタ画像上のコーナー検出
		bool detect_proj = getCorners(projframe, projcorners, draw_projimage); //projcornersがdraw_projimage上でずれるのは、歪み除去してないから

		//コーナー検出できたら、位置推定開始
		if(detect_cam && detect_proj)
		{
			// 対応点の歪み除去
			std::vector<cv::Point2f> undistort_imagePoint;
			std::vector<cv::Point2f> undistort_projPoint;
			cv::undistortPoints(camcorners, undistort_imagePoint, camera.cam_K, camera.cam_dist);
			cv::undistortPoints(projcorners, undistort_projPoint, projector.cam_K, projector.cam_dist);
			for(int i=0; i<camcorners.size(); ++i)
			{
				undistort_imagePoint[i].x = undistort_imagePoint[i].x * camera.cam_K.at<double>(0,0) + camera.cam_K.at<double>(0,2);
				undistort_imagePoint[i].y = undistort_imagePoint[i].y * camera.cam_K.at<double>(1,1) + camera.cam_K.at<double>(1,2);
			}
			for(int i=0; i<projcorners.size(); ++i)
			{
				undistort_projPoint[i].x = undistort_projPoint[i].x * projector.cam_K.at<double>(0,0) + projector.cam_K.at<double>(0,2);
				undistort_projPoint[i].y = undistort_projPoint[i].y * projector.cam_K.at<double>(1,1) + projector.cam_K.at<double>(1,2);
			}

			int result = calcProjectorPose_Corner(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, draw_projimage);

			if(result > 0) return true;
			else return false;
		}
		else{
			return false;
		}
	}

	//計算部分
	int calcProjectorPose_Corner(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT,
								 cv::Mat &chessimage)
	{
		//回転行列から回転ベクトルにする
		Mat initRVec(3, 1,  CV_64F, Scalar::all(0));
		Rodrigues(initialR, initRVec);
		Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

		int n = 6; //変数の数
		int info;
		double level = 1.0;

		VectorXd initial(n);
		initial <<
			initRVec.at<double>(0, 0) + dR.at<double>(0, 0) * level,
			initRVec.at<double>(1, 0) + dR.at<double>(1, 0) * level,
			initRVec.at<double>(2, 0) + dR.at<double>(2, 0) * level,
			initTVec.at<double>(0, 0) + dt.at<double>(0, 0) * level,
			initTVec.at<double>(1, 0) + dt.at<double>(1, 0) * level,
			initTVec.at<double>(2, 0) + dt.at<double>(2, 0) * level;

		//3次元座標が取れた対応点のみを抽出してからLM法に入れる
		std::vector<cv::Point3f> reconstructPoints_valid;
		for(int i = 0; i < imagePoints.size(); i++)
		{
			int image_x = (int)(imagePoints[i].x+0.5);
			int image_y = (int)(imagePoints[i].y+0.5);
			int index = image_y * CAMERA_WIDTH + image_x;
			if(0 <= image_x && image_x <= CAMERA_WIDTH && 0 <= image_y && image_y <= CAMERA_HEIGHT && reconstructPoints[index].x != -1)
			{
				reconstructPoints_valid.emplace_back(reconstructPoints[index]);
			}
		}

		misra2a_functor functor(n, projPoints.size(), projPoints, reconstructPoints_valid, projector.cam_K);
    
		NumericalDiff<misra2a_functor> numDiff(functor);
		LevenbergMarquardt<NumericalDiff<misra2a_functor> > lm(numDiff);
		info = lm.minimize(initial);
    
		std::cout << "学習結果_2a: " << std::endl;
		std::cout <<
			initial[0] << " " <<
			initial[1] << " " <<
			initial[2] << " " <<
			initial[3] << " " <<
			initial[4] << " " <<
			initial[5]	 << std::endl;

		//出力
		Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
		cv::Rodrigues(dstRVec, dstR);
		dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
		Mat dstTVec = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);//保持用

		//対応点の様子を描画
		std::vector<cv::Point2f> pt;
		cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector.cam_K, cv::Mat(), pt); 
		for(int i = 0; i < projPoints.size(); i++)
		{
			cv::circle(chessimage, projPoints[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
		}
		for(int i = 0; i < pt.size(); i++)
		{
			cv::circle(chessimage, pt[i], 5, cv::Scalar(255, 0, 0), 3);//カメラは青
		}

		//重心も描画
		cv::Point2f imageWorldPointAve;
		cv::Point2f projAve;
		calcAveragePoint(reconstructPoints_valid, projPoints, dstRVec, dstTVec,imageWorldPointAve, projAve);
		cv::circle(chessimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//プロジェクタは赤
		cv::circle(chessimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//カメラは青

		//動きベクトル更新
		//dR = initRVec - dstRVec;
		//dt = initTVec - dstTVec;

		std::cout << "info: " << info << std::endl;
		return info;
	}

	//コーナー検出
	bool getCorners(cv::Mat frame, std::vector<cv::Point2f> &corners, cv::Mat &drawimage){
		cv::Mat gray_img;
		//歪み除去
		//cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
		//グレースケール
		cv::cvtColor(frame, gray_img, CV_BGR2GRAY);

		//コーナー検出
		int num = 150;
		cv::goodFeaturesToTrack(gray_img, corners, num, 0.001, 15);

		//描画
		for(int i = 0; i < corners.size(); i++)
		{
			cv::circle(drawimage, corners[i], 1, cv::Scalar(0, 0, 255), 3);
		}

		//コーナー検出ができたかどうか
		if(corners.size() > 0)	return true;
		else	return false;

	}


	//各対応点の重心位置を計算
	void calcAveragePoint(std::vector<cv::Point3f> imageWorldPoints, std::vector<cv::Point2f> projPoints, cv::Mat R, cv::Mat t, cv::Point2f& imageAve, cv::Point2f& projAve)
	{
		//各対応点のプロジェクタ画像上での重心を求める
		//(1)proj_p_
		float sum_px = 0, sum_py = 0, px = 0, py = 0;
		for(int i = 0; i < projPoints.size(); i++)
		{
			sum_px += projPoints[i].x;
			sum_py += projPoints[i].y;
		}
		px = sum_px / projPoints.size();
		py = sum_py / projPoints.size();

		projAve.x = px;
		projAve.y = py;

		//(2)worldPoints_
		// 2次元(プロジェクタ画像)平面へ投影
		std::vector<cv::Point2f> pt;
		cv::projectPoints(imageWorldPoints, R, t, projector.cam_K, cv::Mat(), pt); 
		float sum_wx = 0, sum_wy = 0, wx = 0, wy = 0;
		for(int i = 0; i < pt.size(); i++)
		{
			sum_wx += pt[i].x;
			sum_wy += pt[i].y;
		}
		wx = sum_wx / pt.size();
		wy = sum_wy / pt.size();

		imageAve.x = wx;
		imageAve.y = wy;
	}

//*************************************************************************************************************************//


	//チェッカボード検出によるプロジェクタ位置姿勢を推定
	bool findProjectorPose(cv::Mat frame, cv::Mat& initialR, cv::Mat& initialT, cv::Mat &dstR, cv::Mat &dstT, cv::Mat &draw_image, cv::Mat &chessimage){
		//cv::Mat undist_img1;
		////カメラ画像の歪み除去
		//cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
		//コーナー検出
		//getCheckerCorners(cameraImageCorners, undist_img1, draw_image);

		//チェッカパターン検出(カメラ画像は歪んだまま)
		bool detect = getCheckerCorners(cameraImageCorners, frame, draw_image);

		//検出できたら、位置推定開始
		if(detect)
		{
			// 対応点の歪み除去
			std::vector<cv::Point2f> undistort_imagePoint;
			std::vector<cv::Point2f> undistort_projPoint;
			cv::undistortPoints(cameraImageCorners, undistort_imagePoint, camera.cam_K, camera.cam_dist);
			cv::undistortPoints(projectorImageCorners, undistort_projPoint, projector.cam_K, projector.cam_dist);
			for(int i=0; i<cameraImageCorners.size(); ++i)
			{
				undistort_imagePoint[i].x = undistort_imagePoint[i].x * camera.cam_K.at<double>(0,0) + camera.cam_K.at<double>(0,2);
				undistort_imagePoint[i].y = undistort_imagePoint[i].y * camera.cam_K.at<double>(1,1) + camera.cam_K.at<double>(1,2);
				undistort_projPoint[i].x = undistort_projPoint[i].x * projector.cam_K.at<double>(0,0) + projector.cam_K.at<double>(0,2);
				undistort_projPoint[i].y = undistort_projPoint[i].y * projector.cam_K.at<double>(1,1) + projector.cam_K.at<double>(1,2);
			}

			int result = calcProjectorPose(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT, chessimage);
			if(result > 0) return true;
			else return false;
		}
		else{
			return false;
		}
	}

	//計算部分(Rの自由度3)
	int calcProjectorPose(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat& initialR, cv::Mat& initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat &chessimage)
	{
		//回転行列から回転ベクトルにする
		Mat initRVec(3, 1,  CV_64F, Scalar::all(0));
		Rodrigues(initialR, initRVec);
		Mat initTVec = (cv::Mat_<double>(3, 1) << initialT.at<double>(0, 0), initialT.at<double>(1, 0), initialT.at<double>(2, 0));

		int n = 6; //変数の数
		int info;
		double level = 1.0; //動きベクトルの大きさ

		VectorXd initial(n);
		initial <<
			initRVec.at<double>(0, 0) + dR.at<double>(0, 0) * level,
			initRVec.at<double>(1, 0) + dR.at<double>(1, 0) * level,
			initRVec.at<double>(2, 0) + dR.at<double>(2, 0) * level,
			initTVec.at<double>(0, 0) + dt.at<double>(0, 0) * level,
			initTVec.at<double>(1, 0) + dt.at<double>(1, 0) * level,
			initTVec.at<double>(2, 0) + dt.at<double>(2, 0) * level;

		//3次元座標が取れた対応点のみを抽出してからLM法に入れる
		std::vector<cv::Point3f> reconstructPoints_valid;
		std::vector<cv::Point2f> projPoints_valid;
		for(int i = 0; i < imagePoints.size(); i++)
		{
			int image_x = (int)(imagePoints[i].x+0.5);
			int image_y = (int)(imagePoints[i].y+0.5);
			int index = image_y * CAMERA_WIDTH + image_x;
			if(reconstructPoints[index].x != -1)
			{
				reconstructPoints_valid.emplace_back(reconstructPoints[index]);
				projPoints_valid.emplace_back(projPoints[i]);
			}
		}

		misra1a_functor functor(n, projPoints_valid.size(), projPoints_valid, reconstructPoints_valid, projector.cam_K);
    
		NumericalDiff<misra1a_functor> numDiff(functor);
		LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
		info = lm.minimize(initial); //info=2がかえってくる 時々5
    
		//std::cout << "学習結果: " << std::endl;
		//std::cout <<
		//	initial[0] << " " <<
		//	initial[1] << " " <<
		//	initial[2] << " " <<
		//	initial[3] << " " <<
		//	initial[4] << " " <<
		//	initial[5]	 << std::endl;

		//出力
		Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
		Rodrigues(dstRVec, dstR);
		dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
		Mat dstTVec = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);//保持用

		//対応点の様子を描画
		std::vector<cv::Point2f> pt;
		cv::projectPoints(reconstructPoints_valid, dstRVec, dstTVec, projector.cam_K, cv::Mat(), pt); 
		for(int i = 0; i < projPoints_valid.size(); i++)
		{
			cv::circle(chessimage, projPoints_valid[i], 5, cv::Scalar(0, 0, 255), 3); //プロジェクタは赤
			cv::circle(chessimage, pt[i], 5, cv::Scalar(255, 0, 0), 3);//カメラは青
		}
		//重心も描画
		cv::Point2f imageWorldPointAve;
		cv::Point2f projAve;
		calcAveragePoint(reconstructPoints_valid, projPoints_valid, dstRVec, dstTVec,imageWorldPointAve, projAve);
		cv::circle(chessimage, projAve, 8, cv::Scalar(0, 0, 255), 10);//プロジェクタは赤
		cv::circle(chessimage, imageWorldPointAve, 8, cv::Scalar(255, 0, 0), 10);//カメラは青

		//動きベクトル更新
		//dR = initRVec - dstRVec;
		//dt = initTVec - dstTVec;

		//std::cout << "-----\ndR: \n" << dR << std::endl;
		//std::cout << "dT: \n" << dt << std::endl;


		std::cout << "info: " << info << std::endl;
		return info;
	}

	//計算部分(Rの自由度9)
	//void calcProjectorPose2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
	//{
	//	int n = 12; //変数の数
	//	int info;		
	//	VectorXd initial(n);
	//	initial <<
	//		initialR.at<double>(0, 0),
	//		initialR.at<double>(0, 1),
	//		initialR.at<double>(0, 2),
	//		initialR.at<double>(1, 0),
	//		initialR.at<double>(1, 1),
	//		initialR.at<double>(1, 2),
	//		initialR.at<double>(2, 0),
	//		initialR.at<double>(2, 1),
	//		initialR.at<double>(2, 2),
	//		initialT.at<double>(0, 0),
	//		initialT.at<double>(1, 0),
	//		initialT.at<double>(2, 0);
	//	misra1a_functor functor(n, imagePoints.size(), projPoints, imagePoints, camera.cam_K, projector.cam_K, reconstructPoints);    
	//	NumericalDiff<misra1a_functor> numDiff(functor);
	//	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
	//	info = lm.minimize(initial);   
	//	std::cout << "学習結果: " << std::endl;
	//	std::cout <<
	//		initial[0] << " " <<
	//		initial[1] << " " <<
	//		initial[2] << " " <<
	//		initial[3] << " " <<
	//		initial[4] << " " <<
	//		initial[5] << " " <<
	//		initial[6] << " " <<
	//		initial[7] << " " <<
	//		initial[8] << " " <<
	//		initial[9] << " " <<
	//		initial[10] << " " <<
	//		initial[11] << " " << std::endl;
	//	//出力
	//	dstR = (cv::Mat_<double>(3, 3) << initial[0], initial[1], initial[2], initial[3], initial[4], initial[5], initial[6], initial[7], initial[8]);
	//	dstT = (cv::Mat_<double>(3, 1) << initial[9], initial[10], initial[11]);
	//}


	bool getCheckerCorners(std::vector<cv::Point2f>& imagePoint, const cv::Mat &image, cv::Mat &draw_image)
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

		return detect;
	}

	void getProjectorImageCorners(std::vector<cv::Point2f>& projPoint, int _row, int _col, int _blockSize, cv::Size _offset)
	{
		for (int y = 0; y < _col; y++)
		{
			for(int x = 0; x < _row; x++)
			{
				projPoint.push_back(cv::Point2f(_offset.width + x * _blockSize, _offset.height + y * _blockSize));
			}
		}
	}


//*****非線形最適化関数******************************************************************************************************//

	// Generic functor
	template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
	struct Functor
	{
	  typedef _Scalar Scalar;
	  enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	  };
	  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
	  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
	  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
	};

	struct misra1a_functor : Functor<double>
	{
		// 目的関数
		misra1a_functor(int inputs, int values, vector<Point2f>& proj_p, vector<Point3f>& world_p, const Mat& proj_K)
			: inputs_(inputs),
			  values_(values), 
			  proj_p_(proj_p),
			  worldPoints_(world_p),
			  projK(proj_K){}
			  //cam_p_(cam_p), 
			  //reconstructPoints_(reconstructPoints),
			  //cam_K_(cam_K), 
			  //proj_K_(proj_K),
			  //projK_inv_t(proj_K_.inv().t()), 
			  //camK_inv(cam_K.inv()) {}
    
		vector<Point2f> proj_p_;
		vector<cv::Point3f> worldPoints_;
		const Mat projK;

		//**エピポーラ方程式を用いた最適化**//

		//Rの自由度3にする
		//int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		//{
		//	//回転ベクトルから回転行列にする
		//	Mat rotateVec = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
		//	Mat R(3, 3, CV_64F, Scalar::all(0));
		//	Rodrigues(rotateVec, R);
		//	//[t]x
		//	Mat tx = (cv::Mat_<double>(3, 3) << 0, -_Rt[5], _Rt[4], _Rt[5], 0, -_Rt[3], -_Rt[4], _Rt[3], 0);
		//	for (int i = 0; i < values_; ++i) {
		//		Mat cp = (cv::Mat_<double>(3, 1) << (double)cam_p_.at(i).x,  (double)cam_p_.at(i).y,  1);
		//		Mat pp = (cv::Mat_<double>(3, 1) << (double)proj_p_.at(i).x,  (double)proj_p_.at(i).y,  1);
		//		Mat error = pp.t() * projK_inv_t * tx * R * camK_inv * cp;
		//		fvec[i] = error.at<double>(0, 0);
		//	}
		//	return 0;
		//}

		//Rの自由度を9にする
		//int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		//{
		//	Mat R = (cv::Mat_<double>(3, 3) << _Rt[0], _Rt[1], _Rt[2], _Rt[3], _Rt[4], _Rt[5], _Rt[6], _Rt[7], _Rt[8]);
		//	//[t]x
		//	Mat tx = (cv::Mat_<double>(3, 3) << 0, -_Rt[5], _Rt[4], _Rt[5], 0, -_Rt[3], -_Rt[4], _Rt[3], 0);
		//	for (int i = 0; i < values_; ++i) {
		//		Mat cp = (cv::Mat_<double>(3, 1) << (double)cam_p_.at(i).x,  (double)cam_p_.at(i).y,  1);
		//		Mat pp = (cv::Mat_<double>(3, 1) << (double)proj_p_.at(i).x,  (double)proj_p_.at(i).y,  1);
		//		Mat error = cp.t() * camK_inv_t * tx * R * proj_K_.inv() * pp;
		//		fvec[i] = error.at<double>(0, 0);
		//		//直に計算
		//		//fvec[i] = (double)cam_p_.at(i).x * (camK_inv_t.at<double>(0, 0) * (-_Rt[11] * (_Rt[3] * (projK_inv.at<double>(0, 0) * proj_p_.at(i).x + projK_inv.at<double>(0, 1) * proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(0, 1) * (_Rt[11] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(0, 2) * (-_Rt[10] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))))) +
		//		//		  (double)cam_p_.at(i).y * (camK_inv_t.at<double>(1, 0) * (-_Rt[11] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(1, 1) * (_Rt[11] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(1, 2) * (-_Rt[10] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))))) +
		//		//		                    camK_inv_t.at<double>(2, 0) * (-_Rt[11] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(2, 1) * (_Rt[11] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(2, 2) * (-_Rt[10] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))));
		//	}
		//	return 0;
		//}


		//**3次元復元結果を用いた最適化**//

		//Rの自由度3
		int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		{
			cv::Mat vr = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
			cv::Mat vt = (cv::Mat_<double>(3, 1) << _Rt[3], _Rt[4], _Rt[5]);
			cv::Mat R_33(3, 3, CV_64F, Scalar::all(0));
			Rodrigues(vr, R_33);

			// 2次元(プロジェクタ画像)平面へ投影
			std::vector<cv::Point2f> pt;
			cv::projectPoints(worldPoints_, R_33, vt, projK, cv::Mat(), pt);

			// 射影誤差算出
			for (int i = 0; i < values_; ++i) 
			{
				//Mat wp = (cv::Mat_<double>(4, 1) << worldPoints_[i].x, worldPoints_[i].y, worldPoints_[i].z, 1);
				////4*4行列にする
				//Mat Rt = (cv::Mat_<double>(4, 4) << R_33.at<double>(0,0), R_33.at<double>(0,1), R_33.at<double>(0,2), _Rt[3],
				//	                               R_33.at<double>(1,0), R_33.at<double>(1,1), R_33.at<double>(1,2), _Rt[4],
				//								   R_33.at<double>(2,0), R_33.at<double>(2,1), R_33.at<double>(2,2), _Rt[5],
				//								   0, 0, 0, 1);
				//Mat dst_p = projK * Rt * wp;
				//Point2f project_p(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
				// 射影誤差算出
				fvec[i] = pow(pt[i].x - proj_p_[i].x, 2) + pow(pt[i].y - proj_p_[i].y, 2);
			}
			return 0;
		}

		//Rの自由度9
		//int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		//{
		//	//// 2次元(プロジェクタ画像)平面へ投影
		//	//std::vector<cv::Point2f> pt;
		//	//cv::projectPoints(reconstructPoints_, R, t, proj_K_, cv::Mat(), pt); 
		//	// 射影誤差算出(有効な点のみ)
		//	for (int i = 0; i < values_; ++i) 
		//	{
		//		int image_x = (int)(cam_p_[i].x+0.5);
		//		int image_y = (int)(cam_p_[i].y+0.5);
		//		int index = image_y * CAMERA_WIDTH + image_x;
		//		if(reconstructPoints_[index].x != -1)
		//		{
		//			Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_[index].x, reconstructPoints_[index].y, reconstructPoints_[index].z, 1);
		//			//Mat vr = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
		//			//Mat R_33(3, 3, CV_64F, Scalar::all(0));
		//			//Rodrigues(vr, R_33);
		//			//4*4行列にする
		//			Mat Rt = (cv::Mat_<double>(4, 4) << _Rt[0],_Rt[1],_Rt[2], _Rt[9],
		//				                              _Rt[3], _Rt[4], _Rt[5], _Rt[10],
		//											   _Rt[6],_Rt[7],_Rt[8], _Rt[11],
		//											   0, 0, 0, 1);
		//			Mat projK = (cv::Mat_<double>(3, 4) << proj_K_.at<double>(0,0), proj_K_.at<double>(0,1), proj_K_.at<double>(0,2), 0,
		//				                               proj_K_.at<double>(1,0), proj_K_.at<double>(1,1), proj_K_.at<double>(1,2), 0,
		//											   proj_K_.at<double>(2,0), proj_K_.at<double>(2,1), proj_K_.at<double>(2,2), 0);
		//			//プロジェクタ画像上へ射影
		//			Mat dst_p = projK * Rt * wp;
		//			Point2f project_p(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));
		//			// 射影誤差算出
		//			fvec[i] = pow(project_p.x - proj_p_[i].x, 2) + pow(project_p.y - proj_p_[i].y, 2);
		//		}
		//		else
		//		{
		//			fvec[i] = 0;
		//		}
		//	}
		//	return 0;
		//}

		const int inputs_;
		const int values_;
		int inputs() const { return inputs_; }
		int values() const { return values_; }

	};

	struct misra2a_functor : Functor<double>
	{
		// 目的関数
		misra2a_functor(int inputs, int values, vector<Point2f>& proj_p, vector<Point3f>& world_p, const Mat& proj_K)
			: inputs_(inputs),
			  values_(values), 
			  proj_p_(proj_p),
			  worldPoints_(world_p),
			  projK(proj_K){}
    
		vector<Point2f> proj_p_;
		vector<cv::Point3f> worldPoints_;
		const Mat projK;

		//**3次元復元結果を用いた最適化**//

		//対応点の重心距離を最小化
		int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		{
			Mat vr = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
			Mat vt = (cv::Mat_<double>(3, 1) << _Rt[3], _Rt[4], _Rt[5]);
			Mat R_33(3, 3, CV_64F, Scalar::all(0));
			cv::Rodrigues(vr, R_33);


			//各対応点のプロジェクタ画像上での重心を求める
			//(1)proj_p_
			float sum_px = 0, sum_py = 0, px = 0, py = 0;
			for(int i = 0; i < proj_p_.size(); i++)
			{
				sum_px += proj_p_[i].x;
				sum_py += proj_p_[i].y;
			}
			px = sum_px / proj_p_.size();
			py = sum_py / proj_p_.size();

			//(2)worldPoints_
			// 2次元(プロジェクタ画像)平面へ投影
			std::vector<cv::Point2f> pt;
			cv::projectPoints(worldPoints_, R_33, vt, projK, cv::Mat(), pt); 
			float sum_wx = 0, sum_wy = 0, wx = 0, wy = 0;
			for(int i = 0; i < pt.size(); i++)
			{
				sum_wx += pt[i].x;
				sum_wy += pt[i].y;
			}
			wx = sum_wx / pt.size();
			wy = sum_wy / pt.size();

			//誤差
			for(int i = 0; i < proj_p_.size(); i++)
			{
				fvec[i] = pow(px - wx, 2) + pow(py - wy, 2);
			}

			std::cout << "error: " << fvec[0] << std::endl;
			return 0;
		}

		const int inputs_;
		const int values_;
		int inputs() const { return inputs_; }
		int values() const { return values_; }

	};

};

#endif