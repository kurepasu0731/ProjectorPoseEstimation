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

	bool detect; //交点を検出できたかどうか 

	std::vector<cv::Point2f> projectorImageCorners; //プロジェクタ画像上の対応点座標
	std::vector<cv::Point2f> cameraImageCorners; //カメラ画像上の対応点座標


	//コンストラクタ
	ProjectorEstimation(WebCamera _camera, WebCamera _projector, int _checkerRow, int _checkerCol, int _blockSize, cv::Size offset) //よこ×たて
	{
		camera = _camera;
		projector = _projector;
		checkerPattern = cv::Size(_checkerRow, _checkerCol);
		detect = false;
		//プロジェクタ画像上の対応点初期化
		getProjectorImageCorners(projectorImageCorners, _checkerRow, _checkerCol, _blockSize, offset);
	};

	~ProjectorEstimation(){};


	//コーナー検出
	cv::Mat findCorners(cv::Mat frame){
		cv::Mat undist_img1, gray_img1;
		//歪み除去
		cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
		//グレースケール
		cv::cvtColor(undist_img1, gray_img1, CV_BGR2GRAY);

		//コーナー検出
		std::vector<cv::Point2f> corners1;
		int corners = 150;
		cv::goodFeaturesToTrack(gray_img1, corners1, corners, 0.001, 15);

		//描画
		for(int i = 0; i < corners1.size(); i++)
		{
			cv::circle(undist_img1, corners1[i], 1, cv::Scalar(0, 0, 255), 3);
		}

		return undist_img1;

	}


	//プロジェクタ位置姿勢を推定
	bool findProjectorPose(cv::Mat frame, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat& draw_image){
		//cv::Mat undist_img1;
		////カメラ画像の歪み除去
		//cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
		//コーナー検出
		//getCheckerCorners(cameraImageCorners, undist_img1, draw_image);

		//コーナー検出(カメラ画像は歪んだまま)
		getCheckerCorners(cameraImageCorners, frame, draw_image);

		//コーナー検出できたら、位置推定開始
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

			calcProjectorPose(undistort_imagePoint, undistort_projPoint, initialR, initialT, dstR, dstT);
		}
		else{
			return false;
		}
	}

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
		misra1a_functor(int inputs, int values, vector<Point2f>& proj_p, vector<Point2f>& cam_p, Mat& cam_K, Mat& proj_K) 
		: inputs_(inputs), values_(values), proj_p_(proj_p), cam_p_(cam_p), cam_K_(cam_K), proj_K_(proj_K) {}
    
		vector<Point2f> proj_p_;
		vector<Point2f> cam_p_;
		const Mat cam_K_;
		const Mat proj_K_;

		//int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		//{
		//	Mat Rt = getTransformMat(_Rt);

		//	for (int i = 0; i < values_; ++i) {
		//		//fvec[i] = pow((p->at(i).x - ((c[0] * P->at(i).x + c[1] * P->at(i).y + c[2] * P->at(i).z + c[3]) / (c[8] * P->at(i).x + c[9] * P->at(i).y + c[10] * P->at(i).z + c[11]))), 2) + 
		//		//			pow((p->at(i).y - ((c[4] * P->at(i).x + c[5] * P->at(i).y + c[6] * P->at(i).z + c[7]) / (c[8] * P->at(i).x + c[9] * P->at(i).y + c[10] * P->at(i).z + c[11]))), 2);

		//		Mat cp = (cv::Mat_<double>(3, 1) << (double)cam_p_.at(i).x,  (double)cam_p_.at(i).y,  1);
		//		Mat wp = cam_K_.inv() * cp;
		//		Mat _wp = (cv::Mat_<double>(4, 1) << wp.at<double>(0, 0),  wp.at<double>(0, 1),  wp.at<double>(0, 2), 1.0);
		//		Mat _pp = proj_K_ * Rt * _wp;
		//		Mat pp = (cv::Mat_<double>(2, 1) << _pp.at<double>(0,0) / _pp.at<double>(0,2), _pp.at<double>(0,1) / _pp.at<double>(0,2));

		//		fvec[i] = pow(proj_p_.at(i).x - pp.at<double>(0,0), 2) +
		//						pow(proj_p_.at(i).y -  pp.at<double>(0,0), 2);

		//		std::cout << "proj_p：(" << proj_p_.at(i).x << ", " << proj_p_.at(i).y << ")" << std::endl;
		//		std::cout << "pp：(" <<  pp.at<double>(0,0) << ", " <<  pp.at<double>(0,1) << ")" << std::endl;
		//		std::cout << "二乗誤差：" << fvec[i] << std::endl;

		//	}
		//	return 0;
		//}

		int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		{
			//回転ベクトルから回転行列にする
			Mat rotateVec = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
			Mat R(3, 3, CV_64F, Scalar::all(0));
			Rodrigues(rotateVec, R);
			//[t]x
			Mat tx = (cv::Mat_<double>(3, 3) << 0, -_Rt[5], _Rt[4], _Rt[5], 0, -_Rt[3], -_Rt[4], _Rt[3], 0);

			//カメラのK-t
			Mat camK_inv_t = cam_K_.inv().t();

			for (int i = 0; i < values_; ++i) {

				Mat cp = (cv::Mat_<double>(3, 1) << (double)cam_p_.at(i).x,  (double)cam_p_.at(i).y,  1);
				Mat pp = (cv::Mat_<double>(3, 1) << (double)proj_p_.at(i).x,  (double)proj_p_.at(i).y,  1);

				Mat error = cp.t() * camK_inv_t * tx * R * proj_K_.inv() * pp;
				fvec[i] = error.at<double>(0, 0);
			}
			return 0;
		}

	  /*
		int df(const VectorXd& b, MatrixXd& fjac)
		{	
			for (int i = 0; i < values_; ++i) {
		  fjac(i, 0) = (1.0 - exp(-b[1]*x[i]));
		  fjac(i, 1) = (b[0]*x[i] * exp(-b[1]*x[i]));
			}
			return 0;
		}
	  */
		const int inputs_;
		const int values_;
		int inputs() const { return inputs_; }
		int values() const { return values_; }

		//Rt[rx, ry, rz, tx, ty, tz]からRt行列を作る
		Mat getTransformMat(const VectorXd& _Rt) const
		{
			Mat dst(3, 4, CV_64F, Scalar::all(0));
			//回転ベクトルから回転行列にする
			Mat rotateVec = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
			Mat rotateMat(3, 3, CV_64F, Scalar::all(0));
			Rodrigues(rotateVec, rotateMat);

			//Transform行列生成
			dst.at<double>(0,0) = rotateMat.at<double>(0,0);
			dst.at<double>(0,1) = rotateMat.at<double>(0,1);
			dst.at<double>(0,2) = rotateMat.at<double>(0,2);
			dst.at<double>(0,3) = _Rt[3];
			dst.at<double>(1,0) = rotateMat.at<double>(1,0);
			dst.at<double>(1,1) = rotateMat.at<double>(1,1);
			dst.at<double>(1,2) = rotateMat.at<double>(1,2);
			dst.at<double>(1,3) = _Rt[4];
			dst.at<double>(2,0) = rotateMat.at<double>(2,0);
			dst.at<double>(2,1) = rotateMat.at<double>(2,1);
			dst.at<double>(2,2) = rotateMat.at<double>(2,2);
			dst.at<double>(2,3) = _Rt[5];

			return dst;
		}

	};

	//計算部分
	void calcProjectorPose(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
	{
		//回転行列から回転ベクトルにする
		Mat rotateVec(3, 1,  CV_64F, Scalar::all(0));
		Rodrigues(initialR, rotateVec);

		int n = 6; //変数の数
		int info;
		
		VectorXd initial(n);
		initial <<
			rotateVec.at<double>(0, 0),
			rotateVec.at<double>(1, 0),
			rotateVec.at<double>(2, 0),
			initialT.at<double>(0, 0),
			initialT.at<double>(1, 0),
			initialT.at<double>(2, 0);

		//std::cout << "size: " << cameraImageCorners.size() << std::endl;
		//std::cout << "initial: " << initial << std::endl;

		//std::cout << "camera K: " << camera.cam_K << std::endl;
		//std::cout << "projector K: " << projector.cam_K << std::endl;

		misra1a_functor functor(n, imagePoints.size(), projPoints, imagePoints, camera.cam_K, projector.cam_K);
    
		NumericalDiff<misra1a_functor> numDiff(functor);
		LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
		info = lm.minimize(initial);
    
		std::cout << "学習結果: " << std::endl;
		std::cout <<
			initial[0] << " " <<
			initial[1] << " " <<
			initial[2] << " " <<
			initial[3] << " " <<
			initial[4] << " " <<
			initial[5]	 << std::endl;

		//出力
		Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
		Rodrigues(dstRVec, dstR);
		dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);



	}

	void getCheckerCorners(std::vector<cv::Point2f>& imagePoint, const cv::Mat &image, cv::Mat &draw_image)
	{
		//交点検出
		detect = cv::findChessboardCorners(image, checkerPattern, imagePoint);

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

};

#endif