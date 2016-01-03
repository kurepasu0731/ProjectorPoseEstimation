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
		cv::Mat undist_img1;
		//カメラ画像の歪み除去
		cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
		//コーナー検出
		getCheckerCorners(cameraImageCorners, undist_img1, draw_image);

		//コーナー検出できたら、位置推定開始
		if(detect)
		{
			calcProjectorPose(initialR, initialT, dstR, dstT);
		}
		else{
			return false;
		}
	}

	//計算部分
	void calcProjectorPose(cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
	{
		
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
		misra1a_functor(int inputs, int values, vector<Point2f> *proj_p, vector<Point2f> *cam_p) 
		: inputs_(inputs), values_(values), proj_p_(proj_p), cam_p_(cam_p) {}
    
		vector<Point2f> *proj_p_;
		vector<Point2f> *cam_p_;
		int operator()(const VectorXd& Rt, VectorXd& fvec) const
		{
			for (int i = 0; i < values_; ++i) {
				//fvec[i] = pow((p->at(i).x - ((c[0] * P->at(i).x + c[1] * P->at(i).y + c[2] * P->at(i).z + c[3]) / (c[8] * P->at(i).x + c[9] * P->at(i).y + c[10] * P->at(i).z + c[11]))), 2) + 
				//			pow((p->at(i).y - ((c[4] * P->at(i).x + c[5] * P->at(i).y + c[6] * P->at(i).z + c[7]) / (c[8] * P->at(i).x + c[9] * P->at(i).y + c[10] * P->at(i).z + c[11]))), 2);


			}
			return 0;
		}

		//Rt[rx, ry, rz, tx, ty, tz]からRt行列を作る
		Mat getTransformMat(const VectorXd& _Rt)
		{
			Mat dst(3, 4, CV_64F, Scalar::all(0));
			//回転ベクトルから回転行列にする
			Mat rotateVec = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
			Mat rotateMat(3, 3, CV_64F, Scalar::all(0));
			Rodrigues(rotateVec, rotateMat);

			return dst;
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
	};



	void getCheckerCorners(std::vector<cv::Point2f> &imagePoint, const cv::Mat &image, cv::Mat &draw_image)
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

	void getProjectorImageCorners(std::vector<cv::Point2f> &projPoint, int _row, int _col, int _blockSize, cv::Size _offset)
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