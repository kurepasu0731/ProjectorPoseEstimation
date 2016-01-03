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

	bool detect; //��_�����o�ł������ǂ��� 

	std::vector<cv::Point2f> projectorImageCorners; //�v���W�F�N�^�摜��̑Ή��_���W
	std::vector<cv::Point2f> cameraImageCorners; //�J�����摜��̑Ή��_���W


	//�R���X�g���N�^
	ProjectorEstimation(WebCamera _camera, WebCamera _projector, int _checkerRow, int _checkerCol, int _blockSize, cv::Size offset) //�悱�~����
	{
		camera = _camera;
		projector = _projector;
		checkerPattern = cv::Size(_checkerRow, _checkerCol);
		detect = false;
		//�v���W�F�N�^�摜��̑Ή��_������
		getProjectorImageCorners(projectorImageCorners, _checkerRow, _checkerCol, _blockSize, offset);
	};

	~ProjectorEstimation(){};


	//�R�[�i�[���o
	cv::Mat findCorners(cv::Mat frame){
		cv::Mat undist_img1, gray_img1;
		//�c�ݏ���
		cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
		//�O���[�X�P�[��
		cv::cvtColor(undist_img1, gray_img1, CV_BGR2GRAY);

		//�R�[�i�[���o
		std::vector<cv::Point2f> corners1;
		int corners = 150;
		cv::goodFeaturesToTrack(gray_img1, corners1, corners, 0.001, 15);

		//�`��
		for(int i = 0; i < corners1.size(); i++)
		{
			cv::circle(undist_img1, corners1[i], 1, cv::Scalar(0, 0, 255), 3);
		}

		return undist_img1;

	}


	//�v���W�F�N�^�ʒu�p���𐄒�
	bool findProjectorPose(cv::Mat frame, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT, cv::Mat& draw_image){
		cv::Mat undist_img1;
		//�J�����摜�̘c�ݏ���
		cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
		//�R�[�i�[���o
		getCheckerCorners(cameraImageCorners, undist_img1, draw_image);

		//�R�[�i�[���o�ł�����A�ʒu����J�n
		if(detect)
		{
			calcProjectorPose(initialR, initialT, dstR, dstT);
		}
		else{
			return false;
		}
	}

	//�v�Z����
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
		// �ړI�֐�
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

		//Rt[rx, ry, rz, tx, ty, tz]����Rt�s������
		Mat getTransformMat(const VectorXd& _Rt)
		{
			Mat dst(3, 4, CV_64F, Scalar::all(0));
			//��]�x�N�g�������]�s��ɂ���
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
		//��_���o
		detect = cv::findChessboardCorners(image, checkerPattern, imagePoint);

		//���o�_�̕`��
		image.copyTo(draw_image);
		if(detect)
		{
			//�T�u�s�N�Z�����x
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