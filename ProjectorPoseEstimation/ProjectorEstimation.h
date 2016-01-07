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

	//3�����_(�J�������S)LookUp�e�[�u��
	//** index = �J������f(����n�܂�)
	//** int image_x = i % CAMERA_WIDTH;
	//** int image_y = (int)(i / CAMERA_WIDTH);
	//** Point3f = �J������f��3�������W(�v������Ă��Ȃ��ꍇ��(-1, -1, -1))
	std::vector<cv::Point3f> reconstructPoints;


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


	//3�����������ʓǂݍ���
	void loadReconstructFile(const string& filename)
	{
		//3�����_(�J�������S)LookUp�e�[�u���̃��[�h
		FileStorage fs(filename, FileStorage::READ);
		FileNode node(fs.fs, NULL);

		read(node["points"], reconstructPoints);

		std::cout << "3�����������ʓǂݍ���." << std::endl;
	}

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
		//cv::Mat undist_img1;
		////�J�����摜�̘c�ݏ���
		//cv::undistort(frame, undist_img1, camera.cam_K, camera.cam_dist);
		//�R�[�i�[���o
		//getCheckerCorners(cameraImageCorners, undist_img1, draw_image);

		//�R�[�i�[���o(�J�����摜�͘c�񂾂܂�)
		getCheckerCorners(cameraImageCorners, frame, draw_image);

		//�R�[�i�[���o�ł�����A�ʒu����J�n
		if(detect)
		{
			// �Ή��_�̘c�ݏ���
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
		// �ړI�֐�
		misra1a_functor(int inputs, int values, vector<Point2f>& proj_p, vector<Point2f>& cam_p, Mat& cam_K, Mat& proj_K, std::vector<cv::Point3f> reconstructPoints) 
			: inputs_(inputs),
			  values_(values), 
			  proj_p_(proj_p),
			  cam_p_(cam_p), 
			  reconstructPoints_(reconstructPoints),
			  cam_K_(cam_K), 
			  proj_K_(proj_K),
			  projK_inv_t(proj_K_.inv().t()), 
			  camK_inv(cam_K.inv()) {}
    
		vector<Point2f> proj_p_;
		vector<Point2f> cam_p_;
		vector<cv::Point3f> reconstructPoints_;
		const Mat cam_K_;
		const Mat proj_K_;
		const Mat camK_inv;
		const Mat projK_inv_t;

		//**�G�s�|�[����������p�����œK��**//

		//R�̎��R�x3�ɂ���
		//int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		//{
		//	//��]�x�N�g�������]�s��ɂ���
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

		//R�̎��R�x��9�ɂ���
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
		//		//���Ɍv�Z
		//		//fvec[i] = (double)cam_p_.at(i).x * (camK_inv_t.at<double>(0, 0) * (-_Rt[11] * (_Rt[3] * (projK_inv.at<double>(0, 0) * proj_p_.at(i).x + projK_inv.at<double>(0, 1) * proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(0, 1) * (_Rt[11] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(0, 2) * (-_Rt[10] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))))) +
		//		//		  (double)cam_p_.at(i).y * (camK_inv_t.at<double>(1, 0) * (-_Rt[11] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(1, 1) * (_Rt[11] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(1, 2) * (-_Rt[10] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))))) +
		//		//		                    camK_inv_t.at<double>(2, 0) * (-_Rt[11] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(2, 1) * (_Rt[11] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[7] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[8] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2)))) + camK_inv_t.at<double>(2, 2) * (-_Rt[10] * (_Rt[0] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[1] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[2] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (projK_inv.at<double>(0, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(0, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(0, 2)) + _Rt[4] * (projK_inv.at<double>(1, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(1, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(1, 2)) + _Rt[5] * (projK_inv.at<double>(2, 0) * (double)proj_p_.at(i).x + projK_inv.at<double>(2, 1) * (double)proj_p_.at(i).y + projK_inv.at<double>(2, 2))));
		//	}
		//	return 0;
		//}


		//**3�����������ʂ�p�����œK��**//

		int operator()(const VectorXd& _Rt, VectorXd& fvec) const
		{

			//// 2����(�v���W�F�N�^�摜)���ʂ֓��e
			//std::vector<cv::Point2f> pt;
			//cv::projectPoints(reconstructPoints_, R, t, proj_K_, cv::Mat(), pt); 

			// �ˉe�덷�Z�o(�L���ȓ_�̂�)
			for (int i = 0; i < values_; ++i) 
			{
				int image_x = (int)(cam_p_[i].x+0.5);
				int image_y = (int)(cam_p_[i].y+0.5);
				int index = image_y * CAMERA_WIDTH + image_x;

				if(reconstructPoints_[index].x != -1)
				{
					Mat wp = (cv::Mat_<double>(4, 1) << reconstructPoints_[index].x, reconstructPoints_[index].y, reconstructPoints_[index].z, 1);

					Mat vr = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
					Mat R_33(3, 3, CV_64F, Scalar::all(0));
					Rodrigues(vr, R_33);
					//4*4�s��ɂ���
					Mat Rt = (cv::Mat_<double>(4, 4) << R_33.at<double>(0,0), R_33.at<double>(0,1), R_33.at<double>(0,2), _Rt[3],
						                               R_33.at<double>(1,0), R_33.at<double>(1,1), R_33.at<double>(1,2), _Rt[4],
													   R_33.at<double>(2,0), R_33.at<double>(2,1), R_33.at<double>(2,2), _Rt[5],
													   0, 0, 0, 1);
					Mat projK = (cv::Mat_<double>(3, 4) << proj_K_.at<double>(0,0), proj_K_.at<double>(0,1), proj_K_.at<double>(0,2), 0,
						                               proj_K_.at<double>(1,0), proj_K_.at<double>(1,1), proj_K_.at<double>(1,2), 0,
													   proj_K_.at<double>(2,0), proj_K_.at<double>(2,1), proj_K_.at<double>(2,2), 0);
					//�v���W�F�N�^�摜��֎ˉe
					Mat dst_p = projK * Rt * wp;
					Point2f project_p(dst_p.at<double>(0,0) / dst_p.at<double>(2,0), dst_p.at<double>(1,0) / dst_p.at<double>(2,0));

					// �ˉe�덷�Z�o
					fvec[i] = pow(project_p.x - proj_p_[i].x, 2) + pow(project_p.y - proj_p_[i].y, 2);


				}
				else
				{
					fvec[i] = 0;
				}
			}

			return 0;
		}


		const int inputs_;
		const int values_;
		int inputs() const { return inputs_; }
		int values() const { return values_; }

	};

	//**�G�s�|�[����������p�����œK��**//


	//�v�Z����(R�̎��R�x3)
	void calcProjectorPose(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
	{
		//��]�s�񂩂��]�x�N�g���ɂ���
		Mat rotateVec(3, 1,  CV_64F, Scalar::all(0));
		Rodrigues(initialR, rotateVec);

		int n = 6; //�ϐ��̐�
		int info;
		
		VectorXd initial(n);
		initial <<
			rotateVec.at<double>(0, 0),
			rotateVec.at<double>(1, 0),
			rotateVec.at<double>(2, 0),
			initialT.at<double>(0, 0),
			initialT.at<double>(1, 0),
			initialT.at<double>(2, 0);

		misra1a_functor functor(n, imagePoints.size(), projPoints, imagePoints, camera.cam_K, projector.cam_K, reconstructPoints);
    
		NumericalDiff<misra1a_functor> numDiff(functor);
		LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
		info = lm.minimize(initial);
    
		std::cout << "�w�K����: " << std::endl;
		std::cout <<
			initial[0] << " " <<
			initial[1] << " " <<
			initial[2] << " " <<
			initial[3] << " " <<
			initial[4] << " " <<
			initial[5]	 << std::endl;

		//�o��
		Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
		Rodrigues(dstRVec, dstR);
		dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);



	}

	//�v�Z����(R�̎��R�x9)
	void calcProjectorPose2(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point2f> projPoints, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
	{

		int n = 12; //�ϐ��̐�
		int info;
		
		VectorXd initial(n);
		initial <<
			initialR.at<double>(0, 0),
			initialR.at<double>(0, 1),
			initialR.at<double>(0, 2),
			initialR.at<double>(1, 0),
			initialR.at<double>(1, 1),
			initialR.at<double>(1, 2),
			initialR.at<double>(2, 0),
			initialR.at<double>(2, 1),
			initialR.at<double>(2, 2),
			initialT.at<double>(0, 0),
			initialT.at<double>(1, 0),
			initialT.at<double>(2, 0);


		misra1a_functor functor(n, imagePoints.size(), projPoints, imagePoints, camera.cam_K, projector.cam_K, reconstructPoints);
    
		NumericalDiff<misra1a_functor> numDiff(functor);
		LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
		info = lm.minimize(initial);
    
		std::cout << "�w�K����: " << std::endl;
		std::cout <<
			initial[0] << " " <<
			initial[1] << " " <<
			initial[2] << " " <<
			initial[3] << " " <<
			initial[4] << " " <<
			initial[5] << " " <<
			initial[6] << " " <<
			initial[7] << " " <<
			initial[8] << " " <<
			initial[9] << " " <<
			initial[10] << " " <<
			initial[11] << " " << std::endl;

		//�o��
		dstR = (cv::Mat_<double>(3, 3) << initial[0], initial[1], initial[2], initial[3], initial[4], initial[5], initial[6], initial[7], initial[8]);
		dstT = (cv::Mat_<double>(3, 1) << initial[9], initial[10], initial[11]);
	}


	void getCheckerCorners(std::vector<cv::Point2f>& imagePoint, const cv::Mat &image, cv::Mat &draw_image)
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