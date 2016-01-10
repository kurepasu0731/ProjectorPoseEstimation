
#include "Header.h"
#include "ProjectorEstimation.h"
#include "WebCamera.h"
#include "SfM.h"
#include "Projection.hpp"
//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


//�`�F�X�p�^�[�����e�֘A
const std::string chessimage_name("./chessPattern/chessPattern_14_8.png"); //1�}�X80px, ���g��offset(80, 80)
//const std::string chessimage_name("./chessPattern/chessPattern_18_11_64_48.png"); //1�}�X64px, ���g��offset(64, 48)
const char* projwindowname = "Full Window";

//�v���W�F�N�^
WebCamera mainProjector(1280, 800, "projector0");
//�J����(�v���W�F�N�^�̌�ɃJ���������������邱��)
WebCamera mainCamera(1920, 1080, "webCamera0");

//Calib�f�[�^��R,t(=�����ʒu)
cv::Mat calib_R = cv::Mat::eye(3,3,CV_64F);
cv::Mat calib_t;


void loadProCamCalibFile(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode node(fs.fs, NULL);

	//�J�����p�����[�^�ǂݍ���
	read(node["cam_K"], mainCamera.cam_K);
	read(node["cam_dist"], mainCamera.cam_dist);
	//�v���W�F�N�^�p�����[�^�ǂݍ���
	read(node["proj_K"], mainProjector.cam_K);
	read(node["proj_dist"], mainProjector.cam_dist);

	read(node["R"], calib_R);
	read(node["T"], calib_t);

	std::cout << "ProCamCalib data file loaded." << std::endl;
}


int main()
{
	//�������
		printf("0 : �J�����E�v���W�F�N�^�̃L�����u���[�V�������ʓǂݍ���\n");
		printf("1: �`�F�b�J�[���o�J�n\n");
		printf("c : �B�e\n"); 

		std::cout << "Camera �𑜓x�F" << mainCamera.width << " * " << mainCamera.height << std::endl;
		std::cout << "Projector �𑜓x�F" << mainProjector.width << " * " << mainProjector.height << std::endl;

		double scale = 0.001;
		std::cout << "scale: " << scale << std::endl;

	// �L�[���͎�t�p�̖������[�v
	while(true){
		printf("====================\n");
		printf("��������͂��Ă�������....\n");
		int command;

		//�J�������C�����[�v
		while(true)
		{
			// �����̃L�[�����͂��ꂽ�烋�[�v�𔲂���
			command = cv::waitKey(33);
			if ( command > 0 ){
				//c�L�[�ŎB�e
				if(command == 'c')
					mainCamera.capture();
				//m1�L�[��3s��1��100���A���B�e
				else if(command == 'm')
				{
					while(mainCamera.capture_num < 100)
					{
						Sleep(3000);
						mainCamera.idle();
						mainCamera.capture();
					}
				}
				else break;
			}
			mainCamera.idle();
		}

		// ��������
		switch (command){

		case '0' :
			{
				loadProCamCalibFile("./calibration.xml");
				break;
			}
		case '1':
			{
				//���e�摜�����[�h
				cv::Mat chessimage = cv::imread(chessimage_name,1);

				//�E�B���h�E�쐬
				cv::namedWindow(projwindowname,0);

				//�w��̃E�B���h�E���t���X�N���[���ɐݒ�
				/*****�d�l��*****
				DISP_NUMBER:�\���������f�o�C�X�̔ԍ����w��D
					�f�B�X�v���C�̂ݐڑ����Ă�����
						0=�f�B�X�v���C
					�f�B�X�v���C+�v���W�F�N�^���ڑ����Ă�����
						0=�f�B�X�v���C
						1=�v���W�F�N�^
				windowname:�\���������E�B���h�E�̖��O
				*****************/
				Projection::MySetFullScrean(DISP_NUMBER,projwindowname);

				//�S��ʕ\��
				cv::imshow(projwindowname,chessimage);

				// 3D�r���[�A(GL�Ɠ����E����W�n)
				pcl::visualization::PCLVisualizer viewer("3D Viewer");
				viewer.setBackgroundColor(0, 0, 0);
				viewer.addCoordinateSystem(1.0); //�v���W�F�N�^
				viewer.addCoordinateSystem(0.5,"camera"); //�J����
				viewer.initCameraParameters();
				Eigen::Affine3f view;
				Eigen::Matrix4f trans;

				//ProjectorEstimation projectorestimation(mainCamera, mainProjector, 17, 10, 64, cv::Size(128, 112)); 
				ProjectorEstimation projectorestimation(mainCamera, mainProjector, 13, 7, 80, cv::Size(160, 160));

				//3�����������ʓǂݍ���
				projectorestimation.loadReconstructFile("./reconstructPoints_camera.xml");
				
				//�����l
				Mat initialR = calib_R;
				Mat initialT = calib_t;

				//�J�������C�����[�v
				while(true)
				{
					// �����̃L�[�����͂��ꂽ�烋�[�v�𔲂���
					command = cv::waitKey(33);
					if ( command > 0 ){
						//c�L�[�ŎB�e
						if(command == 'c')
							mainCamera.capture();
						else break;
					}

					cv::Mat draw_image, R, t;

					bool result = projectorestimation.findProjectorPose(mainCamera.getFrame(), initialR, initialT, R, t, draw_image);
					//�ʒu���茋��
					if(result)
					{
						//--viewer�ō��W���\��(�X�V)--//
						trans << (float)R.at<double>(0,0) , (float)R.at<double>(0,1) , (float)R.at<double>(0,2) , (float)t.at<double>(0,0) * scale, 
							(float)R.at<double>(1,0) , (float)R.at<double>(1,1) , (float)R.at<double>(1,2) , (float)t.at<double>(1,0) * scale, 
								  (float)R.at<double>(2,0) , (float)R.at<double>(2,1) , (float)R.at<double>(2,2) , (float)-t.at<double>(2,0) * scale, 
								  0.0f, 0.0f ,0.0f, 1.0f;
						view = trans;
						viewer.updateCoordinateSystemPose("reference", view);
						//--�R���\�[���\��--//
						std::cout << "-----\nR: \n" << R << std::endl;
						std::cout << "t: \n" << t << std::endl;

						//�����l�X�V
						initialR = R;
						initialT = t;

					}
					//�`�F�X�p�^�[�����o����
					cv::imshow("Camera image", draw_image);
				}
				break;
			}
		default:
			exit(0);
			break;
		}
	}

	return 0;
}