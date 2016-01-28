
#include "Header.h"
#include "ProjectorEstimation.h"
#include "WebCamera.h"
#include "SfM.h"
#include "Projection.hpp"
//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

//using namespace cv;

//�`�F�X�p�^�[�����e�֘A
//const std::string chessimage_name("./chessPattern/chessPattern_14_8.png"); //1�}�X80px, ���g��offset(80, 80)
const std::string chessimage_name("./chessPattern/chessPattern_18_11_64_48.png"); //1�}�X64px, ���g��offset(64, 48)
//const std::string chessimage_name("./chessPattern/chessPattern_30_18.png"); //1�}�X40px, ���g��offset(40, 40)
//�h�������񓊉e�摜
const std::string doraimage_name("./chessPattern/projectorimage.png");
const char* projwindowname = "Full Window";

//�v���W�F�N�^
WebCamera mainProjector(cv::VideoCapture(0), 1280, 800, "projector0");
//�J����(�v���W�F�N�^�̌�ɃJ���������������邱��)
WebCamera mainCamera(cv::VideoCapture(0), 1920, 1080, "webCamera0");

//Calib�f�[�^��R,t(=�����ʒu)
cv::Mat calib_R = cv::Mat::eye(3,3,CV_64F);
cv::Mat calib_t;

//�������Ԍv���p
CFileTime cTimeStart, cTimeEnd;
CFileTimeSpan cTimeSpan;


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
		printf("2: �R�[�i�[���o�J�n(�d�S)\n");
		printf("3: �R�[�i�[���o�J�n(�ŋߖT)\n");
		printf("4: ����B�e���[�h\n");
		printf("c : �L���v�`��\n"); 
		printf("q : �I��\n"); 

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
		//�`�F�b�J�{�[�h���o�ɂ��ʒu����
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

				//���e�x���҂�
				cv::waitKey(64);

				// 3D�r���[�A(GL�Ɠ����E����W�n)
				pcl::visualization::PCLVisualizer viewer("3D Viewer");
				viewer.setBackgroundColor(0, 0, 0);
				viewer.addCoordinateSystem(1.0); //�v���W�F�N�^
				viewer.addCoordinateSystem(0.5,"camera"); //�J����
				//viewer.initCameraParameters();
				viewer.setCameraPosition(0, 3, 0, 0, 0, 0, 0, 0, 1);
				Eigen::Affine3f view;
				Eigen::Matrix4f trans;

				ProjectorEstimation projectorestimation(mainCamera, mainProjector, 17, 10, 64, cv::Size(128, 112)); 
				//ProjectorEstimation projectorestimation(mainCamera, mainProjector, 13, 7, 80, cv::Size(160, 160));
				//ProjectorEstimation projectorestimation(mainCamera, mainProjector, 29, 17, 40, cv::Size(80, 80)); 

				//3�����������ʓǂݍ���
				projectorestimation.loadReconstructFile("./reconstructPoints_camera.xml");
				
				//�����l
				cv::Mat initialR = calib_R;
				cv::Mat initialT = calib_t;

				////��O�̐��茋�ʂƌ����茋�ʂ̍���
				//Mat dR = cv::Mat::zeros(3,3,CV_64F);
				//Mat dt = cv::Mat::zeros(3,1,CV_64F);

				try{

					//�J�������C�����[�v
					while(true)
					{
						//�������Ԍv���J�n
						cTimeStart = CFileTime::GetCurrentTime();// ���ݎ���

						// �����̃L�[�����͂��ꂽ�烋�[�v�𔲂���
						command = cv::waitKey(33);
						if ( command > 0 ){
							//c�L�[�ŎB�e
							if(command == 'c')
								mainCamera.idle();
							else break;
						}

						cv::Mat draw_image, R, t;

						////�����\��
						//initialR += dR;
						//initialT += dt;

						std::cout << "===================" << std::endl;
						//std::cout << "-----\ninitialR: \n" << initialR << std::endl;
						//std::cout << "initialT: \n" << initialT << std::endl;

						cv::Mat draw_chessimage = chessimage.clone();
						cv::undistort(chessimage, draw_chessimage, mainProjector.cam_K, mainProjector.cam_dist);

						bool result = false;
						//if(!mainCamera.getFrame().empty())
							result = projectorestimation.findProjectorPose(mainCamera.getFrame(), initialR, initialT, R, t, draw_image, draw_chessimage);
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
							std::cout << "-----\ndstR: \n" << R << std::endl;
							std::cout << "dstT: \n" << t << std::endl;

							////���������
							//dR = R - initialR;
							//dt = t - initialT;

							//�����l�X�V
							initialR = R;
							initialT = t;


						}
						//�`�F�X�p�^�[�����o����
						cv::imshow("Camera image", draw_image);
						//�R�[�i�[���o���ʕ\��
						cv::Mat resize;
						cv::resize(draw_chessimage, resize, cv::Size(), 0.5, 0.5);
						cv::imshow("detected Points", draw_chessimage);

						cTimeEnd = CFileTime::GetCurrentTime();
						cTimeSpan = cTimeEnd - cTimeStart;
						std::cout<< "1frame��������:" << cTimeSpan.GetTimeSpan()/10000 << "[ms]" << std::endl;

					}

					throw "Exception!!\n";
				}

				catch(char *e){
					std::cout << e;
				}

				break;
			}
		//�R�[�i�[���o�ɂ��ʒu����(�d�S)
		case '2':
		//�R�[�i�[���o�ɂ��ʒu����(�ŋߖT)����������
		case '3':
			{
				//���e�摜�����[�h
				//cv::Mat chessimage = cv::imread(chessimage_name,1);
				cv::Mat chessimage = cv::imread(doraimage_name,1);

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

				//���e�x���҂�
				cv::waitKey(64);

				// 3D�r���[�A(GL�Ɠ����E����W�n)
				pcl::visualization::PCLVisualizer viewer("3D Viewer");
				viewer.setBackgroundColor(0, 0, 0);
				viewer.addCoordinateSystem(1.0); //�v���W�F�N�^
				viewer.addCoordinateSystem(0.5,"camera"); //�J����
				viewer.setCameraPosition(0, 3, 0, 0, 0, 0, 0, 0, 1);
				Eigen::Affine3f view;
				Eigen::Matrix4f trans;

				ProjectorEstimation projectorestimation(mainCamera, mainProjector, 17, 10, 64, cv::Size(128, 112)); 

				//3�����������ʓǂݍ���
				projectorestimation.loadReconstructFile("./reconstructPoints_camera.xml");
				
				//�����l
				cv::Mat initialR = calib_R;
				cv::Mat initialT = calib_t;

				//�������Ԍv���J�n
				CFileTime startTime = CFileTime::GetCurrentTime();// ���ݎ���

				try{

					//�J�������C�����[�v
					while(true)
					{
						//�������Ԍv���J�n
						cTimeStart = CFileTime::GetCurrentTime();// ���ݎ���

						// �����̃L�[�����͂��ꂽ�烋�[�v�𔲂���
						command = cv::waitKey(33);
						if ( command > 0 ){
							//c�L�[�ŎB�e
							if(command == 'c')
								mainCamera.capture();
							else break;
						}

						cv::Mat draw_image, R, t;

						std::cout << "===================" << std::endl;
						std::cout << "-----\ninitialR: \n" << initialR << std::endl;
						std::cout << "initialT: \n" << initialT << std::endl;

						cv::Mat draw_chessimage = chessimage.clone();
						cv::undistort(chessimage, draw_chessimage, mainProjector.cam_K, mainProjector.cam_dist);

						bool result = false;
						//if(!mainCamera.getFrame().empty())
							result = projectorestimation.findProjectorPose_Corner(mainCamera.getFrame(), chessimage, initialR, initialT, R, t, draw_image, draw_chessimage);
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
							std::cout << "-----\ndstR: \n" << R << std::endl;
							std::cout << "dstT: \n" << t << std::endl;

							//�����l�X�V
							initialR = R;
							initialT = t;
						}
						//�R�[�i�[���o���ʕ\��
						cv::Mat resize_cam, resize_proj;
						cv::resize(draw_image, resize_cam, cv::Size(), 0.5, 0.5);
						cv::imshow("Camera detected corners", resize_cam);
						cv::resize(draw_chessimage, resize_proj, cv::Size(), 0.5, 0.5);
						cv::imshow("Projector detected corners", draw_chessimage);

						cTimeEnd = CFileTime::GetCurrentTime();
						cTimeSpan = cTimeEnd - cTimeStart;
						std::cout<< "1frame��������:" << cTimeSpan.GetTimeSpan()/10000 << "[ms]" << std::endl;

						throw "Exception!!\n";
					}
				}
				catch(char *e)
				{
					std::cout << e;
				}
				break;
			}
		//����B�e(s�L�[�F�B�e�J�n q�L�[:�B�e�I��)
		case '4':
			{
				printf("����B�e���[�h(s�L�[�F�B�e�J�n q�L�[:�B�e�I��)\n");
				//�o�͓���t�@�C���̐ݒ�(fps��������Ƒ�����ɂȂ�)
				cv::VideoWriter writer("output.avi", CV_FOURCC_DEFAULT, 10, 
					cv::Size((int)mainCamera.vc.get(CV_CAP_PROP_FRAME_WIDTH), (int)mainCamera.vc.get(CV_CAP_PROP_FRAME_HEIGHT)), true);
				
				//���C�����[�v
				while(true)
				{
					// �L�[����
					command = cv::waitKey(33);
					if ( command > 0 ){
						//s�L�[�Ř^��J�n
						if(command == 's')
						{
							printf("�B�e��...");
							cv::Mat frame;
							while(true)
							{
								// �L�[����
								command = cv::waitKey(33);
								if(command == 'q') break;

								//�t���[���̕ۑ�
								frame = mainCamera.getFrame();
								if(!frame.empty())
								{
									writer << frame;
									cv::imshow(mainCamera.winName, frame);
								}
							}
						}

						 if(command == 'q')
						{
							writer.release();
							printf("�B�e�I��.\n");
							break;
						}
					}
					mainCamera.idle();
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