
#include "Header.h"
#include "ProjectorEstimation.h"
#include "WebCamera.h"
#include "SfM.h"
#include "Projection.hpp"
//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


//チェスパターン投影関連
const std::string chessimage_name("./chessPattern/chessPattern_14_8.png"); //1マス80px, 白枠のoffset(80, 80)
//const std::string chessimage_name("./chessPattern/chessPattern_18_11_64_48.png"); //1マス64px, 白枠のoffset(64, 48)
const char* projwindowname = "Full Window";

//プロジェクタ
WebCamera mainProjector(1280, 800, "projector0");
//カメラ(プロジェクタの後にカメラを初期化すること)
WebCamera mainCamera(1920, 1080, "webCamera0");

//CalibデータのR,t(=初期位置)
cv::Mat calib_R = cv::Mat::eye(3,3,CV_64F);
cv::Mat calib_t;


void loadProCamCalibFile(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode node(fs.fs, NULL);

	//カメラパラメータ読み込み
	read(node["cam_K"], mainCamera.cam_K);
	read(node["cam_dist"], mainCamera.cam_dist);
	//プロジェクタパラメータ読み込み
	read(node["proj_K"], mainProjector.cam_K);
	read(node["proj_dist"], mainProjector.cam_dist);

	read(node["R"], calib_R);
	read(node["T"], calib_t);

	std::cout << "ProCamCalib data file loaded." << std::endl;
}


int main()
{
	//操作説明
		printf("0 : カメラ・プロジェクタのキャリブレーション結果読み込み\n");
		printf("1: チェッカー検出開始\n");
		printf("c : 撮影\n"); 

		std::cout << "Camera 解像度：" << mainCamera.width << " * " << mainCamera.height << std::endl;
		std::cout << "Projector 解像度：" << mainProjector.width << " * " << mainProjector.height << std::endl;

		double scale = 0.001;
		std::cout << "scale: " << scale << std::endl;

	// キー入力受付用の無限ループ
	while(true){
		printf("====================\n");
		printf("数字を入力してください....\n");
		int command;

		//カメラメインループ
		while(true)
		{
			// 何かのキーが入力されたらループを抜ける
			command = cv::waitKey(33);
			if ( command > 0 ){
				//cキーで撮影
				if(command == 'c')
					mainCamera.capture();
				//m1キーで3sに1回100枚連続撮影
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

		// 条件分岐
		switch (command){

		case '0' :
			{
				loadProCamCalibFile("./calibration.xml");
				break;
			}
		case '1':
			{
				//投影画像をロード
				cv::Mat chessimage = cv::imread(chessimage_name,1);

				//ウィンドウ作成
				cv::namedWindow(projwindowname,0);

				//指定のウィンドウをフルスクリーンに設定
				/*****仕様書*****
				DISP_NUMBER:表示したいデバイスの番号を指定．
					ディスプレイのみ接続している状態
						0=ディスプレイ
					ディスプレイ+プロジェクタが接続している状態
						0=ディスプレイ
						1=プロジェクタ
				windowname:表示したいウィンドウの名前
				*****************/
				Projection::MySetFullScrean(DISP_NUMBER,projwindowname);

				//全画面表示
				cv::imshow(projwindowname,chessimage);

				// 3Dビューア(GLと同じ右手座標系)
				pcl::visualization::PCLVisualizer viewer("3D Viewer");
				viewer.setBackgroundColor(0, 0, 0);
				viewer.addCoordinateSystem(1.0); //プロジェクタ
				viewer.addCoordinateSystem(0.5,"camera"); //カメラ
				viewer.initCameraParameters();
				Eigen::Affine3f view;
				Eigen::Matrix4f trans;

				//ProjectorEstimation projectorestimation(mainCamera, mainProjector, 17, 10, 64, cv::Size(128, 112)); 
				ProjectorEstimation projectorestimation(mainCamera, mainProjector, 13, 7, 80, cv::Size(160, 160));

				//3次元復元結果読み込み
				projectorestimation.loadReconstructFile("./reconstructPoints_camera.xml");
				
				//初期値
				Mat initialR = calib_R;
				Mat initialT = calib_t;

				//カメラメインループ
				while(true)
				{
					// 何かのキーが入力されたらループを抜ける
					command = cv::waitKey(33);
					if ( command > 0 ){
						//cキーで撮影
						if(command == 'c')
							mainCamera.capture();
						else break;
					}

					cv::Mat draw_image, R, t;

					bool result = projectorestimation.findProjectorPose(mainCamera.getFrame(), initialR, initialT, R, t, draw_image);
					//位置推定結果
					if(result)
					{
						//--viewerで座標軸表示(更新)--//
						trans << (float)R.at<double>(0,0) , (float)R.at<double>(0,1) , (float)R.at<double>(0,2) , (float)t.at<double>(0,0) * scale, 
							(float)R.at<double>(1,0) , (float)R.at<double>(1,1) , (float)R.at<double>(1,2) , (float)t.at<double>(1,0) * scale, 
								  (float)R.at<double>(2,0) , (float)R.at<double>(2,1) , (float)R.at<double>(2,2) , (float)-t.at<double>(2,0) * scale, 
								  0.0f, 0.0f ,0.0f, 1.0f;
						view = trans;
						viewer.updateCoordinateSystemPose("reference", view);
						//--コンソール表示--//
						std::cout << "-----\nR: \n" << R << std::endl;
						std::cout << "t: \n" << t << std::endl;

						//初期値更新
						initialR = R;
						initialT = t;

					}
					//チェスパターン検出結果
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