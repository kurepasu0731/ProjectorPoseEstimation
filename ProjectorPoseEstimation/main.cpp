
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

//チェスパターン投影関連
//const std::string chessimage_name("./chessPattern/chessPattern_14_8.png"); //1マス80px, 白枠のoffset(80, 80)
const std::string chessimage_name("./chessPattern/chessPattern_18_11_64_48.png"); //1マス64px, 白枠のoffset(64, 48)
//const std::string chessimage_name("./chessPattern/chessPattern_30_18.png"); //1マス40px, 白枠のoffset(40, 40)
//ドラえもん投影画像
const std::string doraimage_name("./chessPattern/projectorimage.png");
const char* projwindowname = "Full Window";

//プロジェクタ
WebCamera mainProjector(cv::VideoCapture(0), 1280, 800, "projector0");
//カメラ(プロジェクタの後にカメラを初期化すること)
WebCamera mainCamera(cv::VideoCapture(0), 1920, 1080, "webCamera0");

//CalibデータのR,t(=初期位置)
cv::Mat calib_R = cv::Mat::eye(3,3,CV_64F);
cv::Mat calib_t;

//処理時間計測用
CFileTime cTimeStart, cTimeEnd;
CFileTimeSpan cTimeSpan;


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
		printf("2: コーナー検出開始(重心)\n");
		printf("3: コーナー検出開始(最近傍)\n");
		printf("4: 動画撮影モード\n");
		printf("c : キャプチャ\n"); 
		printf("q : 終了\n"); 

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
		//チェッカボード検出による位置推定
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

				//投影遅延待ち
				cv::waitKey(64);

				// 3Dビューア(GLと同じ右手座標系)
				pcl::visualization::PCLVisualizer viewer("3D Viewer");
				viewer.setBackgroundColor(0, 0, 0);
				viewer.addCoordinateSystem(1.0); //プロジェクタ
				viewer.addCoordinateSystem(0.5,"camera"); //カメラ
				//viewer.initCameraParameters();
				viewer.setCameraPosition(0, 3, 0, 0, 0, 0, 0, 0, 1);
				Eigen::Affine3f view;
				Eigen::Matrix4f trans;

				ProjectorEstimation projectorestimation(mainCamera, mainProjector, 17, 10, 64, cv::Size(128, 112)); 
				//ProjectorEstimation projectorestimation(mainCamera, mainProjector, 13, 7, 80, cv::Size(160, 160));
				//ProjectorEstimation projectorestimation(mainCamera, mainProjector, 29, 17, 40, cv::Size(80, 80)); 

				//3次元復元結果読み込み
				projectorestimation.loadReconstructFile("./reconstructPoints_camera.xml");
				
				//初期値
				cv::Mat initialR = calib_R;
				cv::Mat initialT = calib_t;

				////一個前の推定結果と現推定結果の差分
				//Mat dR = cv::Mat::zeros(3,3,CV_64F);
				//Mat dt = cv::Mat::zeros(3,1,CV_64F);

				try{

					//カメラメインループ
					while(true)
					{
						//処理時間計測開始
						cTimeStart = CFileTime::GetCurrentTime();// 現在時刻

						// 何かのキーが入力されたらループを抜ける
						command = cv::waitKey(33);
						if ( command > 0 ){
							//cキーで撮影
							if(command == 'c')
								mainCamera.idle();
							else break;
						}

						cv::Mat draw_image, R, t;

						////動き予測
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
							std::cout << "-----\ndstR: \n" << R << std::endl;
							std::cout << "dstT: \n" << t << std::endl;

							////差分を取る
							//dR = R - initialR;
							//dt = t - initialT;

							//初期値更新
							initialR = R;
							initialT = t;


						}
						//チェスパターン検出結果
						cv::imshow("Camera image", draw_image);
						//コーナー検出結果表示
						cv::Mat resize;
						cv::resize(draw_chessimage, resize, cv::Size(), 0.5, 0.5);
						cv::imshow("detected Points", draw_chessimage);

						cTimeEnd = CFileTime::GetCurrentTime();
						cTimeSpan = cTimeEnd - cTimeStart;
						std::cout<< "1frame処理時間:" << cTimeSpan.GetTimeSpan()/10000 << "[ms]" << std::endl;

					}

					throw "Exception!!\n";
				}

				catch(char *e){
					std::cout << e;
				}

				break;
			}
		//コーナー検出による位置推定(重心)
		case '2':
		//コーナー検出による位置推定(最近傍)←今こっち
		case '3':
			{
				//投影画像をロード
				//cv::Mat chessimage = cv::imread(chessimage_name,1);
				cv::Mat chessimage = cv::imread(doraimage_name,1);

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

				//投影遅延待ち
				cv::waitKey(64);

				// 3Dビューア(GLと同じ右手座標系)
				pcl::visualization::PCLVisualizer viewer("3D Viewer");
				viewer.setBackgroundColor(0, 0, 0);
				viewer.addCoordinateSystem(1.0); //プロジェクタ
				viewer.addCoordinateSystem(0.5,"camera"); //カメラ
				viewer.setCameraPosition(0, 3, 0, 0, 0, 0, 0, 0, 1);
				Eigen::Affine3f view;
				Eigen::Matrix4f trans;

				ProjectorEstimation projectorestimation(mainCamera, mainProjector, 17, 10, 64, cv::Size(128, 112)); 

				//3次元復元結果読み込み
				projectorestimation.loadReconstructFile("./reconstructPoints_camera.xml");
				
				//初期値
				cv::Mat initialR = calib_R;
				cv::Mat initialT = calib_t;

				//処理時間計測開始
				CFileTime startTime = CFileTime::GetCurrentTime();// 現在時刻

				try{

					//カメラメインループ
					while(true)
					{
						//処理時間計測開始
						cTimeStart = CFileTime::GetCurrentTime();// 現在時刻

						// 何かのキーが入力されたらループを抜ける
						command = cv::waitKey(33);
						if ( command > 0 ){
							//cキーで撮影
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
							std::cout << "-----\ndstR: \n" << R << std::endl;
							std::cout << "dstT: \n" << t << std::endl;

							//初期値更新
							initialR = R;
							initialT = t;
						}
						//コーナー検出結果表示
						cv::Mat resize_cam, resize_proj;
						cv::resize(draw_image, resize_cam, cv::Size(), 0.5, 0.5);
						cv::imshow("Camera detected corners", resize_cam);
						cv::resize(draw_chessimage, resize_proj, cv::Size(), 0.5, 0.5);
						cv::imshow("Projector detected corners", draw_chessimage);

						cTimeEnd = CFileTime::GetCurrentTime();
						cTimeSpan = cTimeEnd - cTimeStart;
						std::cout<< "1frame処理時間:" << cTimeSpan.GetTimeSpan()/10000 << "[ms]" << std::endl;

						throw "Exception!!\n";
					}
				}
				catch(char *e)
				{
					std::cout << e;
				}
				break;
			}
		//動画撮影(sキー：撮影開始 qキー:撮影終了)
		case '4':
			{
				printf("動画撮影モード(sキー：撮影開始 qキー:撮影終了)\n");
				//出力動画ファイルの設定(fpsをあげると早送りになる)
				cv::VideoWriter writer("output.avi", CV_FOURCC_DEFAULT, 10, 
					cv::Size((int)mainCamera.vc.get(CV_CAP_PROP_FRAME_WIDTH), (int)mainCamera.vc.get(CV_CAP_PROP_FRAME_HEIGHT)), true);
				
				//メインループ
				while(true)
				{
					// キー入力
					command = cv::waitKey(33);
					if ( command > 0 ){
						//sキーで録画開始
						if(command == 's')
						{
							printf("撮影中...");
							cv::Mat frame;
							while(true)
							{
								// キー入力
								command = cv::waitKey(33);
								if(command == 'q') break;

								//フレームの保存
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
							printf("撮影終了.\n");
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