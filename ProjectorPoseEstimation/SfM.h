#ifndef SFM_H
#define SFM_H

#include "WebCamera.h"
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\nonfree.hpp> // SIFTまたはSURFを使う場合は必要

class SfM
{
public:
	//カメラ
	WebCamera camera;
	//プロジェクタ
	WebCamera projector;

	//基礎行列
	//cv::Mat F;
	//基本行列
	//cv::Mat E;

	//使用する画像
	cv::Mat src_camImage; // 画像1のファイル名
	cv::Mat src_projImage; // 画像2のファイル名
	cv::Mat result; //結果描画用

	//特徴点から抽出した対応点
	std::vector<cv::Point2d>cam_pts, proj_pts;
	
	//コンストラクタ
	SfM(const char *camImageName, const char *projImageName, WebCamera cam, WebCamera proj)
	{
		camera = cam;
		projector = proj;

		std::cout << "cam_K:\n" << camera.cam_K << std::endl;
		std::cout << "cam_dist:\n" << camera.cam_dist << std::endl;
		std::cout << "proj_K:\n" << projector.cam_K << std::endl;
		std::cout << "proj_dist:\n" << projector.cam_dist << std::endl;

		//歪み除去して読み込み(1枚目：カメラ　2枚目:プロジェクタ)
		cv::undistort(cv::imread(camImageName), src_camImage, camera.cam_K, camera.cam_dist);
		//cv::undistort(cv::imread(projImageName), src_projImage, projector.cam_K, projector.cam_dist);

		//歪み補正なし
		//src_camImage = cv::imread(camImageName);
		src_projImage = cv::imread(projImageName);
	};
	~SfM(){};

	void featureMatching(	const char *featureDetectorName, const char *descriptorExtractorName, const char *descriptorMatcherName, bool crossCheck)
	{
		if(featureDetectorName == "SIFT" || featureDetectorName == "SURF" 
			|| descriptorExtractorName == "SIFT" || descriptorExtractorName == "SURF")
		{
			// SIFTまたはSURFを使う場合はこれを呼び出す．
			cv::initModule_nonfree();
		}

		// 特徴点抽出
		cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(featureDetectorName);
		std::vector<cv::KeyPoint> keypoint1, keypoint2;//1->camera 2->projector
		detector->detect(src_camImage, keypoint1);
		detector->detect(src_projImage, keypoint2);

		// 特徴記述
		cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create(descriptorExtractorName);
		cv::Mat descriptor1, descriptor2;
		extractor->compute(src_camImage, keypoint1, descriptor1);
		extractor->compute(src_projImage, keypoint2, descriptor2);

		// マッチング
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(descriptorMatcherName);
		std::vector<cv::DMatch> dmatch;
		if (crossCheck)
		{
			// クロスチェックする場合
			std::vector<cv::DMatch> match12, match21;
			matcher->match(descriptor1, descriptor2, match12);
			matcher->match(descriptor2, descriptor1, match21);
			for (size_t i = 0; i < match12.size(); i++)
			{
				cv::DMatch forward = match12[i];
				cv::DMatch backward = match21[forward.trainIdx];
				if (backward.trainIdx == forward.queryIdx)
					dmatch.push_back(forward);
			}
		}
		else
		{
			// クロスチェックしない場合
			matcher->match(descriptor1, descriptor2, dmatch);
		}

		//最小距離
		double min_dist = DBL_MAX;
		for(int j = 0; j < (int)dmatch.size(); j++)
		{
			double dist = dmatch[j].distance;
			if(dist < min_dist) min_dist = (dist < 1.0) ? 1.0 : dist;
		}

		std::cout << "min_dist: " << min_dist << std::endl;

		//良いペアのみ残す
		double cutoff = 1.6 * min_dist;
		std::set<int> existing_trainIdx;
		std::vector<cv::DMatch> matches_good;
		for(int j = 0; j < (int)dmatch.size(); j++)
		{
			if(dmatch[j].trainIdx <= 0) dmatch[j].trainIdx = dmatch[j].imgIdx;
			if(dmatch[j].distance > 0.0 && dmatch[j].distance < cutoff){
				if(existing_trainIdx.find(dmatch[j].trainIdx) == existing_trainIdx.end() && dmatch[j].trainIdx >= 0 && dmatch[j].trainIdx < (int)keypoint2.size()) {
					matches_good.push_back(dmatch[j]);
                    existing_trainIdx.insert(dmatch[j].trainIdx);
				}
			}
		}

		std::cout << "対応点数:" << matches_good.size() << std::endl;

        // 対応点の登録(5ペア以上は必要)
        if (matches_good.size() > 10) {
            for (int j = 0; j < (int)matches_good.size(); j++) {
                cam_pts.push_back(keypoint1[matches_good[j].queryIdx].pt);
                proj_pts.push_back(keypoint2[matches_good[j].trainIdx].pt);
            }
		}
  //      if (dmatch.size() > 10) {
  //          for (int j = 0; j < (int)dmatch.size(); j++) {
  //              cam_pts.push_back(keypoint1[dmatch[j].queryIdx].pt);
  //              proj_pts.push_back(keypoint2[dmatch[j].trainIdx].pt);
  //          }
		//}

		// マッチング結果の表示
		cv::drawMatches(src_camImage, keypoint1, src_projImage, keypoint2, matches_good, result);
		//cv::drawMatches(src_camImage, keypoint1, src_projImage, keypoint2, dmatch, result);
		//cv::Mat resize;
		//result.copyTo(resize);
		//cv::resize(result, resize, resize.size(), 0.5, 0.5);
		cv::imshow("good matching", result);
		cv::waitKey(0);
	}

	void saveResult(const char *resultImageName)
	{
		cv::imwrite(resultImageName, result);
	}

	cv::Mat findEssentialMat(){
		// 焦点距離とレンズ主点
        double cam_f, proj_f, cam_fovx, cam_fovy, proj_fovx, proj_fovy, cam_pasp, proj_pasp;
        cv::Point2d cam_pp, proj_pp;
		cv::calibrationMatrixValues(camera.cam_K, cv::Size(camera.width, camera.height), 0.0, 0.0, cam_fovx, cam_fovy, cam_f, cam_pp, cam_pasp);
		cv::calibrationMatrixValues(projector.cam_K, cv::Size(projector.width, projector.height), 0.0, 0.0, proj_fovx, proj_fovy, proj_f, proj_pp, proj_pasp);


		//対応点を正規化(fx=fy=1, cx=cy=0とする)
		std::vector<cv::Point2d>norm_cam_pts, norm_proj_pts;
		for(int i = 0; i < cam_pts.size(); i++)
		{
			double norm_cam_x = (cam_pts[i].x - cam_pp.x) / cam_f;
			double norm_cam_y = (cam_pts[i].y - cam_pp.y) / cam_f;
			double norm_proj_x = (proj_pts[i].x - proj_pp.x) / proj_f;
			double norm_proj_y = (proj_pts[i].y - proj_pp.y) / proj_f;

			//std::cout << "(x, y) : (" << cam_pts[i].x << ", " << cam_pts[i].y  << ") ---> (" << norm_proj_x << ", " << norm_proj_y << ")" << std::endl;

			norm_cam_pts.push_back(cv::Point2d(norm_cam_x, norm_cam_y));
			norm_proj_pts.push_back(cv::Point2d(norm_proj_x, norm_proj_y));
		}

		//基礎行列の算出
		cv::Mat_<double> F;
		//findfundamentalMat( pt1, pt2, F行列を計算する手法, 点からエピポーラ線までの最大距離, Fの信頼度)
		if(norm_cam_pts.size() == 7)
			F = cv::findFundamentalMat(norm_cam_pts, norm_proj_pts,cv::FM_7POINT, 0.1, 0.99);
			//F = cv::findFundamentalMat(cam_pts, proj_pts,cv::FM_7POINT, 3.0, 0.99);
		else if(norm_cam_pts.size() == 8)
			F = cv::findFundamentalMat(norm_cam_pts, norm_proj_pts,cv::FM_8POINT, 0.1, 0.99);
			//F = cv::findFundamentalMat(cam_pts, proj_pts,cv::FM_8POINT, 3.0, 0.99);
		else
			F = cv::findFundamentalMat(norm_cam_pts, norm_proj_pts,cv::RANSAC, 0.1, 0.99);
			//F = cv::findFundamentalMat(cam_pts, proj_pts,cv::RANSAC, 3.0, 0.99);

		//基本行列の算出
//		cv::Mat_<double> Kc = camera.cam_K;
//		cv::Mat_<double> Kp = projector.cam_K;
//		cv::Mat_<double> E = Kc.t() * F * Kp;

		std::cout << "\nEssentiamMat1:\n" << F << std::endl;
		return F;
	}


	cv::Mat findEssentialMat2(){

		std::vector<cv::Point2d>norm_cam_pts, norm_proj_pts;
		for(int i = 0; i < cam_pts.size(); i++)
		{
			cv::Mat ip(3, 1, CV_64FC1);
			cv::Point2d p;
			//カメラの点
			ip.at<double>(0) = cam_pts[i].x;
			ip.at<double>(1) = cam_pts[i].y;
			ip.at<double>(2) = 1.0;
			ip = camera.cam_K.inv()*ip;
			p.x = ip.at<double>(0);
			p.y = ip.at<double>(1);
			norm_cam_pts.push_back(p);
			//プロジェクタの点
			ip.at<double>(0) = proj_pts[i].x;
			ip.at<double>(1) = proj_pts[i].y;
			ip.at<double>(2) = 1.0;
			ip = projector.cam_K.inv()*ip;
			p.x = ip.at<double>(0);
			p.y = ip.at<double>(1);
			norm_proj_pts.push_back(p);
		}

		//基礎行列の算出
		cv::Mat_<double> F;
		//findfundamentalMat( pt1, pt2, F行列を計算する手法, 点からエピポーラ線までの最大距離, Fの信頼度)
		if(norm_cam_pts.size() == 7)
			F = cv::findFundamentalMat(norm_cam_pts, norm_proj_pts,cv::FM_7POINT, 0.1, 0.99);
			//F = cv::findFundamentalMat(cam_pts, proj_pts,cv::FM_7POINT, 3.0, 0.99);
		else if(norm_cam_pts.size() == 8)
			F = cv::findFundamentalMat(norm_cam_pts, norm_proj_pts,cv::FM_8POINT, 0.1, 0.99);
			//F = cv::findFundamentalMat(cam_pts, proj_pts,cv::FM_8POINT, 3.0, 0.99);
		else
			F = cv::findFundamentalMat(norm_cam_pts, norm_proj_pts,cv::RANSAC, 0.1, 0.99);
			//F = cv::findFundamentalMat(cam_pts, proj_pts,cv::RANSAC, 3.0, 0.99);

		//基本行列の算出
//		cv::Mat_<double> Kc = camera.cam_K;
//		cv::Mat_<double> Kp = projector.cam_K;
//		cv::Mat_<double> E = Kc.t() * F * Kp;

		std::cout << "\nEssentiamMat2:\n" << F << std::endl;
		return F;
	}

	void findProCamPose(const cv::Mat& E, const cv::Mat& R, const cv::Mat& t)
	{
            cv::Mat R1 = cv::Mat::eye(3,3,CV_64F);
            cv::Mat R2 = cv::Mat::eye(3,3,CV_64F);
			cv::Mat t_ = cv::Mat::zeros(3,1,CV_64F);
			//[R1,t] [R1, -t] [R2, t], [R2, -t]の可能性がある
			decomposeEssentialMat(E, R1, R2, t_);

			std::cout << "\nR1:\n" << R1 << std::endl;
			std::cout << "R2:\n" << R2 << std::endl;
			std::cout << "t:\n" << t_ << std::endl;
	}

	int recoverPose( cv::InputArray E, cv::OutputArray _R, cv::OutputArray _t)
	{
		////カメラ
		//double cam_fx = camera.cam_K.at<double>(0,0);
		//double cam_fy = camera.cam_K.at<double>(1,1);
		//cv::Point2d cam_pp = cv::Point2d(camera.cam_K.at<double>(0,2), camera.cam_K.at<double>(1,2));
		////プロジェクタ
		//double proj_fx = projector.cam_K.at<double>(0,0);
		//double proj_fy = projector.cam_K.at<double>(1,1);
		//cv::Point2d proj_pp = cv::Point2d(projector.cam_K.at<double>(0,2), projector.cam_K.at<double>(1,2));

		//// 焦点距離とレンズ主点
  //      double cam_f, proj_f, cam_fovx, cam_fovy, proj_fovx, proj_fovy, cam_pasp, proj_pasp;
  //      cv::Point2d cam_pp, proj_pp;
		//cv::calibrationMatrixValues(camera.cam_K, cv::Size(camera.width, camera.height), 0.0, 0.0, cam_fovx, cam_fovy, cam_f, cam_pp, cam_pasp);
		//cv::calibrationMatrixValues(projector.cam_K, cv::Size(projector.width, projector.height), 0.0, 0.0, proj_fovx, proj_fovy, proj_f, proj_pp, proj_pasp);

		std::vector<cv::Point2d>norm_cam_pts, norm_proj_pts;
		for(int i = 0; i < cam_pts.size(); i++)
		{
			cv::Mat ip(3, 1, CV_64FC1);
			cv::Point2d p;
			//カメラの点
			ip.at<double>(0) = cam_pts[i].x;
			ip.at<double>(1) = cam_pts[i].y;
			ip.at<double>(2) = 1.0;
			ip = camera.cam_K.inv()*ip;
			p.x = ip.at<double>(0);
			p.y = ip.at<double>(1);
			norm_cam_pts.push_back(p);
			//プロジェクタの点
			ip.at<double>(0) = proj_pts[i].x;
			ip.at<double>(1) = proj_pts[i].y;
			ip.at<double>(2) = 1.0;
			ip = projector.cam_K.inv()*ip;
			p.x = ip.at<double>(0);
			p.y = ip.at<double>(1);
			norm_proj_pts.push_back(p);
		}

		cv::InputArray _points1 = norm_cam_pts;
		cv::InputArray _points2 = norm_proj_pts;

		//cv::InputArray _points1 = cam_pts;
		//cv::InputArray _points2 = proj_pts;

		cv::Mat points1, points2; //1:カメラ2:プロジェクタ
		_points1.getMat().copyTo(points1);
		_points2.getMat().copyTo(points2);

		int npoints = points1.checkVector(2);
		CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
								  points1.type() == points2.type());

		if (points1.channels() > 1)
		{
			points1 = points1.reshape(1, npoints);
			points2 = points2.reshape(1, npoints);
		}
		points1.convertTo(points1, CV_64F);
		points2.convertTo(points2, CV_64F);

		//points1.col(0) = (points1.col(0) - cam_pp.x) / cam_f;
		//points2.col(0) = (points2.col(0) - proj_pp.x) / proj_f;
		//points1.col(1) = (points1.col(1) - cam_pp.y) / cam_f;
		//points2.col(1) = (points2.col(1) - proj_pp.y) / proj_f;

		points1 = points1.t();
		points2 = points2.t();

		cv::Mat R1, R2, t;
		decomposeEssentialMat(E, R1, R2, t);
		cv::Mat P0 = cv::Mat::eye(3, 4, R1.type());
		cv::Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
		P1(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
		P2(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
		P3(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
		P4(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

		// Do the cheirality check.
		// Notice here a threshold dist is used to filter
		// out far away points (i.e. infinite points) since
		// there depth may vary between postive and negtive.
		double dist = 50.0;
		cv::Mat Q;
		triangulatePoints(P0, P1, points1, points2, Q);
		cv::Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
		Q.row(0) /= Q.row(3);
		Q.row(1) /= Q.row(3);
		Q.row(2) /= Q.row(3);
		Q.row(3) /= Q.row(3);
		mask1 = (Q.row(2) < dist) & mask1;
		Q = P1 * Q;
		mask1 = (Q.row(2) > 0) & mask1;
		mask1 = (Q.row(2) < dist) & mask1;

		triangulatePoints(P0, P2, points1, points2, Q);
		cv::Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
		Q.row(0) /= Q.row(3);
		Q.row(1) /= Q.row(3);
		Q.row(2) /= Q.row(3);
		Q.row(3) /= Q.row(3);
		mask2 = (Q.row(2) < dist) & mask2;
		Q = P2 * Q;
		mask2 = (Q.row(2) > 0) & mask2;
		mask2 = (Q.row(2) < dist) & mask2;

		triangulatePoints(P0, P3, points1, points2, Q);
		cv::Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
		Q.row(0) /= Q.row(3);
		Q.row(1) /= Q.row(3);
		Q.row(2) /= Q.row(3);
		Q.row(3) /= Q.row(3);
		mask3 = (Q.row(2) < dist) & mask3;
		Q = P3 * Q;
		mask3 = (Q.row(2) > 0) & mask3;
		mask3 = (Q.row(2) < dist) & mask3;

		triangulatePoints(P0, P4, points1, points2, Q);
		cv::Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
		Q.row(0) /= Q.row(3);
		Q.row(1) /= Q.row(3);
		Q.row(2) /= Q.row(3);
		Q.row(3) /= Q.row(3);
		mask4 = (Q.row(2) < dist) & mask4;
		Q = P4 * Q;
		mask4 = (Q.row(2) > 0) & mask4;
		mask4 = (Q.row(2) < dist) & mask4;

		mask1 = mask1.t();
		mask2 = mask2.t();
		mask3 = mask3.t();
		mask4 = mask4.t();

		// If _mask is given, then use it to filter outliers.
		//cv::InputOutputArray _mask;

		//if (!_mask.empty())
		//{
		//	cv::Mat mask = _mask.getMat();
		//	CV_Assert(mask.size() == mask1.size());
		//	bitwise_and(mask, mask1, mask1);
		//	bitwise_and(mask, mask2, mask2);
		//	bitwise_and(mask, mask3, mask3);
		//	bitwise_and(mask, mask4, mask4);
		//}
		//if (_mask.empty() && _mask.needed())
		//{
		//	_mask.create(mask1.size(), CV_8U);
		//}

		CV_Assert(_R.needed() && _t.needed());
		_R.create(3, 3, R1.type());
		_t.create(3, 1, t.type());

		int good1 = countNonZero(mask1);
		int good2 = countNonZero(mask2);
		int good3 = countNonZero(mask3);
		int good4 = countNonZero(mask4);

		if (good1 >= good2 && good1 >= good3 && good1 >= good4)
		{
			R1.copyTo(_R);
			t.copyTo(_t);
			//if (_mask.needed()) mask1.copyTo(_mask);
			return good1;
		}
		else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
		{
			R2.copyTo(_R);
			t.copyTo(_t);
			//if (_mask.needed()) mask2.copyTo(_mask);
			return good2;
		}
		else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
		{
			t = -t;
			R1.copyTo(_R);
			t.copyTo(_t);
			//if (_mask.needed()) mask3.copyTo(_mask);
			return good3;
		}
		else
		{
			t = -t;
			R2.copyTo(_R);
			t.copyTo(_t);
			//if (_mask.needed()) mask4.copyTo(_mask);
			return good4;
		}
	}


	//基本行列からR1,R2,tに分解(cv3.0.0より引用)
	void decomposeEssentialMat( cv::InputArray _E, cv::OutputArray _R1, cv::OutputArray _R2, cv::OutputArray _t )
	{
		cv::Mat E = _E.getMat().reshape(1, 3);
		CV_Assert(E.cols == 3 && E.rows == 3);


		cv::Mat D, U, Vt;
		cv::SVD::compute(E, D, U, Vt);

		if (determinant(U) < 0) U *= -1.;
		if (determinant(Vt) < 0) Vt *= -1.;

		cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
		W.convertTo(W, E.type());

		cv::Mat R1, R2, t;
		R1 = U * W * Vt;
		R2 = U * W.t() * Vt;
		t = U.col(2) * 1.0;

		R1.copyTo(_R1);
		R2.copyTo(_R2);
		t.copyTo(_t);
	}

};
	
#endif