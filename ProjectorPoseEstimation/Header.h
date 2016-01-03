#ifndef HEADER_H
#define HEADER_H

#pragma once

#include <Windows.h>
#include <opencv2\opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp> // SIFTまたはSURFを使う場合は必要
//処理時間計測用
#include <time.h>
//PCL
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <pcl/visualization/pcl_visualizer.h>

#define CAMERA_WIDTH 1920
#define CAMERA_HEIGHT 1080

#define PROJECTOR_WIDTH 1280
#define PROJECTOR_HEIGHT 800

#define DISP_NUMBER (1)


#endif