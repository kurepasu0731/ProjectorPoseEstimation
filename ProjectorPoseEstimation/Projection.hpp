#pragma once

#include <Windows.h>
#include <opencv2\opencv.hpp>

using namespace std;

namespace Projection{

	typedef struct disp_prop{
		int index;
		int x,y,width,height;
	} Disp_Prop;

	int dispCount=-1;
	vector<Disp_Prop> Disps_Prop;

	//ディスプレイの情報入手
	BOOL CALLBACK DispEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData ) {
		Disp_Prop di;
		di.index = dispCount++;
		di.x = lprcMonitor->left;
		di.y = lprcMonitor->top;
		di.width = lprcMonitor->right - di.x;
		di.height = lprcMonitor->bottom - di.y;
		Disps_Prop.push_back(di);

		return TRUE; // TRUEは探索継続，FALSEで終了
	}

	//ディスプレイ検出
	void SearchDisplay(void) {
		// 一度だけ実行する
		if (dispCount == -1) {
			dispCount = 0;
			Disps_Prop = vector<Disp_Prop>();
			EnumDisplayMonitors(NULL, NULL, DispEnumProc, 0);
			Sleep(200);
		}
	}

	//プロジェクション
	void MySetFullScrean(const int num, const char *windowname){

		HWND windowHandle = ::FindWindowA(NULL, windowname);

		SearchDisplay();

		if (NULL != windowHandle) {

			//-ウィンドウスタイル変更（メニューバーなし、最前面）-
			SetWindowLongPtr(windowHandle,  GWL_STYLE, WS_POPUP);
			SetWindowLongPtr(windowHandle, GWL_EXSTYLE, WS_EX_TOPMOST);

			//-最大化する-
			ShowWindow(windowHandle, SW_MAXIMIZE);
			cv::setWindowProperty(windowname, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN );

			//-ディスプレイを指定-
			Disp_Prop di = Disps_Prop[num];

			//-クライアント領域をディスプレーに合わせる-
			SetWindowPos(windowHandle, NULL, di.x, di.y, di.width, di.height, SWP_FRAMECHANGED | SWP_NOZORDER);
		}
	}
}