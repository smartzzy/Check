#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/flann/miniflann.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat srcImage = imread("G:/opencv/识别已知物/Check/x64/Debug/1.jpg",1);
	imshow("原始图", srcImage);
	Mat grayImage;
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);

	OrbFeatureDetector featureDetector;
	vector<KeyPoint> keyPoint;
	Mat descriptors;

	featureDetector.detect(grayImage, keyPoint);

	OrbDescriptorExtractor featrueExtractor;
	featrueExtractor.compute(grayImage, keyPoint, descriptors);

	flann::Index flannIndex(descriptors,flann::LshIndexParams(12, 20, 2)
		,cvflann::FLANN_DIST_HAMMING);

	VideoCapture cap(0);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 360);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 900);

	unsigned int frameCount = 0;

	while (1)
	{
		double time0 = static_cast<double>(getTickCount());//记录起始时间
		Mat captureImage, captureImage_gray;
		cap >> captureImage;
		imshow("视频", captureImage);
		if (captureImage.empty())
			continue;
		cvtColor(captureImage, captureImage_gray, CV_BGR2GRAY);

		//检测SIFT关键点并提取测试图像中的描述符
		vector<KeyPoint> captureKeyPoints;
		Mat captureDescription;
		//调用detect函数检测出特征关键点，保存在vetect容器中
		featureDetector.detect(captureImage_gray, captureKeyPoints);

		//计算描述符
		featrueExtractor.compute(captureImage_gray, captureKeyPoints,
			captureDescription);

		//10.匹配和测试描述符，获取两个最邻近的描述符
		Mat matchIndex(captureDescription.rows, 2, CV_32SC1),
			matchDistance(captureDescription.rows, 2, CV_32FC1);
		flannIndex.knnSearch(captureDescription, matchIndex,
			matchDistance, 2, flann::SearchParams());

		//根据劳氏算法选出优秀的匹配
		vector<DMatch> goodMatchs;
		for (int i = 0; i < matchDistance.rows; i++)
		{
			if (matchDistance.at<float>(i, 0) < 0.6 *
				matchDistance.at<float>(i, 1))
			{
				DMatch dmatches(i, matchIndex.at<int>(i, 0),
					matchDistance.at<float>(i, 0));
				goodMatchs.push_back(dmatches);
			}
		}

		Mat resultImage;
		drawMatches(captureImage, captureKeyPoints, srcImage, keyPoint, goodMatchs, resultImage);
		imshow("匹配窗口", resultImage);

		cout << ">帧率=" << getTickFrequency() / (getTickCount() - time0) << endl;
		if (char(waitKey(1) == 27)) break;
	}
	return 0;
}