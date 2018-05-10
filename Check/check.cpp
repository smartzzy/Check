#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/flann/miniflann.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat srcImage = imread("G:/opencv/ʶ����֪��/Check/x64/Debug/1.jpg",1);
	imshow("ԭʼͼ", srcImage);
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
		double time0 = static_cast<double>(getTickCount());//��¼��ʼʱ��
		Mat captureImage, captureImage_gray;
		cap >> captureImage;
		imshow("��Ƶ", captureImage);
		if (captureImage.empty())
			continue;
		cvtColor(captureImage, captureImage_gray, CV_BGR2GRAY);

		//���SIFT�ؼ��㲢��ȡ����ͼ���е�������
		vector<KeyPoint> captureKeyPoints;
		Mat captureDescription;
		//����detect�������������ؼ��㣬������vetect������
		featureDetector.detect(captureImage_gray, captureKeyPoints);

		//����������
		featrueExtractor.compute(captureImage_gray, captureKeyPoints,
			captureDescription);

		//10.ƥ��Ͳ�������������ȡ�������ڽ���������
		Mat matchIndex(captureDescription.rows, 2, CV_32SC1),
			matchDistance(captureDescription.rows, 2, CV_32FC1);
		flannIndex.knnSearch(captureDescription, matchIndex,
			matchDistance, 2, flann::SearchParams());

		//���������㷨ѡ�������ƥ��
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
		imshow("ƥ�䴰��", resultImage);

		cout << ">֡��=" << getTickFrequency() / (getTickCount() - time0) << endl;
		if (char(waitKey(1) == 27)) break;
	}
	return 0;
}