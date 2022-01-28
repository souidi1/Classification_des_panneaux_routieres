#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <string>
using namespace cv;
using namespace cv::ml;
using namespace std;
using std::copy;
using std::string;

float training_matrice[13848][256];
//float  a[13848][256];
int label[13848];


Mat LBP(Mat img_src) {

	Mat des = Mat::zeros(img_src.rows - 2, img_src.cols - 2, 0);

	for (int i = 1; i < img_src.rows - 1; i++)
	{
		for (int j = 1; j < img_src.cols - 1; j++) {
			int code = 0;
			int center = (int)img_src.at<uchar>(i, j);
			code += ((int)img_src.at<uchar>(i - 1, j - 1) >= center) ? 128 : 0;
			code += ((int)img_src.at<uchar>(i - 1, j) >= center) ? 64 : 0;
			code += ((int)img_src.at<uchar>(i - 1, j + 1) >= center) ? 32 : 0;
			code += ((int)img_src.at<uchar>(i, j + 1) >= center) ? 16 : 0;
			code += ((int)img_src.at<uchar>(i + 1, j + 1) >= center) ? 8 : 0;
			code += ((int)img_src.at<uchar>(i + 1, j) >= center) ? 4 : 0;
			code += ((int)img_src.at<uchar>(i + 1, j - 1) >= center) ? 2 : 0;
			code += ((int)img_src.at<uchar>(i, j - 1) >= center) ? 1 : 0;
			des.at<uchar>(i - 1, j - 1) = code;
		}
	}
	return des;
}

float* LBPtoVect(Mat d) {
	int indice;
	float tableau[256] = { 0 };
	for (int i = 0; i < d.rows; i++) {
		for (int j = 0; j < d.cols; j++) {
			for (int k = 0; k < 256; k++) {
				if (k == (float)d.at<uchar>(i, j)) tableau[k] += 1;;
			}
			indice = (int)d.at<uchar>(i, j);
			tableau[indice] += 1;
		}
	}
	return tableau;
}

string concTwoStrings(const string& s1, const string& s2)
{
	return s1 + s2;
}


int main()
{
	Mat img;
	Mat Resized;
	//Mat greyMat;
	int ColumnOfNewImage = 30;
	int RowsOfNewImage = 30;
	float* tab;
	int n = 0;
	for (int i = 0; i < 13848; i++) {

		if (i < 12630) {
			string c = to_string(i + 1);
			string path("Training\\pos\\");
			string path1(".ppm");
			string  positivepath = concTwoStrings(concTwoStrings(path, c), path1);
			img = imread(positivepath,0);
			resize(img, Resized, Size(ColumnOfNewImage, RowsOfNewImage));
			//cv::cvtColor(Resized, greyMat, cv::COLOR_BGR2GRAY);
		}
		else {
			string c = to_string(n + 1);
			string path("Training\\neg\\New folder\\");
			string path1(".jpg");
			string negativepath = concTwoStrings(concTwoStrings(path, c), path1);
			img = imread(negativepath,0);
			resize(img, Resized, Size(ColumnOfNewImage, RowsOfNewImage));
			//cv::cvtColor(Resized, greyMat, cv::COLOR_BGR2GRAY);
			n++;
		}
		tab = LBPtoVect(LBP(Resized));
		for (int j = 0; j < 256; j++) {
			training_matrice[i][j] = tab[j];
		}
	}

	for (int i = 0; i < 13848; i++) {
		if (i < 12630)
		{
			label[i] = 1;
		}
		else {
			label[i] = -1;
		}
	}

	

	Mat labelsmat(13848, 1, CV_32SC1, label);
	Mat trainingDateMat(13848, 256, CV_32FC1, training_matrice);
	// CV_32FC1 float type, 32 bits, single channel

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC); // Set the classifier type
	svm->setKernel(SVM::LINEAR); // Set kernel function
	Ptr <TrainData> tData = TrainData::create(trainingDateMat, ROW_SAMPLE, labelsmat); // The first is
	svm->train(tData);
	svm->save("TrainingWithSVM_L1.xml");
	

	return 1;
}

