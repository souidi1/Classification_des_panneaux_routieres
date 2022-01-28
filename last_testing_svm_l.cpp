#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include<iostream>
using namespace cv;
using namespace cv::ml;
using namespace std;
using std::copy;
using std::string;

float testing_matrice[1827][256];


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
	int a;
	float tableau[256] = { 0 };
	for (int i = 0; i < d.rows; i++) {
		for (int j = 0; j < d.cols; j++) {
			for (int k = 0; k < 256; k++) {
				if (k == (float)d.at<uchar>(i, j)) tableau[k] += 1;;
			}
			a = (int)d.at<uchar>(i, j);
			tableau[a] += 1;
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
	float tp = 0, fn = 0, fp = 0, tn = 0;
	float* tab;
	int n = 0;
	clock_t t1, t2;
	t1 = clock();
	Ptr<SVM> svm = SVM::create();
	//svm = SVM::load("Training_SVM_Linn.xml"); 
	
	
	svm = SVM::load("TrainingWithSVM_L1.xml");
	Mat Resized;
	Mat greyMat;
	int ColumnOfNewImage = 30;
	int RowsOfNewImage = 30;

	for (int i = 0; i < 1827; i++) {
		if (i < 1374) {
			string c = to_string(i + 1);
			string path("Testing\\pos\\");
			string path1(".ppm");
			string  positivepath = concTwoStrings(concTwoStrings(path, c), path1);
			img = imread(positivepath);
			resize(img, Resized, Size(ColumnOfNewImage, RowsOfNewImage));
			cv::cvtColor(Resized, greyMat, cv::COLOR_BGR2GRAY);
			tab = LBPtoVect(LBP(greyMat));
			for (int j = 0; j < 256; j++) {
				testing_matrice[i][j] = tab[j];
			}
		}
		else {
			string c = to_string(n + 1);
			string path("Testing\\neg_jpg\\");
			string path1(".jpg");
			string negativepath = concTwoStrings(concTwoStrings(path, c), path1);
			img = imread(negativepath);
			resize(img, Resized, Size(ColumnOfNewImage, RowsOfNewImage));
			cv::cvtColor(Resized, greyMat, cv::COLOR_BGR2GRAY);
			tab = LBPtoVect(LBP(greyMat));
			for (int j = 0; j < 256; j++) {
				testing_matrice[i][j] = tab[j];
			}
		}
	}

	Mat testing_data = Mat(1827, 256, CV_32FC1, testing_matrice);
	for (int i = 0; i < 1827; ++i)
	{
		//  the type of prediction must be mat
		float response = svm->predict(testing_data.row(i)); // predict
		if (i < 1374) {
			if (response == 1) // Classify the results
			{
				tp++;
			}
			else if (response == -1)
			{
				fp++;
			}
		}
		else {
			if (response == 1) // Classify the results
			{
				fn++;
			}
			else if (response == -1)
			{
				tn++;
			}
		}



	}
	t2 = clock();
	float temps = (float)(t2 - t1) / CLOCKS_PER_SEC;
	cout << "Temps execution :s" << temps << "s" << endl;
	cout << "Temps execution seul image :" << temps / 1827 << "s" << endl;
	cout << "images for testing total : 1827" << endl;
	cout << "\n\n==============================\n";
	cout << "images for testing  positive : 1374" << endl;
	cout << "True positive  :" << tp << endl;
	cout << "\n\n==============================\n";
	cout << "images for testing  negative : 453" << endl;
	cout << "True negative :" << tn << endl;
	cout << "\n\n==============================\n";
	cout << "false negative :" << fn << endl;
	cout << "false positive :" << fp << endl;
	cout << "\n\n==============================\n";
	cout << "Precision ="<<tp/(tp+fp)<<endl;
	cout << "Exactitude =" << (tp + tn) / (tp + tn + fp+ fn) << endl;
	cout << "Rappel =" << tp / (tp + fn) << endl;


	//cout << svm->getKernelType();
	return 2;
}

