//如果要生成python库文件，需要在属性->常规->配置类型中将其改为dll，生成后在release文件夹下将文件名改为pyd即可

#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<Eigen/Dense>
#include<Eigen/SVD>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix.h"
#include "pybind_matrix.h"
#include <opencv2/imgproc/imgproc.hpp>


namespace py = pybind11;
using namespace std;
//using namespace cv;
//using namespace Eigen;


// ----------------
// regular C++ code
// ----------------



void show(vector<cv::Mat_<float>> img, string name) {
	for (int i = 0; i < img.size(); i++) {
		cv::Mat show(img[i].rows, img[i].cols, CV_8UC1);
		convertScaleAbs(img[i], show, 255.0);
		imshow(name.append(to_string(i)), show);
	}
}

//获取图像掩膜
cv::Mat_<float> get_mask(cv::Mat img1, cv::Mat img2) {
	cv::Mat a = img1;
	img1.rows;
	cv::Mat result(img1.rows, img1.cols, CV_32F);

	//存放每行每列的极值,0行最小值，1行最大值
	cv::Mat mm = cv::Mat::zeros(2, img1.cols, CV_32F);
	for (int i = 0; i < mm.rows; i++) {
		for (int j = 0; j < mm.cols; j++) {
			if (i == 0) {
				mm.at<float>(i, j) = 100000;
			}
		}
	}

	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			if ((img1.at<cv::Vec3b>(i, j)[0] + img1.at<cv::Vec3b>(i, j)[1] + img1.at<cv::Vec3b>(i, j)[2]) > 15 &&
				(img2.at<cv::Vec3b>(i, j)[0] + img2.at<cv::Vec3b>(i, j)[1] + img2.at<cv::Vec3b>(i, j)[2]) > 15) {
				result.at<float>(i, j) = float(i);
				if (mm.at<float>(0, j) > i) {
					mm.at<float>(0, j) = i;
				}
				else if (mm.at<float>(1, j) < i) {
					mm.at<float>(1, j) = i;
				}
			}
			else
			{
				result.at<float>(i, j) = 0.0;
			}
		}
	}


	//对掩膜施加权重
	cv::Mat final(img1.rows, img1.cols, CV_32F);
	for (int col = 0; col < result.cols; col++) {
		float min = mm.at<float>(0, col);
		float max = mm.at<float>(1, col);
		for (int row = 0; row < result.rows; row++) {
			if (result.at<float>(row, col) > 0.0) {
				final.at<float>(row, col) = (result.at<float>(row, col) - min) / (max - min);
			}
			else if ((result.at<float>(row, col) == 0.0) &&
				((img1.at<cv::Vec3b>(row, col)[0] + img1.at<cv::Vec3b>(row, col)[1] + img1.at<cv::Vec3b>(row, col)[2]) > 15 &&
					(img2.at<cv::Vec3b>(row, col)[0] + img2.at<cv::Vec3b>(row, col)[1] + img2.at<cv::Vec3b>(row, col)[2]) < 15)) {
				final.at<float>(row, col) = 0.0;
			}
			else
			{
				final.at<float>(row, col) = 1.0;
			}

		}
	}
	return final;
}


//构建高斯金字塔
//大------>>>>小
vector<cv::Mat_<float>> gaussianPyramid(cv::Mat img, int levels) {
	//imshow("img", img);
	vector<cv::Mat_<float>> _gaussianPyramid;
	//第一层为源图
	_gaussianPyramid.push_back(img);
	cv::Mat currentImage = img;
	for (int i = 1; i < levels; i++) {
		cv::Mat downsampleImage;
		cv::pyrDown(currentImage, downsampleImage);
		_gaussianPyramid.push_back(downsampleImage);
		currentImage = downsampleImage;
	}
	return _gaussianPyramid;
}


//构建laplace金字塔,输入为高斯金字塔,大小顺序与高斯金字塔相反
//小------->>>>>>大
vector<cv::Mat_<float>>laplacianPyramid(vector<cv::Mat_<float>> gaussianPyramid)
{
	int levels = gaussianPyramid.size();
	vector<cv::Mat_<float>> _laplacianPyramid;
	//第一层为高斯金字塔最后一层
	_laplacianPyramid.push_back(gaussianPyramid[levels - 1]);
	for (int i = levels - 2; i >= 0; i--) {
		cv::Mat upsampleImage;
		cv::pyrUp(gaussianPyramid[i + 1], upsampleImage, gaussianPyramid[i].size());
		cv::Mat currentImage = gaussianPyramid[i] - upsampleImage;
		_laplacianPyramid.push_back(currentImage);
	}
	return _laplacianPyramid;
}


//对金字塔进行融合
//小--->>>>大
vector<cv::Mat_<float>> blendPyramid(vector<cv::Mat_<float>> pyrA, vector < cv::Mat_<float>> pyrB, vector<cv::Mat_<float>> pyrMask) {
	int levels = pyrA.size();
	vector<cv::Mat_<float>> blendedPyramid;
	for (int i = 0; i < levels; i++) {
		cv::Mat blendedImage(pyrA[i].rows, pyrA[i].cols, CV_32F);
		for (int row = 0; row < pyrMask[levels - 1 - i].rows; row++) {
			for (int col = 0; col < pyrMask[levels - 1 - i].cols; col++) {
				blendedImage.at<float>(row, col) = pyrA[i].at<float>(row, col) * (1.0 - pyrMask[levels - 1 - i].at<float>(row, col)) +
					pyrB[i].at<float>(row, col) * (pyrMask[levels - 1 - i].at<float>(row, col));
			}
		}
		blendedPyramid.push_back(blendedImage);
	}
	return blendedPyramid;
}

//合并图像金字塔
cv::Mat collapsePyramid(vector<cv::Mat_<float>> blendedPyramid) {
	int levels = blendedPyramid.size();
	cv::Mat currentImage = blendedPyramid[0];
	for (int i = 1; i < levels; i++) {
		cv::pyrUp(currentImage, currentImage, blendedPyramid[i].size());
		currentImage += blendedPyramid[i];
	}
	cv::Mat blendedImage;
	convertScaleAbs(currentImage, blendedImage, 255.0);
	return blendedImage;
}

//将opencv格式转回原matrix的格式
Matrix<double> transfer_to_matrix(cv::Mat a) {
    std::vector<size_t> ret(a.channels());
    ret[0] = a.rows;
    ret[1] = a.cols;
    ret[2] = a.channels();
    Matrix<double> matrix(ret);
    for (int channel = 0; channel < a.channels(); channel++) {
        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < a.cols; col++) {
                matrix[channel * (a.rows * a.cols) + row * a.cols + col] = a.at<cv::Vec3b>(row, col)[channel];
            }
        }
    }
    return matrix;
}

Matrix<uint> mul(const Matrix<double>& A, const Matrix<double>& B)
{
    if (A.shape() != B.shape())
        throw std::length_error("Matrix 'A' and 'B' are inconsistent");
    if (A.shape()[2] != 3 || B.shape()[2] != 3)
        throw std::length_error("the channel is not 3!");
    cv::Mat ori_img1 = cv::Mat::zeros(1, A.shape()[0]*A.shape()[1], CV_8UC3);
    cv::Mat ori_img2 = cv::Mat::zeros(1, A.shape()[0] * A.shape()[1], CV_8UC3);
    Matrix<uint> ret(A.shape());
    for (int i = 0,j=0; i < A.size(); i+=3,j++)
    {
		ori_img1.at<cv::Vec3b>(0, j) = cv::Vec3b(uint(A[i]), uint(A[i + 1]), uint(A[i + 2]));
    }
    for (int i = 0, j = 0; i < B.size(); i += 3, j++)
    {
		ori_img2.at<cv::Vec3b>(0, j) = cv::Vec3b(uint(B[i]), uint(B[i + 1]), uint(B[i + 2]));
    }
	cout << "transfer to cv success" << endl;
	cv::Mat img1 = ori_img1.reshape(0, A.shape()[0]);
	cv::Mat img2 = ori_img2.reshape(0, A.shape()[0]);
	cout << "reshape to 2 dimendsion success" << endl;

	vector<cv::Mat> color(3);
	vector<cv::Mat> img1_channels;
	split(img1, img1_channels);
	vector<cv::Mat> img2_channels;
	split(img2, img2_channels);

	int levels = floor(log2(min(img1.rows, img1.cols)));
	cv::Mat mask = get_mask(img1, img2);
	vector<cv::Mat_<float>> mask_pyramid = gaussianPyramid(mask, levels);
	for (int i = 0; i < img1_channels.size(); i++) {
		cv::Mat img1_current_channel = img1_channels.at(i);
		cv::Mat img2_current_channel = img2_channels.at(i);
		cv::Mat A, B;
		img1_current_channel.convertTo(A, CV_32F, 1.0 / 255.0);
		img2_current_channel.convertTo(B, CV_32F, 1.0 / 255.0);
		vector<cv::Mat_<float>> img1_pyramid = gaussianPyramid(A, levels);
		vector<cv::Mat_<float>> img1_lap_pyramid = laplacianPyramid(img1_pyramid);
		vector<cv::Mat_<float>> img2_pyramid = gaussianPyramid(B, levels);
		vector<cv::Mat_<float>> img2_lap_pyramid = laplacianPyramid(img2_pyramid);
		vector<cv::Mat_<float>> result_pyramid = blendPyramid(img1_lap_pyramid, img2_lap_pyramid, mask_pyramid);
		cv::Mat result = collapsePyramid(result_pyramid);
		color[i] = result;
	}
	cv::Mat output(img1.size(), CV_8UC3);
	merge(color, output);
	cout << "mbb process success..." << endl;
    cv::Mat new_img1 = output.reshape(0,1);
    cout << "reshape back success ... " << endl;
    for (int i = 0, j = 0; i < A.size(); i += 3, j++) {
        ret[i] = output.at<cv::Vec3b>(0, j)[0];
        ret[i+1] = output.at<cv::Vec3b>(0, j)[1];
        ret[i+2] = output.at<cv::Vec3b>(0, j)[2];
    }
    cout << "excute all success..." << endl;
    return ret;
}




// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(example, m)
{
    m.doc() = "pybind11 multiple band blending";

    m.def("mul", &mul);
}

