#include <cv.h>
#include <highgui.h>

using namespace cv;

int main(int argc, char** argv)
{
    Mat edges, dist, integ;
	Mat image = imread("../road.png");

	double k = 0.8;
	int maskSize = 5;
	double threshold1 = 100.0, threshold2 = 300.0;

	Canny(image, edges, threshold1, threshold2, 3, false);
	bitwise_not(edges, edges);
	distanceTransform(edges, dist, CV_DIST_L1, maskSize);
	integral(image, integ);

	Mat res(image.rows, image.cols, CV_8UC3);
	for (int i = 0; i < image.rows; i++)
	for (int j = 0; j < image.cols; j++) {
		maskSize = (int)(k * dist.at<float>(i, j));
		
		int x1 = i - maskSize / 2;
		int y1 = j - maskSize / 2;
		int x2 = i + maskSize / 2;
		int y2 = j + maskSize / 2;

		if (x1 < 0) x1 = 0;
		if (y1 < 0) y1 = 0;
		if (x2 > image.rows) x2 = image.rows;
		if (y2 > image.cols) y2 = image.cols;
		if (x1 == x2 || y1 == y2) res.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
		else res.at<Vec3b>(i, j) = (1.0 / ((x2 - x1) * (y2 - y1))) * (integ.at<Vec3i>(x2, y2) - integ.at<Vec3i>(x1, y2) -
			integ.at<Vec3i>(x2, y1) + integ.at<Vec3i>(x1, y1));
	}
	
	imshow("image:", image);
	imshow("invEdges:", edges);	
	imshow("Result:", res);
	waitKey(0);

	return 0;
}
