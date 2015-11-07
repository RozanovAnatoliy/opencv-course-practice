#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat edges, dist, integ;
	Mat image = imread("../road.png");

	double k = 0.8;
	int maskSize = 5;
	double threshold1 = 100.0, threshold2 = 200.0;

	Canny(image, edges, threshold1, threshold2);
	bitwise_not(edges, edges);
	distanceTransform(edges, dist, CV_DIST_L1, maskSize);
	integral(image, integ);

	Mat res(image.rows, image.cols, CV_8UC3);
	for (int i = 0; i < image.rows; i++)
	for (int j = 0; j < image.cols; j++)	
	 {
		maskSize = (int)(k * dist.at<float>(i, j));
		
		int x1 = max(i - maskSize / 2, 0);
		int y1 = max(j - maskSize / 2, 0);
		int x2 = min(i + maskSize / 2, image.rows - 1);
		int y2 = min(j + maskSize / 2, image.cols - 1);

		if (x1 == x2 || y1 == y2) res.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
		else res.at<Vec3b>(i, j) = (1.0 / ((x2 - x1 + 1) * (y2 - y1 + 1))) * (integ.at<Vec3i>(x2 + 1, y2 + 1) - 
			integ.at<Vec3i>(x1, y2 + 1) - integ.at<Vec3i>(x2 + 1, y1) + integ.at<Vec3i>(x1, y1));
	}
	
	imshow("image:", image);
	imshow("invEdges:", edges);	
	imshow("Result:", res);
	waitKey(0);

	return 0;
}
