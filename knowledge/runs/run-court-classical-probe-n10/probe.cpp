// Read-only probe used to (a) record the OpenCV build configuration and
// (b) verify that the frame OpenCV decodes from the AVI matches the source PNG.
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;
  std::cout << "--- build information ---" << std::endl;
  std::cout << cv::getBuildInformation() << std::endl;
  std::cout << "--- end build information ---" << std::endl;

  if (argc < 2)
  {
    return 0;
  }

  cv::VideoCapture vc(argv[1]);
  if (!vc.isOpened())
  {
    std::cerr << "Cannot open " << argv[1] << std::endl;
    return 1;
  }
  std::cout << "frames=" << vc.get(cv::CAP_PROP_FRAME_COUNT)
            << " width=" << vc.get(cv::CAP_PROP_FRAME_WIDTH)
            << " height=" << vc.get(cv::CAP_PROP_FRAME_HEIGHT)
            << " fps=" << vc.get(cv::CAP_PROP_FPS) << std::endl;

  int frameIndex = int(vc.get(cv::CAP_PROP_FRAME_COUNT)) / 2;
  vc.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
  cv::Mat frame;
  if (!vc.read(frame))
  {
    std::cerr << "Failed to read frame " << frameIndex << std::endl;
    return 2;
  }
  cv::Scalar mean = cv::mean(frame);
  std::cout << "frame_index=" << frameIndex
            << " rows=" << frame.rows << " cols=" << frame.cols
            << " type=" << frame.type() << " channels=" << frame.channels()
            << " contiguous=" << frame.isContinuous()
            << " mean=" << mean[0] << "," << mean[1] << "," << mean[2] << std::endl;

  if (argc >= 3)
  {
    if (!cv::imwrite(argv[2], frame))
    {
      std::cerr << "imwrite failed" << std::endl;
      return 3;
    }
    std::cout << "wrote=" << argv[2] << std::endl;
  }
  return 0;
}
