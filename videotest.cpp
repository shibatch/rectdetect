// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

static int fourcc(const char *s) {
  return (((uint32_t)s[0]) << 0) | (((uint32_t)s[1]) << 8) | (((uint32_t)s[2]) << 16) | (((uint32_t)s[3]) << 24);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage : %s [input video file] [output video file]\n", argv[0]);
    exit(-1);
  }

  VideoCapture *cap = NULL;
  if (argc < 2) {
    cap = new VideoCapture(0);
  } else if (strncmp(argv[1], "cam:", 4) != 0) {
    cap = new VideoCapture(argv[1]);
  } else {
    int n = 0, w = 0, h = 0;
    sscanf(argv[1], "cam:%d,%d,%d", &n, &w, &h);
    cap = new VideoCapture(n);
    if (cap->isOpened() && w != 0 && h != 0) {
      cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
      cap->set(CV_CAP_PROP_FRAME_HEIGHT, h);
    }
  }
  if (!cap->isOpened()) {
    fprintf(stderr, "Cannot open %s\n", argv[1]);
    exit(-1);
  }
  int iw = cap->get(CV_CAP_PROP_FRAME_WIDTH);
  int ih = cap->get(CV_CAP_PROP_FRAME_HEIGHT);
  printf("Resolution : %d x %d\n", iw, ih);

  //

  VideoWriter *writer = NULL;

  if (argc < 3) {
    namedWindow("videotest", WINDOW_AUTOSIZE );
    printf("Press ENTER on the window to exit\n");
  } else {
    writer = new VideoWriter(argv[2], fourcc("PIM1"), 30, cvSize(iw, ih), true);
    if (!writer->isOpened()) {
      fprintf(stderr, "Cannot open %s\n", argv[2]);
      exit(-1);
    }
  }

  //
  
  cap->grab();

  for(;;) {
    Mat img;
    if (!cap->retrieve(img, 0)) break;
    cap->grab();
    
    if (writer != NULL) {
      writer->write(img);
    } else {
      imshow( "videotest", img );
      int key = waitKey(33) & 0xff;
      if (key == 27 || key == 13) break;
    }
  }

  if (writer != NULL) delete writer;
  delete cap;
}
