// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <time.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include "vec234.h"

#include "helper.h"
#include "oclhelper.h"

#include "oclimgutil.h"
#include "oclpolyline.h"
#include "oclrect.h"

static void showRect(rect_t rect, int r, int g, int b, int thickness, Mat &img) {
  for(int i=0;i<4;i++) {
    line(img, cvPoint(rect.c2[i].a[0], rect.c2[i].a[1]), cvPoint(rect.c2[(i+1)%4].a[0], rect.c2[(i+1)%4].a[1]), Scalar(r, g, b), thickness, 8, 0);
  }

  line(img,
       cvPoint(rect.c2[0].a[0], rect.c2[0].a[1]),
       cvPoint(rect.c2[2].a[0], rect.c2[2].a[1]), Scalar(r, g, b), 1, 8, 0);

  line(img,
       cvPoint(rect.c2[1].a[0], rect.c2[1].a[1]),
       cvPoint(rect.c2[3].a[0], rect.c2[3].a[1]), Scalar(r, g, b), 1, 8, 0);
}

static int fourcc(const char *s) {
  return (((uint32_t)s[0]) << 0) | (((uint32_t)s[1]) << 8) | (((uint32_t)s[2]) << 16) | (((uint32_t)s[3]) << 24);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage : %s [device number] [input video file] [output video file] [AOV]\n", argv[0]);
    fprintf(stderr, "By specifying cam:<cam id>,<width>,<height> as an input, a camera can be used.\n");
    fprintf(stderr, "Output is displayed on an window by specifying - to the output.\n");
    fprintf(stderr, "Example command line : vidrect 0 cam:0,1280,720 - 72\n");
    fprintf(stderr, "\nAvailable OpenCL Devices :\n");
    simpleGetDevice(-1);
    exit(-1);
  }

  //

  VideoCapture *cap = NULL;
  if (argc < 3) {
    cap = new VideoCapture(0);
    if (!cap->isOpened()) {
      fprintf(stderr, "Cannot open camera 0\n");
      exit(-1);
    }
  } else if (strncmp(argv[2], "cam:", 4) != 0) {
    cap = new VideoCapture(argv[2]);
    if (!cap->isOpened()) {
      fprintf(stderr, "Cannot open %s\n", argv[2]);
      exit(-1);
    }
  } else {
    int n = 0, w = 0, h = 0;
    sscanf(argv[2], "cam:%d,%d,%d", &n, &w, &h);
    cap = new VideoCapture(n);
    if (cap->isOpened() && w != 0 && h != 0) {
      cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
      cap->set(CV_CAP_PROP_FRAME_HEIGHT, h);
    }
    if (!cap->isOpened()) {
      fprintf(stderr, "Cannot open %s\n", argv[2]);
      exit(-1);
    }
  }
  int iw = cap->get(CV_CAP_PROP_FRAME_WIDTH);
  int ih = cap->get(CV_CAP_PROP_FRAME_HEIGHT);

  printf("Resolution : %d x %d\n", iw, ih);

  //

  VideoWriter *writer = NULL;
  const char *winname = "Rectangle Detection Demo";

  if (argc < 4 || strcmp(argv[3], "-") == 0) {
    namedWindow(winname, WINDOW_AUTOSIZE );
  } else {
    writer = new VideoWriter(argv[3], fourcc("PIM1"), 30, cvSize(iw, ih), true);
    if (!writer->isOpened()) {
      fprintf(stderr, "Cannot open %s\n", argv[3]);
      exit(-1);
    }
  }

  //

  double aov = 90;

  int did = 0;
  if (argc >= 2) did = atoi(argv[1]);
  if (argc >= 5) aov = atof(argv[4]);

  printf("Horizontal angle of view : %g degrees\n", aov);

  cl_device_id device = simpleGetDevice(did);
  printf("%s\n", getDeviceName(device));
  cl_context context = simpleCreateContext(device);

  cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

  if (loadPlan("plan.txt", device) != 0) printf("No plan\n");

  if (writer == NULL) printf("\n>>>>> Press ENTER on the window to exit <<<<<\n");

  //

  oclimgutil_t *oclimgutil = init_oclimgutil(device, context);
  oclpolyline_t *oclpolyline = init_oclpolyline(device, context);
  oclrect_t *oclrect = init_oclrect(oclimgutil, oclpolyline, device, context, queue, iw, ih);

  //

  const double tanAOV = tan(aov / 2 / 180.0 * M_PI);
  Mat vimg, img[2];

  int nFrame = 0, lastNFrame = 0;
  uint64_t tm = currentTimeMillis();

  cap->grab();
  {
    cap->retrieve(vimg, 0);
    assert(vimg.channels() == 3);
    img[0] = vimg.clone();
    img[1] = vimg.clone();

    uint8_t *data = (uint8_t *)img[nFrame & 1].data;
    int ws = img[nFrame & 1].step;
    oclrect_enqueueTask(oclrect, data, ws);
    cap->grab();

    nFrame++;
  }

  for(;;) {
    if (!cap->retrieve(vimg, 0)) break;
    vimg.copyTo(img[nFrame & 1]);

    uint8_t *data = (uint8_t *)img[nFrame & 1].data;
    int ws = img[nFrame & 1].step;

    oclrect_enqueueTask(oclrect, data, ws);

    cap->grab();

    nFrame++;

    rect_t *ret = oclrect_pollTask(oclrect, tanAOV);

    for(int i=1;i<ret->nItems;i++) { // >>>> This starts from 1 <<<<
      switch(ret[i].status) {
      case 0:
	showRect(ret[i], 0, 255, 0, 1, img[nFrame & 1]);
	break;
      case 2:
	showRect(ret[i], 255, 0, 0, 1, img[nFrame & 1]);
	break;
      case 1:
	showRect(ret[i], 0, 200, 255, 2, img[nFrame & 1]);
	break;
      case 3:
	showRect(ret[i], 0, 0, 255, 2, img[nFrame & 1]);
	break;
      }
    }

    if (writer != NULL) {
      writer->write(img[nFrame & 1]);

      uint64_t t = currentTimeMillis();
      if (t - tm > 1000) {
	printf("%.3g fps\n",  1000.0 * (nFrame - lastNFrame) / ((double)(t - tm)));
	tm = t;
	lastNFrame = nFrame;
      }
    } else {
      imshow(winname, img[nFrame & 1]);
      int key = waitKey(1) & 0xff;
      if (key == 27 || key == 13) break;
    }
  }

  //

  dispose_oclrect(oclrect);
  dispose_oclpolyline(oclpolyline);
  dispose_oclimgutil(oclimgutil);

  //

  ce(clReleaseCommandQueue(queue));
  ce(clReleaseContext(context));

  //

  if (writer != NULL) delete writer;
  delete cap;
  destroyAllWindows();

  //

  exit(0);
}
