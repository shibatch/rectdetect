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
//#include <sys/time.h>

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

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage : %s <image file name> [device number] [output file name]\n", argv[0]);
    fprintf(stderr, "\nAvailable OpenCL Devices :\n");
    simpleGetDevice(-1);
    exit(-1);
  }

  //

  int did = 0;
  if (argc >= 3) did = atoi(argv[2]);

  cl_device_id device = simpleGetDevice(did);
  printf("%s\n", getDeviceName(device));
  cl_context context = simpleCreateContext(device);

  cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

  //

  Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if( img.data == NULL ) exitf(-1, "Could not load %s\n", argv[1]);

  if (img.channels() != 3) exitf(-1, "nChannels != 3\n");

  int iw = img.cols, ih = img.rows, ws = img.step;
  uint8_t *data = (uint8_t *)img.data;

  //

  oclimgutil_t *oclimgutil = init_oclimgutil(device, context);
  oclpolyline_t *oclpolyline = init_oclpolyline(device, context);
  oclrect_t *oclrect = init_oclrect(oclimgutil, oclpolyline, device, context, queue, iw, ih);

  //

  const double tanAOV = tan(72.0 / 2 / 180.0 * M_PI);

  if (loadPlan("plan.txt", device) != 0) {
    printf("Creating plan\n");
    for(int xs = 2;xs <= 256;xs *= 2) {
      for(int ys = 2;ys <= 64;ys *= 2) {
	printf("%d %d\n", xs, ys);
	startProfiling(xs, ys, 1);

	oclrect_enqueueTask(oclrect, data, ws);
	oclrect_pollTask(oclrect, tanAOV);

	finishProfiling();
      }
    }

    savePlan("plan.txt", device);
  }

  //

  rect_t *ret = oclrect_executeOnce(oclrect, data, ws, tanAOV);

  for(int i=1;i<ret->nItems;i++) { // >>>> This starts from 1 <<<<
    switch(ret[i].status) {
    case 0:
    case 2:
      showRect(ret[i], 255, 0, 0, 1, img);
      break;
    case 1:
      showRect(ret[i], 0, 200, 255, 2, img);
      break;
    case 3:
      showRect(ret[i], 0, 0, 255, 2, img);
      break;
    }
  }

  imwrite(argc >= 4 ? argv[3] : "output.jpg", img);

  //

  dispose_oclrect(oclrect);
  dispose_oclpolyline(oclpolyline);
  dispose_oclimgutil(oclimgutil);

  //

  ce(clReleaseCommandQueue(queue));
  ce(clReleaseContext(context));

  //

  exit(0);
}
