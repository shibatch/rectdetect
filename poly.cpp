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

#include "helper.h"
#include "oclhelper.h"

#include "oclimgutil.h"
#include "oclpolyline.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage : %s <image file name> [<device number>]\nThe program will threshold the image, apply CCL,\nand output the result to output.png.\n", argv[0]);
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

  oclimgutil_t *oclimgutil = init_oclimgutil(device, context);
  oclpolyline_t *oclpolyline = init_oclpolyline(device, context);

  //

  Mat rimg = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if( rimg.data == NULL ) exitf(-1, "Could not load %s\n", argv[1]);

  if (rimg.channels() != 3) exitf(-1, "nChannels != 3\n");

  Mat img = rimg.clone();

  int iw = img.cols, ih = img.rows, ws = img.step;
  uint8_t *data = (uint8_t *)img.data;

  //

  cl_int *buf0 = (cl_int *)allocatePinnedMemory(iw * ih * sizeof(cl_int), context, queue);
  cl_int *buf1 = (cl_int *)allocatePinnedMemory(iw * ih * sizeof(cl_int), context, queue);
  cl_int *buf2 = (cl_int *)calloc(iw * ih * sizeof(cl_int), 1);
  cl_int *buf3 = (cl_int *)calloc(iw * ih * sizeof(cl_int), 1);
  cl_int *buf4 = (cl_int *)calloc(iw * ih * sizeof(cl_int), 1);
  cl_int *buf5 = (cl_int *)calloc(iw * ih * sizeof(cl_int), 1);
  cl_int *buf6 = (cl_int *)calloc(iw * ih * sizeof(cl_int), 1);
  cl_int *buf7 = (cl_int *)calloc(iw * ih * sizeof(cl_int), 1);
  cl_int *buf8 = (cl_int *)calloc(iw * ih * sizeof(cl_int), 1);
  cl_int *buf9 = (cl_int *)calloc(iw * ih * sizeof(cl_int), 1);

  memset(buf0, 0, iw * ih * sizeof(cl_int));
  memset(buf1, 0, iw * ih * sizeof(cl_int));

  cl_int *bufBig = (cl_int *)calloc(iw * ih * sizeof(cl_int) * 4, 1);
  cl_int *bufLS = (cl_int *)calloc(iw * ih * sizeof(cl_int) * 4, 1);

  memcpy(buf0, data, ws * ih);

  cl_mem mem0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf0, NULL);
  cl_mem mem1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf1, NULL);
  cl_mem mem2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf2, NULL);
  cl_mem mem3 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf3, NULL);
  cl_mem mem4 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf4, NULL);
  cl_mem mem5 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf5, NULL);
  cl_mem mem6 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf6, NULL);
  cl_mem mem7 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf7, NULL);
  cl_mem mem8 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf8, NULL);
  cl_mem mem9 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), buf9, NULL);
  cl_mem memBig = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int) * 4, bufBig, NULL);
  cl_mem memLS = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int) * 4, bufLS, NULL);

  //

  ce(clFinish(queue));

  oclimgutil_convert_plab_bgr(oclimgutil, mem4, mem0, iw, ih, ws, queue, NULL);
  oclimgutil_unpack_f_f_f_plab(oclimgutil, mem1, mem2, mem3, mem4, iw, ih, queue, NULL);
  oclimgutil_iirblur_f_f(oclimgutil, mem0, mem1, mem4, mem5, 2, iw, ih, queue, NULL);
  oclimgutil_iirblur_f_f(oclimgutil, mem1, mem2, mem4, mem5, 2, iw, ih, queue, NULL);
  oclimgutil_iirblur_f_f(oclimgutil, mem2, mem3, mem4, mem5, 2, iw, ih, queue, NULL);
  oclimgutil_pack_plab_f_f_f(oclimgutil, mem4, mem0, mem1, mem2, iw, ih, queue, NULL);

  oclimgutil_edgevec_f2_f(oclimgutil, memBig, mem0, iw, ih, queue, NULL);
  oclimgutil_edge_f_plab(oclimgutil, mem5, mem4, iw, ih, queue, NULL);
  oclimgutil_thinthres_f_f_f2(oclimgutil, mem2, mem5, memBig, iw, ih, queue, NULL);

  oclimgutil_threshold_f_f(oclimgutil, mem9, mem2, 0.0, 0.0, 1.0, iw * ih, queue, NULL);
  oclimgutil_cast_i_f(oclimgutil, mem8, mem9, 1, iw * ih, queue, NULL);
  oclimgutil_label8x_int_int(oclimgutil, mem3, mem8, mem9, 0, iw, ih, queue, NULL); // out, in, tmp
  oclimgutil_clear(oclimgutil, mem4, iw*ih*4, queue, NULL);
  oclimgutil_calcStrength(oclimgutil, mem4, mem2, mem3, iw, ih, queue, NULL); // out, edge, label
  oclimgutil_filterStrength(oclimgutil, mem3, mem4, 500, iw, ih, queue, NULL); // label, str
  oclimgutil_threshold_i_i(oclimgutil, mem3, mem3, 0, 0, 1, iw * ih, queue, NULL);

  oclpolyline_execute(oclpolyline, memLS, iw*ih*4*4, mem0, mem3, memBig, mem4, mem5, mem6, mem7, mem8, mem9, 1, 20, iw, ih, queue, NULL);

  //

  ce(clEnqueueReadBuffer(queue, mem0, CL_TRUE, 0, iw * ih * sizeof(cl_int), buf0, 0, NULL, NULL));
  ce(clEnqueueReadBuffer(queue, mem1, CL_TRUE, 0, iw * ih * sizeof(cl_int), buf1, 0, NULL, NULL));
  ce(clEnqueueReadBuffer(queue, memLS, CL_TRUE, 0, iw * ih * sizeof(cl_int) * 4, bufLS, 0, NULL, NULL));

  ce(clFinish(queue));

  //

  //memcpy(data, buf0, ws*ih);
  memset(data, 0, ws * ih);

  int n = bufLS[0];

  linesegment_t *ls = ((linesegment_t *)bufLS);

  for(int i=1;i<=n;i++) { // >>>> This starts from 1 <<<<
    if (ls[i].polyid == 0) continue;
    if (ls[i].leftPtr > 0) continue;

    int cnt = 0;
    for(int j=i;j > 0;j = ls[j].rightPtr, cnt++) {
#ifdef _MSC_VER
      if (ls[j].x0 < 0 || ls[j].x0 >= iw || ls[j].x1 < 0 || ls[j].x1 >= iw ||
	  ls[j].y0 < 0 || ls[j].y0 >= ih || ls[j].y1 < 0 || ls[j].y1 >= ih) continue;
#endif
      line(img, cvPoint(ls[j].x0, ls[j].y0), cvPoint(ls[j].x1, ls[j].y1), (cnt & 1) ? Scalar(100, 100, 255) : Scalar(255, 255, 100), 1, 8, 0);
    }
  }

  imwrite("output.png", img);

  //

  dispose_oclpolyline(oclpolyline);
  dispose_oclimgutil(oclimgutil);

  ce(clReleaseMemObject(memLS));
  ce(clReleaseMemObject(memBig));
  ce(clReleaseMemObject(mem9));
  ce(clReleaseMemObject(mem8));
  ce(clReleaseMemObject(mem7));
  ce(clReleaseMemObject(mem6));
  ce(clReleaseMemObject(mem5));
  ce(clReleaseMemObject(mem4));
  ce(clReleaseMemObject(mem3));
  ce(clReleaseMemObject(mem2));
  ce(clReleaseMemObject(mem1));
  ce(clReleaseMemObject(mem0));

  free(bufLS);
  free(bufBig);
  free(buf9);
  free(buf8);
  free(buf7);
  free(buf6);
  free(buf5);
  free(buf4);
  free(buf3);
  free(buf2);
  freePinnedMemory(buf1, context, queue);
  freePinnedMemory(buf0, context, queue);

  //ce(clReleaseProgram(program));
  ce(clReleaseCommandQueue(queue));
  ce(clReleaseContext(context));

  //

  exit(0);
}
