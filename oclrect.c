// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "vec234.h"

#include "helper.h"
#include "oclhelper.h"
#include "oclimgutil.h"
#include "oclpolyline.h"
#include "oclrect.h"
#include "egbuf.h"

#include "oclrect_cl.h"

#define MAGIC 0x808f3801

#define NBUF 6
#define NTMP 6

//

typedef struct oclrect_t {
  uint32_t magic;
  int iw, ih;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;

  oclimgutil_t *oclimgutil;
  oclpolyline_t *oclpolyline;

  cl_mem buf[NBUF];
  cl_mem tmp[NBUF];
  cl_mem iobuf[2], ioBig[2];
  cl_int *hostiobuf[2][2], *hostioBig[2][2];

  int nextPageToEnqueue, nextPageToPoll;
  cl_event events[2];

  cl_kernel kernel_simpleJunction;
  int kid_simpleJunction;
  cl_kernel kernel_simpleConnect;
  int kid_simpleConnect;
  cl_kernel kernel_stringify;
  int kid_stringify;
  cl_kernel kernel_calcStrength;
  int kid_calcStrength;
  cl_kernel kernel_filterStrength;
  int kid_filterStrength;
  cl_kernel kernel_blblur0;
  int kid_blblur0;
  cl_kernel kernel_blblur1;
  int kid_blblur1;
  cl_kernel kernel_quantize;
  int kid_quantize;
  cl_kernel kernel_despeckle;
  int kid_despeckle;
  cl_kernel kernel_mkMergeMask0;
  int kid_mkMergeMask0;
  cl_kernel kernel_mkMergeMask1;
  int kid_mkMergeMask1;
  cl_kernel kernel_labelxPreprocess;
  int kid_labelxPreprocess;
  cl_kernel kernel_labelMergeMain;
  int kid_labelMergeMain;
  cl_kernel kernel_calcSize;
  int kid_calcSize;
  cl_kernel kernel_despeckle2;
  int kid_despeckle2;
  cl_kernel kernel_markBoundary;
  int kid_markBoundary;
  cl_kernel kernel_colorReassign_pass0;
  int kid_colorReassign_pass0;
  cl_kernel kernel_colorReassign_pass1;
  int kid_colorReassign_pass1;
  cl_kernel kernel_reduceLS;
  int kid_reduceLS;

} oclrect_t;

oclrect_t *init_oclrect(oclimgutil_t *oclimgutil, oclpolyline_t *oclpolyline, cl_device_id device, cl_context context, cl_command_queue queue, int iw, int ih) {
  //char *source = readFileAsStr("oclrect.cl", 1024*1024);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, 0, NULL);
  simpleBuildProgram(program, device, "");
  //free(source);

  oclrect_t *thiz = (oclrect_t *)calloc(1, sizeof(oclrect_t));

  thiz->magic = MAGIC;
  thiz->iw = iw;
  thiz->ih = ih;
  thiz->device = device;
  thiz->context = context;
  thiz->queue = queue;

  thiz->nextPageToEnqueue = 0;
  thiz->nextPageToPoll = 0;
  thiz->events[0] = NULL;
  thiz->events[1] = NULL;

  for(int i=0;i<NBUF;i++) thiz->buf[i] = clCreateBuffer(context, CL_MEM_HOST_NO_ACCESS, iw * ih * sizeof(cl_int), NULL, NULL);
  for(int i=0;i<NTMP;i++) thiz->tmp[i] = clCreateBuffer(context, CL_MEM_HOST_NO_ACCESS, iw * ih * sizeof(cl_int), NULL, NULL);

  thiz->hostiobuf[0][0] = (cl_int *)allocatePinnedMemory(iw * ih * sizeof(cl_int), context, queue);
  thiz->hostiobuf[0][1] = (cl_int *)allocatePinnedMemory(iw * ih * sizeof(cl_int), context, queue);
  thiz->hostiobuf[1][0] = (cl_int *)allocatePinnedMemory(iw * ih * sizeof(cl_int), context, queue);
  thiz->hostiobuf[1][1] = (cl_int *)allocatePinnedMemory(iw * ih * sizeof(cl_int), context, queue);
  thiz->iobuf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), thiz->hostiobuf[0][0], NULL);
  thiz->iobuf[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, iw * ih * sizeof(cl_int), thiz->hostiobuf[0][1], NULL);

  thiz->hostioBig[0][0] = (cl_int *)allocatePinnedMemory(4 * iw * ih * sizeof(cl_int), context, queue);
  thiz->hostioBig[0][1] = (cl_int *)allocatePinnedMemory(4 * iw * ih * sizeof(cl_int), context, queue);
  thiz->hostioBig[1][0] = (cl_int *)allocatePinnedMemory(4 * iw * ih * sizeof(cl_int), context, queue);
  thiz->hostioBig[1][1] = (cl_int *)allocatePinnedMemory(4 * iw * ih * sizeof(cl_int), context, queue);
  thiz->ioBig[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * iw * ih * sizeof(cl_int), thiz->hostioBig[0][0], NULL);
  thiz->ioBig[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4 * iw * ih * sizeof(cl_int), thiz->hostioBig[0][1], NULL);

  thiz->oclimgutil = oclimgutil;
  thiz->oclpolyline = oclpolyline;

  thiz->kernel_simpleJunction = clCreateKernel(program, "simpleJunction", NULL);
  thiz->kid_simpleJunction = getNextKernelID();
  thiz->kernel_simpleConnect = clCreateKernel(program, "simpleConnect", NULL);
  thiz->kid_simpleConnect = getNextKernelID();
  thiz->kernel_stringify = clCreateKernel(program, "stringify", NULL);
  thiz->kid_stringify = getNextKernelID();
  thiz->kernel_calcStrength = clCreateKernel(program, "calcStrength", NULL);
  thiz->kid_calcStrength = getNextKernelID();
  thiz->kernel_filterStrength = clCreateKernel(program, "filterStrength", NULL);
  thiz->kid_filterStrength = getNextKernelID();
  thiz->kernel_blblur0 = clCreateKernel(program, "blblur0", NULL);
  thiz->kid_blblur0 = getNextKernelID();
  thiz->kernel_blblur1 = clCreateKernel(program, "blblur1", NULL);
  thiz->kid_blblur1 = getNextKernelID();
  thiz->kernel_quantize = clCreateKernel(program, "quantize", NULL);
  thiz->kid_quantize = getNextKernelID();
  thiz->kernel_despeckle = clCreateKernel(program, "despeckle", NULL);
  thiz->kid_despeckle = getNextKernelID();
  thiz->kernel_mkMergeMask0 = clCreateKernel(program, "mkMergeMask0", NULL);
  thiz->kid_mkMergeMask0 = getNextKernelID();
  thiz->kernel_mkMergeMask1 = clCreateKernel(program, "mkMergeMask1", NULL);
  thiz->kid_mkMergeMask1 = getNextKernelID();
  thiz->kernel_labelxPreprocess = clCreateKernel(program, "labelxPreprocess", NULL);
  thiz->kid_labelxPreprocess = getNextKernelID();
  thiz->kernel_labelMergeMain = clCreateKernel(program, "labelMergeMain", NULL);
  thiz->kid_labelMergeMain = getNextKernelID();
  thiz->kernel_calcSize = clCreateKernel(program, "calcSize", NULL);
  thiz->kid_calcSize = getNextKernelID();
  thiz->kernel_despeckle2 = clCreateKernel(program, "despeckle2", NULL);
  thiz->kid_despeckle2 = getNextKernelID();
  thiz->kernel_markBoundary = clCreateKernel(program, "markBoundary", NULL);
  thiz->kid_markBoundary = getNextKernelID();
  thiz->kernel_colorReassign_pass0 = clCreateKernel(program, "colorReassign_pass0", NULL);
  thiz->kid_colorReassign_pass0 = getNextKernelID();
  thiz->kernel_colorReassign_pass1 = clCreateKernel(program, "colorReassign_pass1", NULL);
  thiz->kid_colorReassign_pass1 = getNextKernelID();
  thiz->kernel_reduceLS = clCreateKernel(program, "reduceLS", NULL);
  thiz->kid_reduceLS = getNextKernelID();

  ce(clReleaseProgram(program));

  return thiz;
}

void dispose_oclrect(oclrect_t *thiz) {
  assert(thiz->magic == MAGIC);
  thiz->magic = 0;

  ce(clReleaseKernel(thiz->kernel_simpleJunction));
  ce(clReleaseKernel(thiz->kernel_simpleConnect));
  ce(clReleaseKernel(thiz->kernel_stringify));
  ce(clReleaseKernel(thiz->kernel_calcStrength));
  ce(clReleaseKernel(thiz->kernel_filterStrength));
  ce(clReleaseKernel(thiz->kernel_blblur0));
  ce(clReleaseKernel(thiz->kernel_blblur1));
  ce(clReleaseKernel(thiz->kernel_quantize));
  ce(clReleaseKernel(thiz->kernel_despeckle));
  ce(clReleaseKernel(thiz->kernel_mkMergeMask0));
  ce(clReleaseKernel(thiz->kernel_mkMergeMask1));
  ce(clReleaseKernel(thiz->kernel_labelxPreprocess));
  ce(clReleaseKernel(thiz->kernel_labelMergeMain));
  ce(clReleaseKernel(thiz->kernel_calcSize));
  ce(clReleaseKernel(thiz->kernel_despeckle2));
  ce(clReleaseKernel(thiz->kernel_markBoundary));
  ce(clReleaseKernel(thiz->kernel_colorReassign_pass0));
  ce(clReleaseKernel(thiz->kernel_colorReassign_pass1));
  ce(clReleaseKernel(thiz->kernel_reduceLS));

  ce(clReleaseMemObject(thiz->ioBig[1]));
  ce(clReleaseMemObject(thiz->ioBig[0]));
  freePinnedMemory(thiz->hostioBig[1][1], thiz->context, thiz->queue);
  freePinnedMemory(thiz->hostioBig[1][0], thiz->context, thiz->queue);
  freePinnedMemory(thiz->hostioBig[0][1], thiz->context, thiz->queue);
  freePinnedMemory(thiz->hostioBig[0][0], thiz->context, thiz->queue);
  ce(clReleaseMemObject(thiz->iobuf[1]));
  ce(clReleaseMemObject(thiz->iobuf[0]));
  freePinnedMemory(thiz->hostiobuf[1][1], thiz->context, thiz->queue);
  freePinnedMemory(thiz->hostiobuf[1][0], thiz->context, thiz->queue);
  freePinnedMemory(thiz->hostiobuf[0][1], thiz->context, thiz->queue);
  freePinnedMemory(thiz->hostiobuf[0][0], thiz->context, thiz->queue);

  for(int i=0;i<NTMP;i++) ce(clReleaseMemObject(thiz->tmp[i]));
  for(int i=0;i<NBUF;i++) ce(clReleaseMemObject(thiz->buf[i]));

  free(thiz);
}

//

#define CLEARLS6B(x) ((size_t)((x + 31) & ~31))

static cl_event runKernel2Dy(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, size_t ws2, const cl_event *events) {
  return runKernel2Dx(queue, kernel, kernelID, CLEARLS6B(ws1), CLEARLS6B(ws2), events);
}

static cl_event genGPUTask(oclrect_t *thiz, uint8_t *imgData, int page, int ws, cl_command_queue queue, const cl_event *events) {
  const int iw = thiz->iw, ih = thiz->ih;
  cl_mem *buf = thiz->buf, *tmp = thiz->tmp, *ioBig = thiz->ioBig, *iobuf = thiz->iobuf;

  memcpy(thiz->hostiobuf[page][0], imgData, ws * ih);

  ce(clEnqueueWriteBuffer(queue, thiz->iobuf[0], CL_FALSE, 0, iw * ih * sizeof(cl_int) * 1, thiz->hostiobuf[page][0]  , 0, NULL, NULL));

  ce(clFlush(queue));

  oclimgutil_convert_plab_bgr(thiz->oclimgutil, buf[0], iobuf[0], iw, ih, ws, queue, NULL);

  oclimgutil_unpack_f_f_f_plab(thiz->oclimgutil, tmp[0], tmp[1], tmp[2], buf[0], iw, ih, queue, NULL);
  oclimgutil_iirblur_f_f(thiz->oclimgutil, tmp[3], tmp[2], ioBig[0], ioBig[1], 2, iw, ih, queue, NULL);
  oclimgutil_iirblur_f_f(thiz->oclimgutil, tmp[2], tmp[1], ioBig[0], ioBig[1], 2, iw, ih, queue, NULL);
  oclimgutil_iirblur_f_f(thiz->oclimgutil, tmp[1], tmp[0], ioBig[0], ioBig[1], 2, iw, ih, queue, NULL);
  oclimgutil_pack_plab_f_f_f(thiz->oclimgutil, buf[1], tmp[1], tmp[2], tmp[3], iw, ih, queue, NULL);

  oclimgutil_edgevec_f2_f(thiz->oclimgutil, ioBig[0], tmp[1], iw, ih, queue, NULL);

  //oclimgutil_convert_bgr_plab(thiz->oclimgutil, iobuf[0], buf[1], iw, ih, ws, queue, NULL);

  oclimgutil_edge_f_plab(thiz->oclimgutil, tmp[0], buf[1], iw, ih, queue, NULL);
  oclimgutil_thinthres_f_f_f2(thiz->oclimgutil, buf[1], tmp[0], ioBig[0], iw, ih, queue, NULL);

  //oclimgutil_convert_bgr_lumaf(thiz->oclimgutil, iobuf[0], buf[1], 1.0f, iw, ih, ws, queue, NULL);

  oclimgutil_threshold_f_f(thiz->oclimgutil, tmp[0], buf[1], 0.0f, 0.0f, 1.0f, iw * ih, queue, NULL);
  oclimgutil_cast_i_f(thiz->oclimgutil, tmp[1], tmp[0], 1.0f, iw * ih, queue, NULL);

  simpleSetKernelArg(thiz->kernel_simpleJunction, "MMii", buf[2], tmp[1], iw, ih);
  runKernel2Dy(queue, thiz->kernel_simpleJunction, thiz->kid_simpleJunction, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_simpleConnect, "MMii", tmp[1], buf[2], iw, ih);
  runKernel2Dy(queue, thiz->kernel_simpleConnect, thiz->kid_simpleConnect, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_stringify, "MMiii", buf[2], tmp[1], 0, iw, ih);
  runKernel2Dy(queue, thiz->kernel_stringify, thiz->kid_stringify, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_stringify, "MMiii", tmp[1], buf[2], 1, iw, ih);
  runKernel2Dy(queue, thiz->kernel_stringify, thiz->kid_stringify, iw, ih, NULL);

  oclimgutil_label8x_int_int(thiz->oclimgutil, buf[2], tmp[1], tmp[0], -1, iw, ih, queue, NULL);
  simpleSetKernelArg(thiz->kernel_calcStrength, "MMMii", buf[3], buf[1], buf[2], iw, ih);
  runKernel2Dy(queue, thiz->kernel_calcStrength, thiz->kid_calcStrength, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_filterStrength, "MMiii", buf[2], buf[3], 500, iw, ih);
  runKernel2Dy(queue, thiz->kernel_filterStrength, thiz->kid_filterStrength, iw, ih, NULL);

  //oclimgutil_convert_bgr_labeli(thiz->oclimgutil, iobuf[0], buf[2], -1, iw, ih, ws, queue, NULL);

  oclimgutil_threshold_i_i(thiz->oclimgutil, tmp[0], buf[2], 0, 0, 1, iw * ih, queue, NULL);

  oclimgutil_cast_c_i(thiz->oclimgutil, tmp[1], tmp[0], iw * ih, queue, NULL);

  simpleSetKernelArg(thiz->kernel_blblur0, "MMMii", tmp[0], tmp[1], buf[0], iw, ih);
  runKernel2Dy(queue, thiz->kernel_blblur0, thiz->kid_blblur0, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_blblur1, "MMMii", buf[4], tmp[1], tmp[0], iw, ih);
  runKernel2Dy(queue, thiz->kernel_blblur1, thiz->kid_blblur1, iw, ih, NULL);

  for(int i=0;i<9;i++) { // 10 - 1
    simpleSetKernelArg(thiz->kernel_blblur0, "MMMii", tmp[0], tmp[1], buf[4], iw, ih);
    runKernel2Dy(queue, thiz->kernel_blblur0, thiz->kid_blblur0, iw, ih, NULL);
    simpleSetKernelArg(thiz->kernel_blblur1, "MMMii", buf[4], tmp[1], tmp[0], iw, ih);
    runKernel2Dy(queue, thiz->kernel_blblur1, thiz->kid_blblur1, iw, ih, NULL);
  }

  //oclimgutil_convert_bgr_plab(thiz->oclimgutil, iobuf[0], buf[4], iw, ih, ws, queue, NULL); // Result of edge-preserving blur

  simpleSetKernelArg(thiz->kernel_quantize, "MMiiiii", tmp[0], buf[4], 24, 24, 24, iw, ih);
  runKernel2Dy(queue, thiz->kernel_quantize, thiz->kid_quantize, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_despeckle, "MMMii", buf[4], tmp[0], buf[1], iw, ih);
  runKernel2Dy(queue, thiz->kernel_despeckle, thiz->kid_despeckle, iw, ih, NULL);

  //oclimgutil_convert_bgr_plab(thiz->oclimgutil, iobuf[0], buf[4], iw, ih, ws, queue, NULL); // Result of edge-preserving blur

  simpleSetKernelArg(thiz->kernel_filterStrength, "MMiii", buf[2], buf[3], 2500, iw, ih);
  runKernel2Dy(queue, thiz->kernel_filterStrength, thiz->kid_filterStrength, iw, ih, NULL);

  //oclimgutil_convert_bgr_labeli(thiz->oclimgutil, iobuf[0], buf[2], -1, iw, ih, ws, queue, NULL); // Strong edge

  oclimgutil_threshold_i_i(thiz->oclimgutil, buf[3], buf[2], 0, 0, 1, iw * ih, queue, NULL);

  simpleSetKernelArg(thiz->kernel_simpleJunction, "MMii", tmp[0], buf[2], iw, ih);
  runKernel2Dy(queue, thiz->kernel_simpleJunction, thiz->kid_simpleJunction, iw, ih, NULL);

  oclimgutil_clear(thiz->oclimgutil, tmp[1], iw * ih * 4, queue, NULL);
  simpleSetKernelArg(thiz->kernel_mkMergeMask0, "MMii", tmp[1], tmp[0], iw, ih);
  runKernel2Dy(queue, thiz->kernel_mkMergeMask0, thiz->kid_mkMergeMask0, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_mkMergeMask1, "MMii", tmp[1], tmp[0], iw, ih);
  runKernel2Dy(queue, thiz->kernel_mkMergeMask1, thiz->kid_mkMergeMask1, iw, ih, NULL);

  //oclimgutil_convert_bgr_labeli(thiz->oclimgutil, iobuf[0], tmp[1], -1, iw, ih, ws, queue, NULL); // merge mask

  simpleSetKernelArg(thiz->kernel_labelxPreprocess, "MMii", buf[5], buf[4], iw, ih);
  runKernel2Dy(queue, thiz->kernel_labelxPreprocess, thiz->kid_labelxPreprocess, iw, ih, NULL);

  for(int i=1;i<=8;i++) {
    simpleSetKernelArg(thiz->kernel_labelMergeMain, "MMMMii", buf[5], buf[4], tmp[1], buf[2], iw, ih);
    runKernel2Dy(queue, thiz->kernel_labelMergeMain, thiz->kid_labelMergeMain, iw, ih, NULL);
  }

  simpleSetKernelArg(thiz->kernel_calcSize, "MMii", tmp[0], buf[5], iw, ih);
  runKernel2Dy(queue, thiz->kernel_calcSize, thiz->kid_calcSize, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_despeckle2, "MMiii", buf[5], tmp[0], 16, iw, ih);
  runKernel2Dy(queue, thiz->kernel_despeckle2, thiz->kid_despeckle2, iw, ih, NULL);

  //oclimgutil_convert_bgr_labeli(thiz->oclimgutil, iobuf[0], buf[5], 0, iw, ih, ws, queue, NULL); // merge mask

  simpleSetKernelArg(thiz->kernel_markBoundary, "MMMii", tmp[1], buf[5], buf[2], iw, ih);
  runKernel2Dy(queue, thiz->kernel_markBoundary, thiz->kid_markBoundary, iw, ih, NULL);
  oclimgutil_label8x_int_int(thiz->oclimgutil, iobuf[1], tmp[1], tmp[0], -1, iw, ih, queue, NULL);

  //oclimgutil_convert_bgr_labeli(thiz->oclimgutil, iobuf[0], iobuf[1], -1, iw, ih, ws, queue, NULL); // boundary label

#if 0
  oclimgutil_copy(thiz->oclimgutil, tmp[4], buf[0], iw * ih * 4, queue, NULL);
  oclimgutil_clear(thiz->oclimgutil, tmp[0], iw * ih * 4, queue, NULL);
  oclimgutil_clear(thiz->oclimgutil, tmp[1], iw * ih * 4, queue, NULL);
  oclimgutil_clear(thiz->oclimgutil, tmp[2], iw * ih * 4, queue, NULL);
  oclimgutil_clear(thiz->oclimgutil, tmp[3], iw * ih * 4, queue, NULL);

  simpleSetKernelArg(thiz->kernel_colorReassign_pass0, "MMMMMMii", tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], buf[5], iw, ih);
  runKernel2Dy(queue, thiz->kernel_colorReassign_pass0, thiz->kid_colorReassign_pass0, iw, ih, NULL);
  simpleSetKernelArg(thiz->kernel_colorReassign_pass1, "MMMMMMii", tmp[4], tmp[0], tmp[1], tmp[2], tmp[3], buf[5], iw, ih);
  runKernel2Dy(queue, thiz->kernel_colorReassign_pass1, thiz->kid_colorReassign_pass1, iw, ih, NULL);

  oclimgutil_convert_bgr_plab(thiz->oclimgutil, iobuf[0], tmp[4], iw, ih, ws, queue, NULL); // Result of edge-preserving blur
#endif

  oclpolyline_execute(thiz->oclpolyline, ioBig[0], iw*ih*4*4, buf[0], buf[3], ioBig[1], tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], 4.0f, 20, iw, ih, queue, NULL);

  //oclimgutil_convert_bgr_labeli(thiz->oclimgutil, iobuf[0], buf[0], 0, iw, ih, ws, queue, NULL); // boundary label

  oclimgutil_clear(thiz->oclimgutil, ioBig[1], iw * ih * 4 * 4, queue, NULL);
  simpleSetKernelArg(thiz->kernel_reduceLS, "MMMiii", ioBig[1], iobuf[1], buf[0], iw, ih, iw * ih * 4 / 5);
  runKernel2Dy(queue, thiz->kernel_reduceLS, thiz->kid_reduceLS, iw, ih, NULL);

  cl_event ev;

  ce(clEnqueueReadBuffer(queue, thiz->ioBig[0], CL_FALSE, 0, iw * ih * sizeof(cl_int) * 4, thiz->hostioBig[page][0], 0, NULL, NULL));
  ce(clEnqueueReadBuffer(queue, thiz->ioBig[1], CL_FALSE, 0, iw * ih * sizeof(cl_int) * 4, thiz->hostioBig[page][1], 0, NULL, NULL));
#if 0
  ce(clEnqueueReadBuffer(queue, thiz->iobuf[0], CL_FALSE, 0, iw * ih * sizeof(cl_int) * 1, thiz->hostiobuf[page][0], 0, NULL, NULL));
#endif
  ce(clEnqueueReadBuffer(queue, thiz->iobuf[1], CL_FALSE, 0, iw * ih * sizeof(cl_int) * 1, thiz->hostiobuf[page][1], 0, NULL, &ev));

  ce(clFlush(queue));

  return ev;
}

//

typedef struct {
  vec2 e0, e1;
} ls_t;

static inline double squ(double x) { return x * x; }
static float lsSquLen(const ls_t *ls) { return distanceSqu2(ls->e0, ls->e1); }

static inline double cross2(vec2 v, vec2 w) {
  return v.a[0]*w.a[1] - v.a[1]*w.a[0];
}

static inline vec3 cross3(vec3 v, vec3 w) {
  return cvec3( v.a[1] * w.a[2] - v.a[2] * w.a[1], v.a[2] * w.a[0] - v.a[0] * w.a[2], v.a[0] * w.a[1] - v.a[1] * w.a[0] );
}

static inline vec2 closestPoint2(vec2 v, vec2 w, vec2 p) {
  double l2 = distanceSqu2(v, w);
  if (l2 == 0.0) return cvec2(v.a[0], v.a[1]);
  double t = ((p.a[0] - v.a[0]) * (w.a[0] - v.a[0]) + (p.a[1] - v.a[1]) * (w.a[1] - v.a[1])) / l2;

  return cvec2(v.a[0] + t * (w.a[0] - v.a[0]), v.a[1] + t * (w.a[1] - v.a[1]));
}

static inline vec2 closestPointLS2(vec2 v, vec2 w, vec2 p) {
  double l2 = distanceSqu2(v, w);
  if (l2 == 0.0) return cvec2 ( v.a[0], v.a[1] );
  double t = ((p.a[0] - v.a[0]) * (w.a[0] - v.a[0]) + (p.a[1] - v.a[1]) * (w.a[1] - v.a[1])) / l2;
  if (t < 0) return cvec2 ( v.a[0], v.a[1] );
  else if (t > 1.0) return cvec2 ( w.a[0], w.a[1] );

  return cvec2 ( v.a[0] + t * (w.a[0] - v.a[0]), v.a[1] + t * (w.a[1] - v.a[1]) );
}

static inline vec2 intersection2(ls_t u, ls_t v) {
  double d = (v.e1.a[0] - v.e0.a[0]) * (u.e1.a[1] - u.e0.a[1]) - (v.e1.a[1] - v.e0.a[1]) * (u.e1.a[0] - u.e0.a[0]);
  if (fabs(d) < 1e-4) return cvec2 ( NAN, NAN );
  double n = (v.e0.a[1] - u.e0.a[1]) * (u.e1.a[0] - u.e0.a[0]) - (v.e0.a[0] - u.e0.a[0]) * (u.e1.a[1] - u.e0.a[1]);
  double q = n / d;

  return cvec2 ( v.e0.a[0] + q * (v.e1.a[0] - v.e0.a[0]), v.e0.a[1] + q * (v.e1.a[1] - v.e0.a[1]) );
}

// Pose estimator

#define initScale (1.0)
#define EPS (1e-6)

typedef struct {
  vec4 a[2];
} vec24;

typedef struct {
  vec3 *points;
  int mode;
} arg_t;

static double value(vec4 v, void *arg) {
  vec3 *points = ((arg_t *)arg)->points;
  int mode = ((arg_t *)arg)->mode;

  vec3 q[4];
  for(int i=0;i<4;i++) q[i] = dot3(points[i], v.a[i]);

  double score = 0;

  double l01 = distanceSqu3(q[0], q[1]);
  double l12 = distanceSqu3(q[1], q[2]);
  double l23 = distanceSqu3(q[2], q[3]);
  double l03 = distanceSqu3(q[0], q[3]);
  double l02 = distanceSqu3(q[0], q[2]);
  double l13 = distanceSqu3(q[1], q[3]);

  double comp = 1.0;

  score += squ((mode ? l23 : l03) - 1);
  score += squ((mode ? l01 : l12) - 1);
  comp = 1.0 / (mode ? l12 : l01);

  score += lengthSqu3(plus3(minus3(mode ? q[0] : q[2], q[1]), minus3(mode ? q[2] : q[0], q[3])));
  score += comp * lengthSqu3(plus3(minus3(q[1], mode ? q[2] : q[0]), minus3(q[3], mode ? q[0] : q[2])));

  score += squ(l01 + l12 - l02);
  score += squ(l03 + l23 - l02);
  score += squ(l01 + l03 - l13);
  score += squ(l12 + l23 - l13);

  vec3 n013 = cross3(minus3(q[1], q[0]), minus3(q[3], q[0]));
  score += comp * squ(vdot3(n013, q[2]) - vdot3(n013, q[0])) / vdot3(n013, n013);
  vec3 n102 = cross3(minus3(q[0], q[1]), minus3(q[2], q[1]));
  score += comp * squ(vdot3(n102, q[3]) - vdot3(n102, q[1])) / vdot3(n102, n102);

  return score;
}

static inline vec3 gradient(vec4 v, vec4 dir, void *arg) {
  double h = EPS;
  double f0 = value(v, arg);
  double fp = value(plus4(v, dot4(dir, h)), arg);
  double fm = value(plus4(v, dot4(dir,-h)), arg);

  vec3 ret;
  ret.a[0] = f0;
  ret.a[1] = (fp - fm) * (1.0 / (2*h));
  ret.a[2] = (fp + fm - 2*f0) * (1.0 / (h*h));
  return ret;
}

static inline vec24 gradient2(vec4 v, void *arg) {
  vec4 a, a2, d;

  double fx = value(v, arg);

  for(int i=0;i<4;i++) {
    for(int j=0;j<4;j++) {
      d.a[j] = 0;
      if (j == i) d.a[j] = EPS;
    }

    double fxmh = value(minus4(v, d), arg);
    double fxph = value(plus4(v, d), arg);
    a.a[i] = (fxph - fxmh) / (2*EPS);
    a2.a[i] = (fxmh - 2*fx + fxph) / (EPS*EPS);
  }

  vec24 ret = { { a, a2 } };

  return ret;
}

static inline vec4 lineSearch(vec4 iv, vec4 dir, int nIter2, void *arg) {
  dir = normalize4(dir);
  vec3 gd;
  double scale = initScale;

  for(int i=0;i<nIter2;i++) {
    gd = gradient(iv, dir, arg);
    double ep = gd.a[0];
    if (gd.a[2]*gd.a[2] < 1e-10) gd.a[2] = 1;
    double delta = fabs(gd.a[1] / gd.a[2]);
    if (delta < 1e-10) return iv;
    vec4 v = plus4(iv, dot4(dir, delta * scale));
    double e1 = value(v, arg);
    if (ep < e1) {
      scale *= 0.5;
      continue;
    }

    iv = v;
  }

  return iv;
}

static inline vec4 inversedot(vec4 m, vec4 r) {
  vec4 a;

  int isAllPositive = 1;

  for(int i=0;i<4;i++) {
    if (m.a[i] <= 0) isAllPositive = 0;
  }

  if (isAllPositive) {
    for(int i=0;i<4;i++) {
      a.a[i] = 1.0 / m.a[i];
      a.a[i] *= r.a[i];
    }
  } else return r;

  return a;
}

static vec4 cgexecute(vec4 iv, int loopCnt, int nIter2, void *arg) {
  int i = 0, k = 0;
  vec4 x = iv;
  vec24 g2 = gradient2(x, arg);
  vec4 r = dot4(g2.a[0], -1);
  vec4 m = g2.a[1];
  vec4 s = inversedot(m, r), d = s;
  double deltanew = vdot4(r, d);

  while(i < loopCnt) {
    x = lineSearch(x, d, nIter2, arg);
    g2 = gradient2(x, arg);
    r = dot4(g2.a[0], -1);
    m = g2.a[1];
    double deltaold = deltanew;
    double deltamid = vdot4(r, s);
    s = inversedot(m, r);
    deltanew = vdot4(r, s);
    double beta = (deltanew - deltamid) / deltaold;
    if (k == 10 || beta <= 0 || deltaold == 0) {
      d = s;
      k = 0;
    } else {
      d = plus4(s, dot4(d, beta));
    }

    k++;
    i++;
  }

  return x;
}

static struct rect_t poseEstimation(ls_t *als, vec2 gv, int iw, int ih, double tanAOV) {
  vec3 p[4];

  int tl = 0;

  double min = 1e+100;
	    
  for(int i=0;i<4;i++) {
    vec2 v = normalize2(minus2(als[i].e1, als[i].e0));
    v = cvec2 ( -v.a[1], v.a[0] );
    if (vdot2(minus2(als[i].e0, gv), v) < 0) v = dot2(v, -1);
    if (v.a[1] < min) { min = v.a[1]; tl = i; }
  }

  for(int i=0;i<4;i++) {
    p[i] = normalize3(cvec3( (als[(i+tl)&3].e0.a[0] - (iw/2)), (-(als[(i+tl)&3].e0.a[1] - ih/2)), iw/2 / tanAOV ) );
  }

  double d01 = 1.0 / distance3(p[0], p[1]);
  double d23 = 1.0 / distance3(p[2], p[3]);

  arg_t arg0 = { p, 1 };
  vec4 x0 = cgexecute(cvec4( d01, d01, d23, d23 ), 12, 10, &arg0);
  double v0 = value(x0, &arg0);

  double d12 = 1.0 / distance3(p[1], p[2]);
  double d03 = 1.0 / distance3(p[0], p[3]);

  arg_t arg1 = { p, 0 };
  vec4 x1 = cgexecute(cvec4( d03, d12, d12, d03 ), 12, 10, &arg1);
  double v1 = value(x1, &arg1);

  rect_t ret;
  ret.value = v0 < v1 ? v0 : v1;

  vec4 x = v0 < v1 ? x0 : x1;
  if (x.a[0] < 0) x = dot4(x, -1);

  for(int i=0;i<4;i++) {
    ret.c3[i] = dot3(p[i], x.a[i]);
    ret.c2[i] = cvec2 ( als[(i+tl)&3].e0.a[0], als[(i+tl)&3].e0.a[1] );
  }
	
  return ret;
}

static int looksLikeAScreen(rect_t r) {
  if (r.value > 0.05) return 0;

  if (r.c3[0].a[2] < 0 || r.c3[1].a[2] < 0 || r.c3[2].a[2] < 0 || r.c3[3].a[2] < 0) return 0;

  double asp = distance3(r.c3[0], r.c3[1]) / distance3(r.c3[1], r.c3[2]);

  if (asp < 1.0 / 12 || 12 < asp) return 0;

  double maxs = 0, mins = 1e+100;
  for(int i=0;i<4;i++) {
    double s0 = distanceSqu2(r.c2[(i+2)%4], closestPointLS2(r.c2[i], r.c2[(i+1)%4], r.c2[(i+2)%4]));
    double s1 = distanceSqu2(r.c2[(i+3)%4], closestPointLS2(r.c2[i], r.c2[(i+1)%4], r.c2[(i+3)%4]));
    maxs = fmax(maxs, fmax(s0, s1));
    mins = fmin(mins, fmax(s0, s1));
  }

  if (maxs / mins > 100) return 0;

  return 1;
}

// Quick hull

static void findHull2(EGBuf *hull, EGBuf *s, vec2 vLeft, vec2 vRight) {
  vec2 *pFarthest = NULL;

  double d = 0;
  for(int i=0;i<s->size;i++) {
    vec2 *p = &((vec2 *)s->ptr)[i];
    double e = distanceSqu2(closestPoint2(vLeft, vRight, *p), *p);

    if (pFarthest == NULL || e > d) {
      pFarthest = p;
      d = e;
    }
  }

  if (d < 0.01 || pFarthest == NULL) return;

  vec2 vTopRight = cvec2( pFarthest->a[1] - vRight.a[1], vRight.a[0] - pFarthest->a[0] );
  vec2 vTopLeft  = cvec2( vLeft.a[1]  - pFarthest->a[1], pFarthest->a[0] -  vLeft.a[0] );

  EGBuf *sTopRight = EGBuf_init(sizeof(vec2));
  EGBuf *sTopLeft  = EGBuf_init(sizeof(vec2));

  for(int i=0;i<s->size;i++) {
    vec2 *p = &((vec2 *)s->ptr)[i];
    if (p == pFarthest) continue;
    if (vdot2(minus2(*p, *pFarthest), vTopRight) > 0) EGBuf_add(sTopRight, p);
    if (vdot2(minus2(*p, *pFarthest), vTopLeft ) > 0) EGBuf_add(sTopLeft , p);
  }

  findHull2(hull, sTopRight, *pFarthest, vRight);
  EGBuf_add(hull, pFarthest);
  findHull2(hull, sTopLeft, vLeft, *pFarthest);

  EGBuf_dispose(sTopLeft);
  EGBuf_dispose(sTopRight);
}

static EGBuf *quickHull2(EGBuf *s) {
  EGBuf *hull = EGBuf_init(sizeof(vec2));

  if (s->size == 0) return hull;
	
  vec2 vRight = ((vec2 *)s->ptr)[0], vLeft = ((vec2 *)s->ptr)[0];

  for(int i=0;i<s->size;i++) {
    if (((vec2 *)s->ptr)[i].a[0] > vRight.a[0]) vRight = ((vec2 *)s->ptr)[i];
    if (((vec2 *)s->ptr)[i].a[0] < vLeft.a [0]) vLeft  = ((vec2 *)s->ptr)[i];
  }

  vec2 vTop = cvec2 ( vLeft.a[1] - vRight.a[1], vRight.a[0] - vLeft.a[0] );
	
  EGBuf *sTop = EGBuf_init(sizeof(vec2));
  EGBuf *sBot = EGBuf_init(sizeof(vec2));

  for(int i=0;i<s->size;i++) {
    vec2 *p = &((vec2 *)s->ptr)[i];
    if (p->a[0] == vLeft .a[0] && p->a[1] == vLeft .a[1]) continue;
    if (p->a[0] == vRight.a[0] && p->a[1] == vRight.a[1]) continue;
    if (vdot2(minus2(*p, vLeft), vTop) > 0) {
      EGBuf_add(sTop, p);
    } else {
      EGBuf_add(sBot, p);
    }
  }

  EGBuf_add(hull, &vRight);
  findHull2(hull, sTop, vLeft, vRight);
  EGBuf_add(hull, &vLeft);
  findHull2(hull, sBot, vRight, vLeft);

  EGBuf_dispose(sTop);
  EGBuf_dispose(sBot);

  return hull;
}

// Cohen Sutherland clipping algorithm 

// The following two functions(computeOutCode and clipLineWithRect)
// use material from the Wikipedia article
// (https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm),
// which is released under the Creative Commons
// Attribution-Share-Alike License 3.0.

#define LEFT 1   // 0001
#define RIGHT 2  // 0010
#define BOTTOM 4 // 0100
#define TOP 8    // 1000

static inline int computeOutCode(double x, double y, double xmin, double ymin, double xmax, double ymax) {
  int code = 0;
  if (x < xmin) code |= LEFT;
  if (x > xmax) code |= RIGHT;
  if (y < ymin) code |= BOTTOM;
  if (y > ymax) code |= TOP;
  return code;
}

static vec4 clipLineWithRect(double x0, double y0, double x1, double y1, double xmin, double ymin, double xmax, double ymax) {
  int outcode0 = computeOutCode(x0, y0, xmin, ymin, xmax, ymax);
  int outcode1 = computeOutCode(x1, y1, xmin, ymin, xmax, ymax);
  int accept = 0;

  for(;;) {
    if ((outcode0 | outcode1) == 0) {
      accept = 1;
      break;
    } else if ((outcode0 & outcode1) != 0) {
      break;
    } else {
      double x = 0, y = 0;
      int outcodeOut = outcode0 != 0 ? outcode0 : outcode1;

      if ((outcodeOut & TOP) != 0) {
	x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0);
	y = ymax;
      } else if ((outcodeOut & BOTTOM) != 0) {
	x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0);
	y = ymin;
      } else if ((outcodeOut & RIGHT) != 0) {
	y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0);
	x = xmax;
      } else if ((outcodeOut & LEFT) != 0) {
	y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0);
	x = xmin;
      }

      if (outcodeOut == outcode0) {
	x0 = x;
	y0 = y;
	outcode0 = computeOutCode(x0, y0, xmin, ymin, xmax, ymax);
      } else {
	x1 = x;
	y1 = y;
	outcode1 = computeOutCode(x1, y1, xmin, ymin, xmax, ymax);
      }
    }
  }

  if (accept) return cvec4( x0, y0, x1, y1 );

  return cvec4( NAN, NAN, NAN, NAN );
}

// --

static int lengthComparator(const void *p0, const void *p1) {
  const ls_t *ls0 = (const ls_t *)p0;
  const ls_t *ls1 = (const ls_t *)p1;

  float squlen0 = lsSquLen(ls0), squlen1 = lsSquLen(ls1);
  if (squlen0 > squlen1) return +1;
  if (squlen0 < squlen1) return -1;
  return 0;
}

static void sortByLength(EGBuf *als) {
  assert(als->sizeMember == sizeof(ls_t));
  qsort(als->ptr, als->size, sizeof(ls_t), lengthComparator);
}

static int angleComparator(const void *p0, const void *p1, void *arg) {
  const ls_t *ls0 = (const ls_t *)p0;
  const ls_t *ls1 = (const ls_t *)p1;
  if (ls0 == ls1) return 0;

  vec2 gv = *(vec2 *)arg;

  double a0, a1;
  {
    vec2 v = minus2(ls0->e0, ls0->e1);
    v = cvec2 ( v.a[1], -v.a[0] );
    if (vdot2(v, minus2(ls0->e0, gv)) < 0) v = dot2(v, -1);
    a0 = atan2(v.a[0], v.a[1]);
  }
  {
    vec2 v = minus2(ls1->e0, ls1->e1);
    v = cvec2 ( v.a[1], -v.a[0] );
    if (vdot2(v, minus2(ls1->e0, gv)) < 0) v = dot2(v, -1);
    a1 = atan2(v.a[0], v.a[1]);
  }

  if (a0 > a1) return +1;
  if (a0 < a1) return -1;

  return 0;
}

#ifndef _MSC_VER
static void sortByAngle(EGBuf *als, vec2 gv) {
  assert(als->sizeMember == sizeof(ls_t));
  qsort_r(als->ptr, als->size, sizeof(ls_t), angleComparator, &gv);
}
#else
static int angleComparatorMS(void *arg, const void *p0, const void *p1) {
  return angleComparator(p0, p1, arg);
}

static void sortByAngle(EGBuf *als, vec2 gv) {
  assert(als->sizeMember == sizeof(ls_t));
  qsort_s(als->ptr, als->size, sizeof(ls_t), angleComparatorMS, &gv);
}
#endif

static vec2 gv(EGBuf *als) {
  vec2 gv = cvec2 ( 0, 0 );

  double lenSum = 0;

  for(int i=0;i<als->size;i++) {
    ls_t ls = ((ls_t *)als->ptr)[i];
    double len = distance2(ls.e0, ls.e1);
    gv = plus2(gv, dot2(plus2(ls.e0, ls.e1), len));
    lenSum += len;
  }
	
  return dot2(gv, 0.5 / lenSum);
}

static double sumLength(EGBuf *als) {
  if (als == 0) return 0;
  double ret = 0;
  for(int i=0;i<als->size;i++) ret += sqrt(lsSquLen(&((ls_t *)als->ptr)[i]));
  return ret;
}

static int closeToTriangle(EGBuf *als, double ratio) {
  for(int i=0;i<als->size;i++) {
    ls_t ls0 = ((ls_t *)als->ptr)[i];
    ls_t ls1 = ((ls_t *)als->ptr)[(i+1)%als->size];
    double d0 = distanceSqu2(ls0.e1, closestPoint2(ls0.e0, ls1.e1, ls0.e1));
    double d1 = distanceSqu2(ls0.e0, ls1.e1);
    if (d0 / d1 < ratio) return 1;
  }
  return 0;
}

static int isConvex(EGBuf *als) {
  ls_t ls0 = ((ls_t *)als->ptr)[0];
  ls_t ls1 = ((ls_t *)als->ptr)[1];

  double px0 = ls0.e1.a[0] - ls0.e0.a[0];
  double py0 = ls0.e1.a[1] - ls0.e0.a[1];
  double px1 = ls1.e1.a[0] - ls1.e0.a[0];
  double py1 = ls1.e1.a[1] - ls1.e0.a[1];

  int sign = px0 * py1 - py0 * px1 > 0;

  const int as = als->size;

  for(int i=1;i<as;i++) {
    ls_t ls0 = ((ls_t *)als->ptr)[i];
    ls_t ls1 = ((ls_t *)als->ptr)[(i+1)%as];

    double qx0 = ls0.e1.a[0] - ls0.e0.a[0];
    double qy0 = ls0.e1.a[1] - ls0.e0.a[1];
    double qx1 = ls1.e1.a[0] - ls1.e0.a[0];
    double qy1 = ls1.e1.a[1] - ls1.e0.a[1];
    if (sign != (qx0 * qy1 - qy0 * qx1 > 0)) return 0;
  }

  return 1;
}

//

static EGBuf *removeShortLS(EGBuf *als, float ratio) {
  if (als->size <= 4) return als;

  sortByLength(als);

  float longestSquLen = lsSquLen(&((ls_t *)als->ptr)[als->size-1]);

  for(;;) {
    if (als->size <= 4) break;
    float shortestSquLen = lsSquLen(&((ls_t *)als->ptr)[0]);

    if (shortestSquLen / longestSquLen > ratio * ratio) break;

    EGBuf_remove(als, 0);
  }

  return als;
}

static EGBuf *pickExternalLS(EGBuf *als) {
  EGBuf *plist = EGBuf_init(sizeof(vec2));

  for(int i=0;i<als->size;i++) {
    ls_t ls = ((ls_t *)als->ptr)[i];
    EGBuf_add(plist, &ls.e0);
    EGBuf_add(plist, &ls.e1);
  }

  EGBuf *q = quickHull2(plist);
  EGBuf *als2 = EGBuf_init(sizeof(ls_t));

  const double DTHRE0 = 1, ATHRE1 = 0.95, DTHRE1 = 0.01;

  for(int i=0;i<q->size;i++) {
    vec2 q0 = ((vec2 *)q->ptr)[i];
    vec2 q1 = ((vec2 *)q->ptr)[(i+1)%q->size];
    vec2 m = midpoint2(q0, q1), nq01 = normalize2(minus2(q0, q1));
    int lastAdded = -1;

    sortByLength(als);

    for(int j=als->size-1;j>=0;j--) {
      ls_t e = ((ls_t *)als->ptr)[j];

      if (distanceSqu2(m, closestPointLS2(e.e0, e.e1, m)) < DTHRE0) {
	EGBuf_add(als2, &e);
	lastAdded = j;
	break;
      }

      if (fabs(vdot2(nq01, normalize2(minus2(e.e0, e.e1)))) > ATHRE1 &&
	  distanceSqu2(m, closestPointLS2(e.e0, e.e1, m)) / distanceSqu2(q0, q1) < DTHRE1) {
	EGBuf_add(als2, &e);
	lastAdded = j;
	break;
      }
    }

    if (lastAdded != -1) EGBuf_remove(als, lastAdded);
  }

  EGBuf_dispose(q);
  EGBuf_dispose(plist);
  EGBuf_dispose(als);

  return als2;
}

static EGBuf *pickLongestLS(EGBuf *als, int n) {
  if (als->size <= n) return als;

  sortByLength(als);

  EGBuf *ret = EGBuf_init(sizeof(ls_t));

  for(int j=als->size-1;j>=0;j--) {
    EGBuf_add(ret, &((ls_t *)als->ptr)[j]);
    if (ret->size == n) break;
  }
	    
  EGBuf_dispose(als);

  return ret;
}

static EGBuf *findCorners(EGBuf *als) {
#ifndef _MSC_VER
  vec2 c[als->size];
#else
  vec2 *c = (vec2 *)malloc(sizeof(vec2) * als->size);
#endif

  for(int i=0;i<als->size;i++) {
    ls_t ls0 = ((ls_t *)als->ptr)[i];
    ls_t ls1 = ((ls_t *)als->ptr)[(i+1)%als->size];
    c[i] = intersection2(ls0, ls1);
    if (isnan(c[i].a[0])) {
      EGBuf_dispose(als);
#ifdef _MSC_VER
      free(c);
#endif
      return NULL;
    }
  }

  EGBuf *ret = EGBuf_init(sizeof(ls_t));

  for(int i=0;i<als->size;i++) {
    ls_t ls = { c[i], c[(i+1) % als->size] };
    EGBuf_add(ret, &ls);
  }

  EGBuf_dispose(als);

#ifdef _MSC_VER
  free(c);
#endif

  return ret;
}

//

rect_t *executeCPUTask(oclrect_t *thiz, int page, const double tanAOV) {
  const int iw = thiz->iw, ih = thiz->ih;

  int n = thiz->hostioBig[page][0][0];
  //printf("number of line segments : %d\n", n);

  linesegment_t *ls = (linesegment_t *)(thiz->hostioBig[page][0]);

  EGBuf *ret = EGBuf_init(sizeof(rect_t));

  {
    rect_t fi;
    EGBuf_add(ret, &fi);
  }

  {
    ArrayMap *lsMap = initArrayMap();
    for(int i=1;i<=n;i++) {
      if (ls[i].polyid == 0) continue;
      double x0 = rint(ls[i].x0), y0 = rint(ls[i].y0), x1 = rint(ls[i].x1), y1 = rint(ls[i].y1);

      const int N = 3, DIST = 2;
		
      vec2 d = normalize2(minus2(cvec2 ( x1, y1 ), cvec2 ( x0, y0 )));
      vec2 vd = cvec2 ( -d.a[1], d.a[0] );

      for(int j=0;j<N;j++) {
	for(int dist=-DIST;dist<=DIST;dist++) {
	  vec2 p = plus2(cvec2 ( x0, y0 ), dot2(minus2(cvec2 ( x1, y1 ), cvec2 ( x0, y0 )), (j + 0.5) / N));
	  vec2 c = plus2(p, dot2(vd, dist));
		    
	  int x = (int)(c.a[0] + 0.5), y = (int)(c.a[1] + 0.5);
	  if (x < 0 || x >= iw || y < 0 || y >= ih) continue;
	  int lsid = i;
	  int segid = thiz->hostiobuf[page][1][x + y * iw];

	  if (segid <= 0) continue;

	  EGBuf *set = (EGBuf *)ArrayMap_get(lsMap, segid);
	  if (set == NULL) {
	    set = EGBuf_init(sizeof(int));
	    ArrayMap_put(lsMap, segid, set);
	  }

	  int k;
	  for(k=0;k<set->size;k++) if (((int *)(set->ptr))[k] == lsid) break;
	  if (k == set->size) EGBuf_add(set, &lsid);
	}
      }
    }

    uint64_t *keyArray = ArrayMap_keyArray(lsMap);
    int lsMapSize = ArrayMap_size(lsMap);

    for(int i=0;i<lsMapSize;i++) {
      int segid = keyArray[i];
      EGBuf *lsidSet = (EGBuf *)ArrayMap_get(lsMap, segid);

      assert(lsidSet != NULL);
      if (lsidSet->size < 4) continue;
		
      EGBuf *als = EGBuf_init(sizeof(ls_t));

      for(int j=0;j<lsidSet->size;j++) {
	int lsid = ((int *)(lsidSet->ptr))[j];
	int hash = (int)((((uint32_t)lsid*(uint32_t)segid) & 0x7fffffff) % (unsigned int)(iw * ih * 4 / 5));

	if (thiz->hostioBig[page][1][hash*5+0] != lsid) {
	  if (thiz->hostioBig[page][1][hash*5+0] != 0) {
	    ls_t lse = { cvec2 ( ls[lsid].x0, ls[lsid].y0 ), cvec2 ( ls[lsid].x1, ls[lsid].y1 ) };
	    EGBuf_add(als, &lse);
	  }
	  continue;
	}

	vec4 cl = clipLineWithRect(ls[lsid].x0, ls[lsid].y0, ls[lsid].x1, ls[lsid].y1,
				   iw - thiz->hostioBig[page][1][hash*5+1], ih - thiz->hostioBig[page][1][hash*5+3],
				   thiz->hostioBig[page][1][hash*5+2], thiz->hostioBig[page][1][hash*5+4]);

	if (isnan(cl.a[0])) continue;

	ls_t lse = { cvec2 ( cl.a[0], cl.a[1] ), cvec2 ( cl.a[2], cl.a[3] ) };
	EGBuf_add(als, &lse);
      }

      als = removeShortLS(als, 0.05);

      als = pickExternalLS(als);

      double len0 = sumLength(als);

      als = pickLongestLS(als, 4);

      sortByAngle(als, gv(als));

      als = findCorners(als);

      double len1 = sumLength(als);

      if (als == NULL || closeToTriangle(als, 0.001) || als->size < 4 || len1 / len0 > 2 || !isConvex(als)) {
	if (als != NULL) EGBuf_dispose(als);
	continue;
      }

      rect_t rect = poseEstimation((ls_t *)als->ptr, gv(als), iw, ih, tanAOV);
      rect.status = 0;

      if (looksLikeAScreen(rect)) rect.status |= 1;

      EGBuf_add(ret, &rect);

      EGBuf_dispose(als);
    }

    for(int i=0;i<lsMapSize;i++) {
      int segid = keyArray[i];
      EGBuf *lsidSet = (EGBuf *)ArrayMap_get(lsMap, segid);
      EGBuf_dispose(lsidSet);
    }

    free(keyArray);
    ArrayMap_dispose(lsMap);
  }

  //

  for(int i=1;i<=n;i++) {
    if (ls[i].polyid == 0) continue;
    if (ls[i].leftPtr > 0) continue;

    EGBuf *als = EGBuf_init(sizeof(ls_t));

    for(int j=i;j > 0;j = ls[j].rightPtr) {
      const double LSTHRE = 32;
      vec2 e0 = cvec2 ( ls[j].x0, ls[j].y0 ), e1 = cvec2 ( ls[j].x1, ls[j].y1 );
      if (distanceSqu2(e0, e1) > LSTHRE * LSTHRE) {
	ls_t lse = { e0, e1 };
	EGBuf_add(als, &lse);
      }
    }

    als = removeShortLS(als, 0.05);

    als = pickExternalLS(als);

    double len0 = sumLength(als);

    als = pickLongestLS(als, 4);

    sortByAngle(als, gv(als));

    als = findCorners(als);

    double len1 = sumLength(als);

    if (als == NULL || closeToTriangle(als, 0.001) || als->size < 4 || len1 / len0 > 2 || !isConvex(als)) {
      if (als != NULL) EGBuf_dispose(als);
      continue;
    }

    rect_t rect = poseEstimation((ls_t *)als->ptr, gv(als), iw, ih, tanAOV);
    rect.status = 2;

    if (looksLikeAScreen(rect)) rect.status |= 1;

    EGBuf_add(ret, &rect);

    EGBuf_dispose(als);
  }

  rect_t *rptr = (rect_t *)(ret->ptr);
  rptr->nItems = ret->size;

  ret->magic = 0;
  free(ret);

  return rptr;
}

//

rect_t *oclrect_executeOnce(oclrect_t *thiz, uint8_t *imgData, int ws, const double tanAOV) {
  assert(thiz->magic == MAGIC);

  const int page = 0;

  memcpy(thiz->hostiobuf[page][0], imgData, ws * thiz->ih);

  cl_event ev = genGPUTask(thiz, imgData, page, ws, thiz->queue, NULL);
  waitForEvent(ev);
  ce(clReleaseEvent(ev));

  memcpy(imgData, thiz->hostiobuf[page][0], ws * thiz->ih);

  rect_t *ret = executeCPUTask(thiz, page, tanAOV);

  return ret;
}

void oclrect_enqueueTask(oclrect_t *thiz, uint8_t *imgData, int ws) {
  assert(thiz->magic == MAGIC);

  const int page = 1 & thiz->nextPageToEnqueue;
  thiz->nextPageToEnqueue++;

  assert(thiz->events[page] == NULL);

  memcpy(thiz->hostiobuf[page][0], imgData, ws * thiz->ih);

  //clFinish(thiz->queue);

  thiz->events[page] = genGPUTask(thiz, imgData, page, ws, thiz->queue, NULL);
}

rect_t *oclrect_pollTask(oclrect_t *thiz, const double tanAOV) {
  assert(thiz->magic == MAGIC);

  const int page = 1 & thiz->nextPageToPoll;
  thiz->nextPageToPoll++;

  if (thiz->events[page] != NULL) waitForEvent(thiz->events[page]);

  ce(clReleaseEvent(thiz->events[page]));

  rect_t *ret = executeCPUTask(thiz, page, tanAOV);

  thiz->events[page] = NULL;

  return ret;
}
