// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "helper.h"
#include "oclhelper.h"
#include "oclpolyline.h"

#include "oclpolyline_cl.h"

#define MAGIC 0x808f3801

oclpolyline_t *init_oclpolyline(cl_device_id device, cl_context context) {
  char param_value[1024];
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1000, param_value, NULL);
  int enableAtomics64 = strstr(param_value, "cl_khr_int64_base_atomics") != NULL || strstr(param_value, "cl_nv_compiler_options") != NULL;

  //char *source = readFileAsStr("oclpolyline.cl", 1024*1024);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, 0, NULL);
  simpleBuildProgram(program, device, enableAtomics64 ? "-DENABLE_ATOMICS64" : "");
  //free(source);

  oclpolyline_t *thiz = (oclpolyline_t *)calloc(1, sizeof(oclpolyline_t));

  thiz->magic = MAGIC;
  thiz->device = device;
  thiz->context = context;

  thiz->kernel_clear = clCreateKernel(program, "clear", NULL);
  thiz->kid_clear = getNextKernelID();
  thiz->kernel_copy = clCreateKernel(program, "copy", NULL);
  thiz->kid_copy = getNextKernelID();
  thiz->kernel_rand = clCreateKernel(program, "rand", NULL);
  thiz->kid_rand = getNextKernelID();
  thiz->kernel_labelxPreprocess_int_int = clCreateKernel(program, "labelxPreprocess_int_int", NULL);
  thiz->kid_labelxPreprocess_int_int = getNextKernelID();
  thiz->kernel_label8xMain_int_int = clCreateKernel(program, "label8xMain_int_int", NULL);
  thiz->kid_label8xMain_int_int = getNextKernelID();
  thiz->kernel_simpleJunction = clCreateKernel(program, "simpleJunction", NULL);
  thiz->kid_simpleJunction = getNextKernelID();
  thiz->kernel_simpleConnect = clCreateKernel(program, "simpleConnect", NULL);
  thiz->kid_simpleConnect = getNextKernelID();
  thiz->kernel_stringify = clCreateKernel(program, "stringify", NULL);
  thiz->kid_stringify = getNextKernelID();
  thiz->kernel_removeBranch = clCreateKernel(program, "removeBranch", NULL);
  thiz->kid_removeBranch = getNextKernelID();
  thiz->kernel_countEnds = clCreateKernel(program, "countEnds", NULL);
  thiz->kid_countEnds = getNextKernelID();
  thiz->kernel_breakLoops = clCreateKernel(program, "breakLoops", NULL);
  thiz->kid_breakLoops = getNextKernelID();
  thiz->kernel_findEnds0 = clCreateKernel(program, "findEnds0", NULL);
  thiz->kid_findEnds0 = getNextKernelID();
  thiz->kernel_findEnds1 = clCreateKernel(program, "findEnds1", NULL);
  thiz->kid_findEnds1 = getNextKernelID();
  thiz->kernel_findEnds2 = clCreateKernel(program, "findEnds2", NULL);
  thiz->kid_findEnds2 = getNextKernelID();
  thiz->kernel_number = clCreateKernel(program, "number", NULL);
  thiz->kid_number = getNextKernelID();
  thiz->kernel_labelpl_preprocess = clCreateKernel(program, "labelpl_preprocess", NULL);
  thiz->kid_labelpl_preprocess = getNextKernelID();
  thiz->kernel_labelpl_main = clCreateKernel(program, "labelpl_main", NULL);
  thiz->kid_labelpl_main = getNextKernelID();
  thiz->kernel_calcSize = clCreateKernel(program, "calcSize", NULL);
  thiz->kid_calcSize = getNextKernelID();
  thiz->kernel_filterSize = clCreateKernel(program, "filterSize", NULL);
  thiz->kid_filterSize = getNextKernelID();
  thiz->kernel_relabel_pass0 = clCreateKernel(program, "relabel_pass0", NULL);
  thiz->kid_relabel_pass0 = getNextKernelID();
  thiz->kernel_relabel_pass1 = clCreateKernel(program, "relabel_pass1", NULL);
  thiz->kid_relabel_pass1 = getNextKernelID();
  thiz->kernel_mkpl_pass0a = clCreateKernel(program, "mkpl_pass0a", NULL);
  thiz->kid_mkpl_pass0a = getNextKernelID();
  thiz->kernel_mkpl_pass0b = clCreateKernel(program, "mkpl_pass0b", NULL);
  thiz->kid_mkpl_pass0b = getNextKernelID();
  thiz->kernel_mkpl_pass1 = clCreateKernel(program, "mkpl_pass1", NULL);
  thiz->kid_mkpl_pass1 = getNextKernelID();
  thiz->kernel_mkpl_pass2 = clCreateKernel(program, "mkpl_pass2", NULL);
  thiz->kid_mkpl_pass2 = getNextKernelID();
  thiz->kernel_mkpl_pass3 = clCreateKernel(program, "mkpl_pass3", NULL);
  thiz->kid_mkpl_pass3 = getNextKernelID();
  thiz->kernel_mkpl_pass4 = clCreateKernel(program, "mkpl_pass4", NULL);
  thiz->kid_mkpl_pass4 = getNextKernelID();
  thiz->kernel_refine_pass0 = clCreateKernel(program, "refine_pass0", NULL);
  thiz->kid_refine_pass0 = getNextKernelID();
  thiz->kernel_refine_pass1 = clCreateKernel(program, "refine_pass1", NULL);
  thiz->kid_refine_pass1 = getNextKernelID();
  thiz->kernel_refine_pass2 = clCreateKernel(program, "refine_pass2", NULL);
  thiz->kid_refine_pass2 = getNextKernelID();
  thiz->kernel_refine_pass3 = clCreateKernel(program, "refine_pass3", NULL);
  thiz->kid_refine_pass3 = getNextKernelID();

  ce(clReleaseProgram(program));

  return thiz;
}

void dispose_oclpolyline(oclpolyline_t *thiz) {
  assert(thiz->magic == MAGIC);
  thiz->magic = 0;

  ce(clReleaseKernel(thiz->kernel_clear));
  ce(clReleaseKernel(thiz->kernel_copy));
  ce(clReleaseKernel(thiz->kernel_rand));
  ce(clReleaseKernel(thiz->kernel_labelxPreprocess_int_int));
  ce(clReleaseKernel(thiz->kernel_label8xMain_int_int));
  ce(clReleaseKernel(thiz->kernel_simpleJunction));
  ce(clReleaseKernel(thiz->kernel_simpleConnect));
  ce(clReleaseKernel(thiz->kernel_stringify));
  ce(clReleaseKernel(thiz->kernel_removeBranch));
  ce(clReleaseKernel(thiz->kernel_countEnds));
  ce(clReleaseKernel(thiz->kernel_breakLoops));
  ce(clReleaseKernel(thiz->kernel_findEnds0));
  ce(clReleaseKernel(thiz->kernel_findEnds1));
  ce(clReleaseKernel(thiz->kernel_findEnds2));
  ce(clReleaseKernel(thiz->kernel_number));
  ce(clReleaseKernel(thiz->kernel_labelpl_preprocess));
  ce(clReleaseKernel(thiz->kernel_labelpl_main));
  ce(clReleaseKernel(thiz->kernel_calcSize));
  ce(clReleaseKernel(thiz->kernel_filterSize));
  ce(clReleaseKernel(thiz->kernel_relabel_pass0));
  ce(clReleaseKernel(thiz->kernel_relabel_pass1));
  ce(clReleaseKernel(thiz->kernel_mkpl_pass0a));
  ce(clReleaseKernel(thiz->kernel_mkpl_pass0b));
  ce(clReleaseKernel(thiz->kernel_mkpl_pass1));
  ce(clReleaseKernel(thiz->kernel_mkpl_pass2));
  ce(clReleaseKernel(thiz->kernel_mkpl_pass3));
  ce(clReleaseKernel(thiz->kernel_mkpl_pass4));
  ce(clReleaseKernel(thiz->kernel_refine_pass0));
  ce(clReleaseKernel(thiz->kernel_refine_pass1));
  ce(clReleaseKernel(thiz->kernel_refine_pass2));
  ce(clReleaseKernel(thiz->kernel_refine_pass3));

  free(thiz);
}

#define CLEARLS6B(x) ((size_t)((x + 31) & ~31))

static cl_event runKernel1Dy(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, const cl_event *events) {
  return runKernel1Dx(queue, kernel, kernelID, CLEARLS6B(ws1), events);
}

static cl_event runKernel2Dy(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, size_t ws2, const cl_event *events) {
  return runKernel2Dx(queue, kernel, kernelID, CLEARLS6B(ws1), CLEARLS6B(ws2), events);
}

static cl_event oclpolyline_label8x_int_int(oclpolyline_t *thiz, cl_mem out, cl_mem in, cl_mem tmp, int bgc, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  const int MAXPASS = 10;
  cl_event ev[] = { NULL, NULL };

  simpleSetKernelArg(thiz->kernel_labelxPreprocess_int_int, "MMMiiii", out, in, tmp, MAXPASS, bgc, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_labelxPreprocess_int_int, thiz->kid_labelxPreprocess_int_int, iw, ih, events);

  for(int i=1;i<=MAXPASS;i++) {
    simpleSetKernelArg(thiz->kernel_label8xMain_int_int, "MMMiii", out, in, tmp, i, iw, ih);
    ev[0] = runKernel2Dy(queue, thiz->kernel_label8xMain_int_int, thiz->kid_label8xMain_int_int, iw, ih, events == NULL ? NULL : ev);
  }

  return ev[0];
}

static cl_event oclpolyline_labelpl(oclpolyline_t *thiz, cl_mem out, cl_mem in, cl_mem tmp, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  const int N = 12;
  cl_event ev[] = { NULL, NULL };

  simpleSetKernelArg(thiz->kernel_labelpl_preprocess, "MMMiii", out, in, tmp, N, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_labelpl_preprocess, thiz->kid_labelpl_preprocess, iw, ih, events);

  for(int i=0;i<N-1;i++) {
    simpleSetKernelArg(thiz->kernel_labelpl_main, "MMMiii", out, in, tmp, i+1, iw, ih);
    ev[0] = runKernel2Dy(queue, thiz->kernel_labelpl_main, thiz->kid_labelpl_main, iw, ih, events == NULL ? NULL : ev);
  }

  return ev[0];
}

static cl_event oclpolyline_mkpl(oclpolyline_t *thiz, cl_mem lsList, cl_mem tmpBig, int lsListSize, cl_mem labelinout, cl_mem numberin, cl_mem randtmp, cl_mem tmp2, cl_mem flags, float minerror, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  const int N = 16;
  cl_event ev[] = { NULL, NULL };

  simpleSetKernelArg(thiz->kernel_clear, "Mi", lsList, (lsListSize+3) / 4);
  ev[0] = runKernel1Dy(queue, thiz->kernel_clear, thiz->kid_clear, (lsListSize+3) / 4, events);

  simpleSetKernelArg(thiz->kernel_mkpl_pass0a, "MiMMMiii", lsList, lsListSize, numberin, labelinout, flags, N, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_mkpl_pass0a, thiz->kid_mkpl_pass0a, iw, ih, events == NULL ? NULL : ev);
  simpleSetKernelArg(thiz->kernel_mkpl_pass0b, "MiMMii", lsList, lsListSize, numberin, labelinout, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_mkpl_pass0b, thiz->kid_mkpl_pass0b, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_rand, "Mli", randtmp, (uint64_t)0, iw * ih);
  runKernel1Dy(queue, thiz->kernel_rand, thiz->kid_rand, iw * ih, NULL);

  for(int i=0;i<N-1;i++) {
    simpleSetKernelArg(thiz->kernel_mkpl_pass1, "MiMMMMMiii", lsList, lsListSize, tmp2, numberin, labelinout, randtmp, flags, i+1, iw, ih);
    ev[0] = runKernel2Dy(queue, thiz->kernel_mkpl_pass1, thiz->kid_mkpl_pass1, iw, ih, events == NULL ? NULL : ev);

    simpleSetKernelArg(thiz->kernel_copy, "MMi", tmpBig, lsList, (lsListSize+3)/4);
    ev[0] = runKernel1Dy(queue, thiz->kernel_copy, thiz->kid_copy, (lsListSize+3)/4, events == NULL ? NULL : ev);

    simpleSetKernelArg(thiz->kernel_mkpl_pass2, "MMiMMMMMifii", lsList, tmpBig, lsListSize, tmp2, numberin, labelinout, randtmp, flags, i+1, minerror, iw, ih);
    ev[0] = runKernel2Dy(queue, thiz->kernel_mkpl_pass2, thiz->kid_mkpl_pass2, iw, ih, events == NULL ? NULL : ev);
    simpleSetKernelArg(thiz->kernel_mkpl_pass3, "MiMMMiii", lsList, lsListSize, numberin, labelinout, flags, i+1, iw, ih);
    ev[0] = runKernel2Dy(queue, thiz->kernel_mkpl_pass3, thiz->kid_mkpl_pass3, iw, ih, events == NULL ? NULL : ev);
  }

  return ev[0];
}

cl_event oclpolyline_execute(oclpolyline_t *thiz, cl_mem lsList, int lsListSize, cl_mem lsIdOut, cl_mem in, cl_mem tmpBig, cl_mem tmp0, cl_mem tmp1, cl_mem tmp2, cl_mem tmp3, cl_mem tmp4, cl_mem tmp5, float minerror, int sizeThre, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  cl_event ev[] = { NULL, NULL };

  simpleSetKernelArg(thiz->kernel_simpleJunction, "MMii", lsIdOut, in, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_simpleJunction, thiz->kid_simpleJunction, iw, ih, events);

  simpleSetKernelArg(thiz->kernel_simpleConnect, "MMii", tmp2, lsIdOut, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_simpleConnect, thiz->kid_simpleConnect, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_stringify, "MMiii", tmp1, tmp2, 0, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_stringify, thiz->kid_stringify, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_stringify, "MMiii", tmp2, tmp1, 1, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_stringify, thiz->kid_stringify, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_removeBranch, "MMii", tmp1, tmp2, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_removeBranch, thiz->kid_removeBranch, iw, ih, events == NULL ? NULL : ev);

  oclpolyline_label8x_int_int(thiz, lsIdOut, tmp1, tmp2, 0, iw, ih, queue, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_simpleJunction, "MMii", tmp2, tmp1, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_simpleJunction, thiz->kid_simpleJunction, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_clear, "Mi", tmp3, iw * ih);
  ev[0] = runKernel1Dy(queue, thiz->kernel_clear, thiz->kid_clear, iw * ih, events == NULL ? NULL : ev);
  simpleSetKernelArg(thiz->kernel_countEnds, "MMMii", tmp3, tmp2, lsIdOut, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_countEnds, thiz->kid_countEnds, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_breakLoops, "MMMii", tmp1, lsIdOut, tmp3, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_breakLoops, thiz->kid_breakLoops, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_findEnds0, "MMMMii", tmp0, tmp2, tmpBig, lsIdOut, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_findEnds0, thiz->kid_findEnds0, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_findEnds1, "MMMMMMiii", tmp3, tmp4, tmpBig, tmp0, tmp2, lsIdOut, 0, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_findEnds1, thiz->kid_findEnds1, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_findEnds1, "MMMMMMiii", tmp0, tmp2, tmpBig, tmp3, tmp4, lsIdOut, 1, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_findEnds1, thiz->kid_findEnds1, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_findEnds1, "MMMMMMiii", tmp3, tmp4, tmpBig, tmp0, tmp2, lsIdOut, 0, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_findEnds1, thiz->kid_findEnds1, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_findEnds1, "MMMMMMiii", tmp0, tmp2, tmpBig, tmp3, tmp4, lsIdOut, 1, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_findEnds1, thiz->kid_findEnds1, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_findEnds2, "MMMMMii", tmpBig, tmp4, tmp0, tmp2, lsIdOut, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_findEnds2, thiz->kid_findEnds2, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_number, "MMMMii", tmp2, tmp3, tmpBig, tmp4, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_number, thiz->kid_number, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_number, "MMMMii", tmpBig, tmp4, tmp2, tmp3, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_number, thiz->kid_number, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_number, "MMMMii", tmp2, tmp3, tmpBig, tmp4, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_number, thiz->kid_number, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_copy, "MMi", tmp1, tmp2, iw * ih);
  ev[0] = runKernel1Dy(queue, thiz->kernel_copy, thiz->kid_copy, iw * ih, events == NULL ? NULL : ev);

  oclpolyline_labelpl(thiz, tmpBig, tmp1, tmp3, iw, ih, queue, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_clear, "Mi", tmp1, iw * ih);
  ev[0] = runKernel1Dy(queue, thiz->kernel_clear, thiz->kid_clear, iw * ih, events == NULL ? NULL : ev);
  simpleSetKernelArg(thiz->kernel_calcSize, "MMii", tmp1, tmpBig, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_calcSize, thiz->kid_calcSize, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_filterSize, "MMMiii", lsIdOut, tmpBig, tmp1, sizeThre, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_filterSize, thiz->kid_filterSize, iw, ih, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_clear, "Mi", tmpBig, iw * ih);
  ev[0] = runKernel1Dy(queue, thiz->kernel_clear, thiz->kid_clear, iw * ih * 4, events == NULL ? NULL : ev);
  simpleSetKernelArg(thiz->kernel_relabel_pass0, "MMii", tmpBig, lsIdOut, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_relabel_pass0, thiz->kid_relabel_pass0, iw, ih, events == NULL ? NULL : ev);
  simpleSetKernelArg(thiz->kernel_relabel_pass1, "MMii", lsIdOut, tmpBig, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_relabel_pass1, thiz->kid_relabel_pass1, iw, ih, events == NULL ? NULL : ev);

  oclpolyline_mkpl(thiz, lsList, tmpBig, lsListSize, lsIdOut, tmp2, tmp5, tmp3, tmp4, minerror, iw, ih, queue, events == NULL ? NULL : ev);

  simpleSetKernelArg(thiz->kernel_refine_pass0, "MM", tmpBig, lsList, lsListSize / sizeof(linesegment_t));
  ev[0] = runKernel1Dy(queue, thiz->kernel_refine_pass0, thiz->kid_refine_pass0, iw * ih, events == NULL ? NULL : ev);
  simpleSetKernelArg(thiz->kernel_refine_pass1, "MMMii", tmpBig, lsList, lsIdOut, iw, ih);
  ev[0] = runKernel2Dy(queue, thiz->kernel_refine_pass1, thiz->kid_refine_pass1, iw, ih, events == NULL ? NULL : ev);
  simpleSetKernelArg(thiz->kernel_refine_pass2, "MM", tmpBig, lsList);
  ev[0] = runKernel1Dy(queue, thiz->kernel_refine_pass2, thiz->kid_refine_pass2, lsListSize / sizeof(linesegment_t), events == NULL ? NULL : ev);
  simpleSetKernelArg(thiz->kernel_refine_pass3, "M", lsList);
  ev[0] = runKernel1Dy(queue, thiz->kernel_refine_pass3, thiz->kid_refine_pass3, lsListSize / sizeof(linesegment_t), events == NULL ? NULL : ev);

  return ev[0];
}
