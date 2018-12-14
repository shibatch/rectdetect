// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "helper.h"
#include "oclhelper.h"
#include "oclimgutil.h"

#include "oclimgutil_cl.h"

#define MAGIC 0xa640d893

oclimgutil_t *init_oclimgutil(cl_device_id device, cl_context context) {
  //char *source = readFileAsStr("oclimgutil.cl", 1024*1024);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, 0, NULL);
  simpleBuildProgram(program, device, "");
  //free(source);

  oclimgutil_t *thiz = (oclimgutil_t *)calloc(1, sizeof(oclimgutil_t));

  thiz->magic = MAGIC;
  thiz->device = device;
  thiz->context = context;

  thiz->kernel_clear = clCreateKernel(program, "clear", NULL);
  thiz->kid_clear = getNextKernelID();
  thiz->kernel_copy = clCreateKernel(program, "copy", NULL);
  thiz->kid_copy = getNextKernelID();
  thiz->kernel_cast_i_f = clCreateKernel(program, "cast_i_f", NULL);
  thiz->kid_cast_i_f = getNextKernelID();
  thiz->kernel_cast_c_i = clCreateKernel(program, "cast_c_i", NULL);
  thiz->kid_cast_c_i = getNextKernelID();
  thiz->kernel_threshold_i_i = clCreateKernel(program, "threshold_i_i", NULL);
  thiz->kid_threshold_i_i = getNextKernelID();
  thiz->kernel_threshold_f_f = clCreateKernel(program, "threshold_f_f", NULL);
  thiz->kid_threshold_f_f = getNextKernelID();
  thiz->kernel_threshold_f = clCreateKernel(program, "threshold_f", NULL);
  thiz->kid_threshold_f = getNextKernelID();
  thiz->kernel_rand = clCreateKernel(program, "rand", NULL);
  thiz->kid_rand = getNextKernelID();
  thiz->kernel_bgr2plab = clCreateKernel(program, "bgr2plab", NULL);
  thiz->kid_bgr2plab = getNextKernelID();
  thiz->kernel_plab2bgr = clCreateKernel(program, "plab2bgr", NULL);
  thiz->kid_plab2bgr = getNextKernelID();
  thiz->kernel_convert_bgr_luminancef = clCreateKernel(program, "convert_bgr_luminancef", NULL);
  thiz->kid_convert_bgr_luminancef = getNextKernelID();
  thiz->kernel_convert_bgr_lumaf = clCreateKernel(program, "convert_bgr_lumaf", NULL);
  thiz->kid_convert_bgr_lumaf = getNextKernelID();
  thiz->kernel_convert_bgr_labeli = clCreateKernel(program, "convert_bgr_labeli", NULL);
  thiz->kid_convert_bgr_labeli = getNextKernelID();
  thiz->kernel_pack_plab = clCreateKernel(program, "pack_plab", NULL);
  thiz->kid_pack_plab = getNextKernelID();
  thiz->kernel_unpack_plab = clCreateKernel(program, "unpack_plab", NULL);
  thiz->kid_unpack_plab = getNextKernelID();
  thiz->kernel_edgevec_plab = clCreateKernel(program, "edgevec_plab", NULL);
  thiz->kid_edgevec_plab = getNextKernelID();
  thiz->kernel_edgevec_f = clCreateKernel(program, "edgevec_f", NULL);
  thiz->kid_edgevec_f = getNextKernelID();
  thiz->kernel_edge_plab = clCreateKernel(program, "edge_plab", NULL);
  thiz->kid_edge_plab = getNextKernelID();
  thiz->kernel_edge_f_f = clCreateKernel(program, "edge_f_f", NULL);
  thiz->kid_edge_f_f = getNextKernelID();
  thiz->kernel_thinthres_f_f_f2 = clCreateKernel(program, "thinthres_f_f_f2", NULL);
  thiz->kid_thinthres_f_f_f2 = getNextKernelID();
  thiz->kernel_thincubic_float_float = clCreateKernel(program, "thincubic_float_float", NULL);
  thiz->kid_thincubic_float_float = getNextKernelID();
  thiz->kernel_labelxPreprocess_int_int = clCreateKernel(program, "labelxPreprocess_int_int", NULL);
  thiz->kid_labelxPreprocess_int_int = getNextKernelID();
  thiz->kernel_label8xMain_int_int = clCreateKernel(program, "label8xMain_int_int", NULL);
  thiz->kid_label8xMain_int_int = getNextKernelID();
  thiz->kernel_iirblur_f_f_pass0a = clCreateKernel(program, "iirblur_f_f_pass0a", NULL);
  thiz->kid_iirblur_f_f_pass0a = getNextKernelID();
  thiz->kernel_iirblur_f_f_pass0b = clCreateKernel(program, "iirblur_f_f_pass0b", NULL);
  thiz->kid_iirblur_f_f_pass0b = getNextKernelID();
  thiz->kernel_iirblur_f_f_pass2a = clCreateKernel(program, "iirblur_f_f_pass2a", NULL);
  thiz->kid_iirblur_f_f_pass2a = getNextKernelID();
  thiz->kernel_iirblur_f_f_pass2b = clCreateKernel(program, "iirblur_f_f_pass2b", NULL);
  thiz->kid_iirblur_f_f_pass2b = getNextKernelID();
  thiz->kernel_iirblur_f_f_pass1 = clCreateKernel(program, "iirblur_f_f_pass1", NULL);
  thiz->kid_iirblur_f_f_pass1 = getNextKernelID();
  thiz->kernel_iirblur_f_f_pass3 = clCreateKernel(program, "iirblur_f_f_pass3", NULL);
  thiz->kid_iirblur_f_f_pass3 = getNextKernelID();

  ce(clReleaseProgram(program));

  return thiz;
}

void dispose_oclimgutil(oclimgutil_t *thiz) {
  assert(thiz->magic == MAGIC);
  thiz->magic = 0;

  ce(clReleaseKernel(thiz->kernel_clear));
  ce(clReleaseKernel(thiz->kernel_copy));
  ce(clReleaseKernel(thiz->kernel_cast_i_f));
  ce(clReleaseKernel(thiz->kernel_cast_c_i));
  ce(clReleaseKernel(thiz->kernel_threshold_i_i));
  ce(clReleaseKernel(thiz->kernel_threshold_f_f));
  ce(clReleaseKernel(thiz->kernel_threshold_f));
  ce(clReleaseKernel(thiz->kernel_rand));
  ce(clReleaseKernel(thiz->kernel_bgr2plab));
  ce(clReleaseKernel(thiz->kernel_plab2bgr));
  ce(clReleaseKernel(thiz->kernel_convert_bgr_luminancef));
  ce(clReleaseKernel(thiz->kernel_convert_bgr_lumaf));
  ce(clReleaseKernel(thiz->kernel_convert_bgr_labeli));
  ce(clReleaseKernel(thiz->kernel_pack_plab));
  ce(clReleaseKernel(thiz->kernel_unpack_plab));
  ce(clReleaseKernel(thiz->kernel_edgevec_plab));
  ce(clReleaseKernel(thiz->kernel_edgevec_f));
  ce(clReleaseKernel(thiz->kernel_edge_plab));
  ce(clReleaseKernel(thiz->kernel_edge_f_f));
  ce(clReleaseKernel(thiz->kernel_thincubic_float_float));
  ce(clReleaseKernel(thiz->kernel_labelxPreprocess_int_int));
  ce(clReleaseKernel(thiz->kernel_label8xMain_int_int));
  ce(clReleaseKernel(thiz->kernel_iirblur_f_f_pass0a));
  ce(clReleaseKernel(thiz->kernel_iirblur_f_f_pass0b));
  ce(clReleaseKernel(thiz->kernel_iirblur_f_f_pass2a));
  ce(clReleaseKernel(thiz->kernel_iirblur_f_f_pass2b));
  ce(clReleaseKernel(thiz->kernel_iirblur_f_f_pass1));
  ce(clReleaseKernel(thiz->kernel_iirblur_f_f_pass3));

  free(thiz);
}

#define CLEARLS6B(x) ((size_t)((x + 31) & ~31))

cl_event oclimgutil_clear(oclimgutil_t *thiz, cl_mem out, int size, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  size = (size + 3) / 4;
  simpleSetKernelArg(thiz->kernel_clear, "Mi", out, size);
  return runKernel1Dx(queue, thiz->kernel_clear, thiz->kid_clear, size, events);
}

cl_event oclimgutil_copy(oclimgutil_t *thiz, cl_mem out, cl_mem in, int size, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  size = (size + 3) / 4;
  simpleSetKernelArg(thiz->kernel_copy, "MMi", out, in, size);
  return runKernel1Dx(queue, thiz->kernel_copy, thiz->kid_copy, size, events);
}

cl_event oclimgutil_cast_i_f(oclimgutil_t *thiz, cl_mem out, cl_mem in, float scale, int size, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_cast_i_f, "MMfi", out, in, scale, size);
  return runKernel1Dx(queue, thiz->kernel_cast_i_f, thiz->kid_cast_i_f, size, events);
}

cl_event oclimgutil_cast_c_i(oclimgutil_t *thiz, cl_mem out, cl_mem in, int size, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_cast_c_i, "MMi", out, in, size);
  return runKernel1Dx(queue, thiz->kernel_cast_c_i, thiz->kid_cast_c_i, size, events);
}

cl_event oclimgutil_threshold_i_i(oclimgutil_t *thiz, cl_mem out, cl_mem in, int vlow, int threshold, int vhigh, int size, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_threshold_i_i, "MMiiii", out, in, vlow, threshold, vhigh, size);
  return runKernel1Dx(queue, thiz->kernel_threshold_i_i, thiz->kid_threshold_i_i, size, events);
}

cl_event oclimgutil_threshold_f_f(oclimgutil_t *thiz, cl_mem out, cl_mem in, float vlow, float threshold, float vhigh, int size, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_threshold_f_f, "MMfffi", out, in, vlow, threshold, vhigh, size);
  return runKernel1Dx(queue, thiz->kernel_threshold_f_f, thiz->kid_threshold_f_f, size, events);
}

cl_event oclimgutil_rand(oclimgutil_t *thiz, cl_mem out, int size, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  size = (size + 3) / 4;
  simpleSetKernelArg(thiz->kernel_rand, "Mi", out, size);
  return runKernel1Dx(queue, thiz->kernel_rand, thiz->kid_rand, size, events);
}

cl_event oclimgutil_convert_bgr_luminancef(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_convert_bgr_luminancef, "MMiii", out, in, iw, ih, ws);
  return runKernel2Dx(queue, thiz->kernel_convert_bgr_luminancef, thiz->kid_convert_bgr_luminancef, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_convert_bgr_lumaf(oclimgutil_t *thiz, cl_mem out, cl_mem in, float f, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_convert_bgr_lumaf, "MMfiii", out, in, f, iw, ih, ws);
  return runKernel2Dx(queue, thiz->kernel_convert_bgr_lumaf, thiz->kid_convert_bgr_lumaf, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_convert_bgr_labeli(oclimgutil_t *thiz, cl_mem out, cl_mem in, int bgc, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_convert_bgr_labeli, "MMiiii", out, in, bgc, iw, ih, ws);
  return runKernel2Dx(queue, thiz->kernel_convert_bgr_labeli, thiz->kid_convert_bgr_labeli, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_edge_f_f(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_edge_f_f, "MMii", out, in, iw, ih);
  return runKernel2Dx(queue, thiz->kernel_edge_f_f, thiz->kid_edge_f_f, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_edgevec_f2_f(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_edgevec_f, "MMii", out, in, iw, ih);
  return runKernel2Dx(queue, thiz->kernel_edgevec_f, thiz->kid_edgevec_f, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_thinthres_f_f_f2(oclimgutil_t *thiz, cl_mem out, cl_mem in, cl_mem vec, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_thinthres_f_f_f2, "MMMii", out, in, vec, iw, ih);
  return runKernel2Dx(queue, thiz->kernel_thinthres_f_f_f2, thiz->kid_thinthres_f_f_f2, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_thincubic_f_f_f2(oclimgutil_t *thiz, cl_mem out, cl_mem in, cl_mem vec, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_thincubic_float_float, "MMMii", out, in, vec, iw, ih);
  return runKernel2Dx(queue, thiz->kernel_thincubic_float_float, thiz->kid_thincubic_float_float, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_label8x_int_int(oclimgutil_t *thiz, cl_mem out, cl_mem in, cl_mem tmp, int bgc, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  const int MAXPASS = 10;

  simpleSetKernelArg(thiz->kernel_labelxPreprocess_int_int, "MMMiiii", out, in, tmp, MAXPASS, bgc, iw, ih);
  cl_event e0 = runKernel2Dx(queue, thiz->kernel_labelxPreprocess_int_int, thiz->kid_labelxPreprocess_int_int, CLEARLS6B(iw), CLEARLS6B(ih), events);

  for(int i=1;i<=MAXPASS;i++) {
    cl_event ev[] = { e0, NULL };
    simpleSetKernelArg(thiz->kernel_label8xMain_int_int, "MMMiii", out, in, tmp, i, iw, ih);
    e0 = runKernel2Dx(queue, thiz->kernel_label8xMain_int_int, thiz->kid_label8xMain_int_int, CLEARLS6B(iw), CLEARLS6B(ih), events == NULL ? NULL : ev);
  }

  return e0;
}

cl_event oclimgutil_iirblur_f_f(oclimgutil_t *thiz, cl_mem obuf, cl_mem ibuf, cl_mem tmp0, cl_mem tmp1, int r, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  cl_event e0 = oclimgutil_clear(thiz, tmp0, iw * ih * 4, queue, events);
  cl_event e1 = oclimgutil_clear(thiz, tmp1, iw * ih * 4, queue, events);

  simpleSetKernelArg(thiz->kernel_iirblur_f_f_pass0a, "MMiii", tmp0, ibuf, r, iw, ih);
  cl_event ev0[] = { e0, NULL };
  e0 = runKernel1Dx(queue, thiz->kernel_iirblur_f_f_pass0a, thiz->kid_iirblur_f_f_pass0a, CLEARLS6B(ih), events == NULL ? NULL : ev0);

  simpleSetKernelArg(thiz->kernel_iirblur_f_f_pass0b, "MMiii", tmp1, ibuf, r, iw, ih);
  cl_event ev1[] = { e1, NULL };
  e0 = runKernel1Dx(queue, thiz->kernel_iirblur_f_f_pass0b, thiz->kid_iirblur_f_f_pass0b, CLEARLS6B(ih), events == NULL ? NULL : ev1);

  simpleSetKernelArg(thiz->kernel_iirblur_f_f_pass1, "MMMMiii", obuf, tmp0, tmp1, ibuf, r, iw, ih);
  cl_event ev01[] = { e0, e1, NULL };
  e1 = runKernel2Dx(queue, thiz->kernel_iirblur_f_f_pass1, thiz->kid_iirblur_f_f_pass1, CLEARLS6B(iw), CLEARLS6B(ih), events == NULL ? NULL : ev01);

  simpleSetKernelArg(thiz->kernel_iirblur_f_f_pass2a, "MMiii", obuf, tmp0, r, iw, ih);
  ev1[0] = e1;
  e0 = runKernel1Dx(queue, thiz->kernel_iirblur_f_f_pass2a, thiz->kid_iirblur_f_f_pass2a, CLEARLS6B(iw), events == NULL ? NULL : ev1);

  simpleSetKernelArg(thiz->kernel_iirblur_f_f_pass2b, "MMiii", obuf, tmp1, r, iw, ih);
  ev1[0] = e1;
  e1 = runKernel1Dx(queue, thiz->kernel_iirblur_f_f_pass2b, thiz->kid_iirblur_f_f_pass2b, CLEARLS6B(iw), events == NULL ? NULL : ev1);

  simpleSetKernelArg(thiz->kernel_iirblur_f_f_pass3, "MMMiii", obuf, tmp0, tmp1, r, iw, ih);
  ev01[0] = e0; ev01[1] = e1;
  e0 = runKernel2Dx(queue, thiz->kernel_iirblur_f_f_pass3, thiz->kid_iirblur_f_f_pass3, CLEARLS6B(iw), CLEARLS6B(ih), events == NULL ? NULL : ev01);

  return e0;
}

cl_event oclimgutil_convert_plab_bgr(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_bgr2plab, "MMiii", out, in, iw, ih, ws);
  return runKernel2Dx(queue, thiz->kernel_bgr2plab, thiz->kid_bgr2plab, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_convert_bgr_plab(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_plab2bgr, "MMiii", out, in, iw, ih, ws);
  return runKernel2Dx(queue, thiz->kernel_plab2bgr, thiz->kid_plab2bgr, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_unpack_f_f_f_plab(oclimgutil_t *thiz, cl_mem out0, cl_mem out1, cl_mem out2, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_unpack_plab, "MMMMii", out0, out1, out2, in, iw, ih);
  return runKernel2Dx(queue, thiz->kernel_unpack_plab, thiz->kid_unpack_plab, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_pack_plab_f_f_f(oclimgutil_t *thiz, cl_mem out, cl_mem in0, cl_mem in1, cl_mem in2, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_pack_plab, "MMMMii", out, in0, in1, in2, iw, ih);
  return runKernel2Dx(queue, thiz->kernel_pack_plab, thiz->kid_pack_plab, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_edgevec_f2_plab(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_edgevec_plab, "MMii", out, in, iw, ih);
  return runKernel2Dx(queue, thiz->kernel_edgevec_plab, thiz->kid_edgevec_plab, CLEARLS6B(iw), CLEARLS6B(ih), events);
}

cl_event oclimgutil_edge_f_plab(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events) {
  assert(thiz->magic == MAGIC);
  simpleSetKernelArg(thiz->kernel_edge_plab, "MMii", out, in, iw, ih);
  return runKernel2Dx(queue, thiz->kernel_edge_plab, thiz->kid_edge_plab, CLEARLS6B(iw), CLEARLS6B(ih), events);
}
