#if defined(__cplusplus)
extern "C" {
#endif

typedef struct oclimgutil_t {
  uint32_t magic;
  cl_device_id device;
  cl_context context;

  cl_kernel kernel_clear;
  int kid_clear;
  cl_kernel kernel_copy;
  int kid_copy;
  cl_kernel kernel_cast_i_f;
  int kid_cast_i_f;
  cl_kernel kernel_cast_c_i;
  int kid_cast_c_i;
  cl_kernel kernel_threshold_i_i;
  int kid_threshold_i_i;
  cl_kernel kernel_threshold_f_f;
  int kid_threshold_f_f;
  cl_kernel kernel_threshold_f;
  int kid_threshold_f;
  cl_kernel kernel_rand;
  int kid_rand;
  cl_kernel kernel_bgr2plab;
  int kid_bgr2plab;
  cl_kernel kernel_plab2bgr;
  int kid_plab2bgr;
  cl_kernel kernel_convert_bgr_luminancef;
  int kid_convert_bgr_luminancef;
  cl_kernel kernel_convert_bgr_lumaf;
  int kid_convert_bgr_lumaf;
  cl_kernel kernel_convert_bgr_labeli;
  int kid_convert_bgr_labeli;
  cl_kernel kernel_pack_plab;
  int kid_pack_plab;
  cl_kernel kernel_unpack_plab;
  int kid_unpack_plab;
  cl_kernel kernel_edgevec_plab;
  int kid_edgevec_plab;
  cl_kernel kernel_edgevec_f;
  int kid_edgevec_f;
  cl_kernel kernel_edge_plab;
  int kid_edge_plab;
  cl_kernel kernel_edge_f_f;
  int kid_edge_f_f;
  cl_kernel kernel_thinthres_f_f_f2;
  int kid_thinthres_f_f_f2;
  cl_kernel kernel_thincubic_float_float;
  int kid_thincubic_float_float;
  cl_kernel kernel_labelxPreprocess_int_int;
  int kid_labelxPreprocess_int_int;
  cl_kernel kernel_label8xMain_int_int;
  int kid_label8xMain_int_int;
  cl_kernel kernel_iirblur_f_f_pass0a;
  int kid_iirblur_f_f_pass0a;
  cl_kernel kernel_iirblur_f_f_pass0b;
  int kid_iirblur_f_f_pass0b;
  cl_kernel kernel_iirblur_f_f_pass2a;
  int kid_iirblur_f_f_pass2a;
  cl_kernel kernel_iirblur_f_f_pass2b;
  int kid_iirblur_f_f_pass2b;
  cl_kernel kernel_iirblur_f_f_pass1;
  int kid_iirblur_f_f_pass1;
  cl_kernel kernel_iirblur_f_f_pass3;
  int kid_iirblur_f_f_pass3;
} oclimgutil_t;

oclimgutil_t *init_oclimgutil(cl_device_id device, cl_context context);
void dispose_oclimgutil(oclimgutil_t *thiz);

cl_event oclimgutil_clear(oclimgutil_t *thiz, cl_mem out, int size, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_copy(oclimgutil_t *thiz, cl_mem out, cl_mem in, int size, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_cast_i_f(oclimgutil_t *thiz, cl_mem out, cl_mem in, float scale, int size, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_cast_c_i(oclimgutil_t *thiz, cl_mem out, cl_mem in, int size, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_threshold_i_i(oclimgutil_t *thiz, cl_mem out, cl_mem in, int vlow, int threshold, int vhigh, int size, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_threshold_f_f(oclimgutil_t *thiz, cl_mem out, cl_mem in, float vlow, float threshold, float vhigh, int size, cl_command_queue queue, const cl_event *event);
cl_event oclimgutil_rand(oclimgutil_t *thiz, cl_mem out, int size, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_convert_bgr_luminancef(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_convert_bgr_lumaf(oclimgutil_t *thiz, cl_mem out, cl_mem in, float f, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_convert_bgr_labeli(oclimgutil_t *thiz, cl_mem out, cl_mem in, int bgc, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_edge_f_f(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_edgevec_f2_f(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_thinthres_f_f_f2(oclimgutil_t *thiz, cl_mem out, cl_mem in, cl_mem vec, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_thincubic_f_f_f2(oclimgutil_t *thiz, cl_mem out, cl_mem in, cl_mem vec, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_label8x_int_int(oclimgutil_t *thiz, cl_mem out, cl_mem in, cl_mem tmp, int bgc, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_iirblur_f_f(oclimgutil_t *thiz, cl_mem obuf, cl_mem ibuf, cl_mem tmp0, cl_mem tmp1, int r, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_convert_plab_bgr(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_convert_bgr_plab(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, int ws, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_unpack_f_f_f_plab(oclimgutil_t *thiz, cl_mem out0, cl_mem out1, cl_mem out2, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_pack_plab_f_f_f(oclimgutil_t *thiz, cl_mem out, cl_mem in0, cl_mem in1, cl_mem in2, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_edgevec_f2_plab(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events);
cl_event oclimgutil_edge_f_plab(oclimgutil_t *thiz, cl_mem out, cl_mem in, int iw, int ih, cl_command_queue queue, const cl_event *events);

#if defined(__cplusplus)
}
#endif
