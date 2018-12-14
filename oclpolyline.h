#if defined(__cplusplus)
extern "C" {
#endif

typedef struct oclpolyline_t {
  uint32_t magic;
  cl_device_id device;
  cl_context context;

  cl_kernel kernel_clear;
  int kid_clear;
  cl_kernel kernel_copy;
  int kid_copy;
  cl_kernel kernel_rand;
  int kid_rand;
  cl_kernel kernel_labelxPreprocess_int_int;
  int kid_labelxPreprocess_int_int;
  cl_kernel kernel_label8xMain_int_int;
  int kid_label8xMain_int_int;
  cl_kernel kernel_simpleJunction;
  int kid_simpleJunction;
  cl_kernel kernel_simpleConnect;
  int kid_simpleConnect;
  cl_kernel kernel_stringify;
  int kid_stringify;
  cl_kernel kernel_removeBranch;
  int kid_removeBranch;
  cl_kernel kernel_countEnds;
  int kid_countEnds;
  cl_kernel kernel_breakLoops;
  int kid_breakLoops;
  cl_kernel kernel_findEnds0;
  int kid_findEnds0;
  cl_kernel kernel_findEnds1;
  int kid_findEnds1;
  cl_kernel kernel_findEnds2;
  int kid_findEnds2;
  cl_kernel kernel_number;
  int kid_number;
  cl_kernel kernel_labelpl_preprocess;
  int kid_labelpl_preprocess;
  cl_kernel kernel_labelpl_main;
  int kid_labelpl_main;
  cl_kernel kernel_calcSize;
  int kid_calcSize;
  cl_kernel kernel_filterSize;
  int kid_filterSize;
  cl_kernel kernel_relabel_pass0;
  int kid_relabel_pass0;
  cl_kernel kernel_relabel_pass1;
  int kid_relabel_pass1;
  cl_kernel kernel_mkpl_pass0a;
  int kid_mkpl_pass0a;
  cl_kernel kernel_mkpl_pass0b;
  int kid_mkpl_pass0b;
  cl_kernel kernel_mkpl_pass1;
  int kid_mkpl_pass1;
  cl_kernel kernel_mkpl_pass2;
  int kid_mkpl_pass2;
  cl_kernel kernel_mkpl_pass3;
  int kid_mkpl_pass3;
  cl_kernel kernel_mkpl_pass4;
  int kid_mkpl_pass4;
  cl_kernel kernel_refine_pass0;
  int kid_refine_pass0;
  cl_kernel kernel_refine_pass1;
  int kid_refine_pass1;
  cl_kernel kernel_refine_pass2;
  int kid_refine_pass2;
  cl_kernel kernel_refine_pass3;
  int kid_refine_pass3;
} oclpolyline_t;

typedef struct linesegment_t {
  float x0, y0, x1, y1;
  int32_t startIndex, endIndex;
  int32_t leftPtr, rightPtr;
  int32_t startCount, endCount;
  int32_t maxDist;
  int32_t polyid;
  int32_t npix;
  int32_t level;
} linesegment_t;

oclpolyline_t *init_oclpolyline(cl_device_id device, cl_context context);
void dispose_oclpolyline(oclpolyline_t *thiz);

cl_event oclpolyline_execute(oclpolyline_t *thiz, cl_mem lsList, int lsListSize, cl_mem lsIdOut, cl_mem in, cl_mem tmp0, cl_mem tmp1, cl_mem tmp2, cl_mem tmp3, cl_mem tmp4, cl_mem tmp5, cl_mem tmp6, float minerror, int sizeThre, int iw, int ih, cl_command_queue queue, const cl_event *events);

#if defined(__cplusplus)
}
#endif
