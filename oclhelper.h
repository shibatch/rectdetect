// Written by Naoki Shibata shibatch.sf.net@gmail.com 
// http://ito-lab.naist.jp/~n-sibata/

// This software is in public domain. You can use and modify this code
// for any purpose without any obligation.


#if defined(__cplusplus)
extern "C" {
#endif

const char *clStrError(int c);
cl_int checkError(cl_int ret, const char *s);
cl_int ce(cl_int ret);

char *getDeviceName(cl_device_id device);
cl_device_id simpleGetDevice(int did);
int simpleGetDevices(cl_device_id *devices, int maxDevices);
cl_context simpleCreateContext(cl_device_id device);
int simpleBuildProgram(cl_program program, cl_device_id device, const char *optionString);
void simpleSetKernelArg(cl_kernel kernel, const char *format, ...);
cl_event runKernel1D(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, int nev, ...);
cl_event runKernel2D(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, size_t ws2, int nev, ...);
cl_event runKernel1Dx(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, const cl_event *);
cl_event runKernel2Dx(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, size_t ws2, const cl_event *);

void waitForEvent(cl_event ev);

void clearPlan();
int loadPlan(const char *fn, cl_device_id device);
void savePlan(const char *fn, cl_device_id device);
void startProfiling(size_t ws1, size_t ws2, size_t ws3);
void finishProfiling();
void showPlan();

void *allocatePinnedMemory(size_t z, cl_context context, cl_command_queue queue);
void freePinnedMemory(void *p, cl_context context, cl_command_queue queue);

int getNextKernelID();

#if defined(__cplusplus)
}
#endif
