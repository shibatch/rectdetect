// This file was written by Naoki Shibata in 2018.
// No copyright is claimed, and the content in this file is hereby placed in the public domain.

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <assert.h>

#ifdef _MSC_VER
#include <Windows.h>
#include <MmSystem.h>
#include <sys/types.h>
#include <sys/timeb.h>
#else
#include <unistd.h>
#include <sys/time.h>
#include <execinfo.h>
#endif

#include <CL/cl.h>

#include "helper.h"
#include "oclhelper.h"

static const char *errorStrings[] = {
  "CL_SUCCESS", // 0
  "CL_DEVICE_NOT_FOUND", // -1
  "CL_DEVICE_NOT_AVAILABLE",
  "CL_COMPILER_NOT_AVAILABLE",
  "CL_MEM_OBJECT_ALLOCATION_FAILURE",
  "CL_OUT_OF_RESOURCES",
  "CL_OUT_OF_HOST_MEMORY",
  "CL_PROFILING_INFO_NOT_AVAILABLE",
  "CL_MEM_COPY_OVERLAP",
  "CL_IMAGE_FORMAT_MISMATCH",
  "CL_IMAGE_FORMAT_NOT_SUPPORTED", // -10
  "CL_BUILD_PROGRAM_FAILURE",
  "CL_MAP_FAILURE",
  "CL_MISALIGNED_SUB_BUFFER_OFFSET",
  "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
  "CL_COMPILE_PROGRAM_FAILURE",
  "CL_LINKER_NOT_AVAILABLE",
  "CL_LINK_PROGRAM_FAILURE",
  "CL_DEVICE_PARTITION_FAILED",
  "CL_KERNEL_ARG_INFO_NOT_AVAILABLE", // -19
  "Not defined -20",
  "Not defined -21",
  "Not defined -22",
  "Not defined -23",
  "Not defined -24",
  "Not defined -25",
  "Not defined -26",
  "Not defined -27",
  "Not defined -28",
  "Not defined -29",
  "CL_INVALID_VALUE", // -30
  "CL_INVALID_DEVICE_TYPE",
  "CL_INVALID_PLATFORM",
  "CL_INVALID_DEVICE",
  "CL_INVALID_CONTEXT",
  "CL_INVALID_QUEUE_PROPERTIES",
  "CL_INVALID_COMMAND_QUEUE",
  "CL_INVALID_HOST_PTR",
  "CL_INVALID_MEM_OBJECT",
  "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
  "CL_INVALID_IMAGE_SIZE", // -40
  "CL_INVALID_SAMPLER",
  "CL_INVALID_BINARY",
  "CL_INVALID_BUILD_OPTIONS",
  "CL_INVALID_PROGRAM",
  "CL_INVALID_PROGRAM_EXECUTABLE",
  "CL_INVALID_KERNEL_NAME",
  "CL_INVALID_KERNEL_DEFINITION",
  "CL_INVALID_KERNEL",
  "CL_INVALID_ARG_INDEX",
  "CL_INVALID_ARG_VALUE", // -50
  "CL_INVALID_ARG_SIZE",
  "CL_INVALID_KERNEL_ARGS",
  "CL_INVALID_WORK_DIMENSION",
  "CL_INVALID_WORK_GROUP_SIZE",
  "CL_INVALID_WORK_ITEM_SIZE",
  "CL_INVALID_GLOBAL_OFFSET",
  "CL_INVALID_EVENT_WAIT_LIST",
  "CL_INVALID_EVENT",
  "CL_INVALID_OPERATION",
  "CL_INVALID_GL_OBJECT", // -60
  "CL_INVALID_BUFFER_SIZE",
  "CL_INVALID_MIP_LEVEL",
  "CL_INVALID_GLOBAL_WORK_SIZE",
  "CL_INVALID_IMAGE_DESCRIPTOR",
  "CL_INVALID_COMPILER_OPTIONS",
  "CL_INVALID_LINKER_OPTIONS",
  "CL_INVALID_DEVICE_PARTITION_COUNT", // -68
};

const char *clStrError(int c) {
  int ec = -c;
  if (ec < 69) return errorStrings[ec];
  return "Unknown error";
}

cl_int checkError(cl_int ret, const char *s) {
  if (ret != CL_SUCCESS) {
    if (s == NULL) exitf(-1, "%s(%d)\n", clStrError(ret), ret);
    exitf(-1, "%s : %s(%d)\n", s, clStrError(ret), ret);
  }
  return CL_SUCCESS;
}

static cl_int checkError2(cl_int ret, cl_kernel kernel, const char *s) {
  if (ret != CL_SUCCESS) {
    char kn[1030];
    kn[0] = '\0';
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024, kn, NULL);

    if (s == NULL) exitf(-1, "%s : %s(%d)\n", kn, clStrError(ret), ret);
    exitf(-1, "%s : %s : %s(%d)\n", s, kn, clStrError(ret), ret);
  }
  return CL_SUCCESS;
}

cl_int ce(cl_int ret) {
  if (ret != CL_SUCCESS) {
    exitf(-1, "%s\n", clStrError(ret));
  }
  return CL_SUCCESS;
}

#define MAXPLATFORMS 10
#define MAXDEVICES 10

char *getDeviceName(cl_device_id device) {
  size_t len0, len1;
  clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &len0);
  clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &len1);

  size_t len = len0 + len1 + 10;
  char *strbuf = (char *)malloc(len);

  clGetDeviceInfo(device, CL_DEVICE_NAME, len, strbuf, &len0);
  strncat(strbuf, ", ", len);
  clGetDeviceInfo(device, CL_DEVICE_VERSION, len-strlen(strbuf), strbuf+strlen(strbuf), NULL);

  String_trim(strbuf);

  return strbuf;
}

static char *getDeviceName2(cl_device_id device) {
  char *strbuf = getDeviceName(device);

  for(char *p = strbuf;*p != '\0';p++) {
    if (*p == ':') *p = ';';
    if (isspace(*p)) *p = '_';
  }

  return strbuf;
}

cl_device_id simpleGetDevice(int did) {
  cl_int ret;
  cl_uint nPlatforms, nTotalDevices=0;
  cl_platform_id platformIDs[MAXPLATFORMS];
  cl_device_id devices[MAXDEVICES];

  clGetPlatformIDs(MAXPLATFORMS, platformIDs, &nPlatforms);
  if (nPlatforms == 0) exitf(-1, "No platform available\n");

  for(int p=0;p<(int)nPlatforms;p++) {
    cl_uint nDevices;
    ret = clGetDeviceIDs(platformIDs[p], CL_DEVICE_TYPE_ALL, MAXDEVICES-nTotalDevices, &devices[nTotalDevices], &nDevices);
    if (ret != CL_SUCCESS) continue;
    nTotalDevices += nDevices;
  }

  if (did < 0 || did >= (int)nTotalDevices) {
    if (did >= 0) fprintf(stderr, "Device %d does not exist\n", did);
    for(int i=0;i<(int)nTotalDevices;i++) {
      fprintf(stderr, "Device %d : %s\n", i, getDeviceName(devices[i]));
    }
    exit(-1);
  }

  return devices[did];
}

int simpleGetDevices(cl_device_id *devices, int maxDevices) {
  cl_int ret;
  cl_uint nPlatforms, nTotalDevices=0;
  cl_platform_id platformIDs[MAXPLATFORMS];

  clGetPlatformIDs(MAXPLATFORMS, platformIDs, &nPlatforms);
  if (nPlatforms == 0) return 0;

  for(int p=0;p<(int)nPlatforms;p++) {
    cl_uint nDevices;
    ret = clGetDeviceIDs(platformIDs[p], CL_DEVICE_TYPE_ALL, maxDevices-nTotalDevices, &devices[nTotalDevices], &nDevices);
    if (ret != CL_SUCCESS) continue;
    nTotalDevices += nDevices;
  }

  return nTotalDevices;
}

static void openclErrorCallback(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  fprintf(stderr, "\nError callback called, info = %s\n", errinfo);
#ifndef _MSC_VER
  void *array[20];
  size_t size = backtrace(array, 20);
  backtrace_symbols_fd(array, size, STDERR_FILENO);  
#endif
}

cl_context simpleCreateContext(cl_device_id device) {
  cl_int ret;
  cl_context hContext;

  hContext = clCreateContext(NULL, 1, &device, openclErrorCallback, NULL, &ret);
  if (ret != CL_SUCCESS) exitf(-1, "Could not create context : %s\n", clStrError(ret));

  return hContext;
}

int simpleBuildProgram(cl_program program, cl_device_id device, const char *optionString) {
  cl_int ret = clBuildProgram(program, 1, &device, optionString, NULL, NULL);
  if (ret != CL_SUCCESS) {
    fprintf(stderr, "Could not build program : %s\n", clStrError(ret));

    size_t len;
    ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *logbuf = (char *)calloc(len, sizeof(char));

    if ((ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, logbuf, NULL)) == CL_SUCCESS) {
      fprintf(stderr, "Build log follows\n\n");
      fprintf(stderr, "%s\n", logbuf);
    }

    return -1;
  }
  return 0;
}

void simpleSetKernelArg(cl_kernel kernel, const char *format, ...) {
  va_list valist;
  va_start(valist, format);
  int i;
  cl_int ret;
  for(i=0;;i++) {
    switch(format[i]) {
    case '\0' : {
      break;
    }

    case 'i' : {
      cl_int ai = va_arg(valist, int);
      ret = clSetKernelArg(kernel, i, sizeof(cl_int), (void *) &ai);
      if (ret != 0) exitf(-1, "simpleSetKernelArg %d : %s\n", i, clStrError(ret));
      break;
    }

    case 'l' : {
      cl_long al = va_arg(valist, int64_t);
      ret = clSetKernelArg(kernel, i, sizeof(cl_long), (void *) &al);
      if (ret != 0) exitf(-1, "simpleSetKernelArg %d : %s\n", i, clStrError(ret));
      break;
    }

    case 'f' : {
      cl_float af = va_arg(valist, double);
      ret = clSetKernelArg(kernel, i, sizeof(cl_float), (void *) &af);
      if (ret != 0) exitf(-1, "simpleSetKernelArg %d : %s\n", i, clStrError(ret));
      break;
    }

    case 'd' : {
      cl_double ad = va_arg(valist, double);
      ret = clSetKernelArg(kernel, i, sizeof(cl_double), (void *) &ad);
      if (ret != 0) exitf(-1, "simpleSetKernelArg %d : %s\n", i, clStrError(ret));
      break;
    }

    case 'M' : {
      cl_mem aM = va_arg(valist, cl_mem);
      ret = clSetKernelArg(kernel, i, sizeof(cl_mem), (void *) &aM);
      if (ret != 0) exitf(-1, "simpleSetKernelArg %d : %s\n", i, clStrError(ret));
      break;
    }

    default: {
      exitf(-1, "simpleSetKernelArg : format string error : %s\n", format);
    }
    }
    if (format[i] == '\0') break;
  }

  va_end(valist);
}

//

#define KERNELIDMAX 1000

typedef struct plan_t {
  int kernelID, valid, nitems;
  size_t lws[3];
  cl_kernel kernel;
  cl_event event;
  uint64_t time;
} plan_t;

#if 0
static void printPlan(plan_t p) {
  printf("kid = %d, valid = %d, nitems = %d, lws = %d, %d, %d, time = %lld\n",
	 p.kernelID, p.valid, p.nitems, (int)p.lws[0], (int)p.lws[1], (int)p.lws[2], (long long int)p.time);
}
#endif

static int profilingOngoing = 0, nPlanData = 0, planBufLen = 0;

static plan_t *planBuf = NULL;
static size_t profiling_lws[3];

static plan_t optimizedPlan[KERNELIDMAX];

static void addProfileEvent(int kernelID, cl_kernel kernel, size_t lws[3], cl_event ev) {
  if (planBuf == NULL) return;

  if (nPlanData >= planBufLen) {
    planBufLen *= 2;
    planBuf = (plan_t *)realloc(planBuf, planBufLen * sizeof(plan_t));
  }

  checkError(clRetainEvent(ev), "addProfileEvent");

  planBuf[nPlanData].kernelID = kernelID;
  planBuf[nPlanData].kernel = kernel;
  planBuf[nPlanData].valid = 1;
  planBuf[nPlanData].lws[0] = lws[0];
  planBuf[nPlanData].lws[1] = lws[1];
  planBuf[nPlanData].lws[2] = lws[2];
  planBuf[nPlanData].event = ev;
  nPlanData++;
}

static size_t *getLocalWorkSize(size_t lws[3], int kernelID) {
  if (profilingOngoing) {
    lws[0] = profiling_lws[0];
    lws[1] = profiling_lws[1];
    lws[2] = profiling_lws[2];
    return lws;
  } else if (kernelID < KERNELIDMAX && optimizedPlan[kernelID].valid) {
    lws[0] = optimizedPlan[kernelID].lws[0];
    lws[1] = optimizedPlan[kernelID].lws[1];
    lws[2] = optimizedPlan[kernelID].lws[2];
    return lws;
  }

  lws[0] = lws[1] = lws[2] = 0;
  return NULL;
}

void clearPlan() {
  for(int i=0;i<KERNELIDMAX;i++) {
    optimizedPlan[i].valid = 0;
  }
}

static void setOptimizedPlan(int kernelID, size_t ws1, size_t ws2, size_t ws3, uint64_t time) {
  if (kernelID < 0 || kernelID >= KERNELIDMAX) exitf(-1, "setOptimizedPlan : Invalid kernel id %d\n", kernelID);

  optimizedPlan[kernelID].kernelID = kernelID;
  optimizedPlan[kernelID].kernel = NULL;
  optimizedPlan[kernelID].valid = 1;
  optimizedPlan[kernelID].nitems = 1;
  optimizedPlan[kernelID].lws[0] = ws1;
  optimizedPlan[kernelID].lws[1] = ws2;
  optimizedPlan[kernelID].lws[2] = ws3;
  optimizedPlan[kernelID].time = time;
}

#define LINELEN 1000

int loadPlan(const char *fn, cl_device_id device) {
  char *dn = getDeviceName2(device);
#ifdef _MSC_VER
  char *deviceName = (char *)malloc(strlen(dn) + 10);
#else
  char *deviceName = (char *)alloca(strlen(dn) + 10);
#endif
  strcpy(deviceName, dn);
  strcat(deviceName, " : ");
  int nameLen = strlen(deviceName);

  char line[LINELEN+10];

  clearPlan();

  FILE *fp = fopen(fn, "r");
  if (fp == NULL) {
#ifdef _MSC_VER
    free(deviceName);
#endif
    free(dn);
    return -1;
  }

  int found = 0;

  for(;;) {
    if (fgets(line, LINELEN, fp) == NULL) break;
    if (strncmp(line, deviceName, nameLen) != 0) continue;

    int kid;
    long long int ws1, ws2, ws3;
    long long int time;
    if (sscanf(line, "%*s : %d : %lld : %lld : %lld : %lld\n", &kid, &ws1, &ws2, &ws3, &time) == 5) {
      found = 1;
      setOptimizedPlan(kid, (size_t)ws1, (size_t)ws2, (size_t)ws3, (uint64_t)time);
    }
  }

  fclose(fp);

#ifdef _MSC_VER
  free(deviceName);
#endif
  free(dn);

  return found ? 0 : -1;
}

void showPlan() {
  printf("%2s : %40s : %4s : %4s : %4s : %9s\n", "ID", "Kernel function name", "WS0", "WS1", "WS2", "Nano sec");
  printf("------------------------------------------------------------------------------\n");
  
  for(int i=0;i<KERNELIDMAX;i++) {
    if (!optimizedPlan[i].valid) continue;

    char kernelName[256] = "Unknown";

    if (optimizedPlan[i].kernel != NULL) {
	ce(clGetKernelInfo(optimizedPlan[i].kernel, CL_KERNEL_FUNCTION_NAME, 250, kernelName, NULL));
    }

    printf("%2d : %40s : %4lld : %4lld : %4lld : %9lld\n", optimizedPlan[i].kernelID, kernelName,
	   (long long int)optimizedPlan[i].lws[0], (long long int)optimizedPlan[i].lws[1], (long long int)optimizedPlan[i].lws[2], 
	    (long long int)optimizedPlan[i].time);
  }

  fflush(stdout);
}

void savePlan(const char *fn, cl_device_id device) {
  char *dn = getDeviceName2(device);
#ifdef _MSC_VER
  char *deviceName = (char *)malloc(strlen(dn) + 10);
#else
  char *deviceName = (char *)alloca(strlen(dn) + 10);
#endif
  strcpy(deviceName, dn);
  strcat(deviceName, " : ");
  int nameLen = strlen(deviceName);

#ifdef _MSC_VER
  char *line = (char *)malloc(LINELEN+10);
  FILE *tmpfp = fopen("baumtmp.txt", "w+");
#else
  char *line = (char *)alloca(LINELEN+10);
  FILE *tmpfp = tmpfile();
#endif

  if (tmpfp == NULL) exitf(-1, "Couldn't open temporary file\n");

  FILE *fp = fopen(fn, "r");
  if (fp != NULL) {
    for(;;) {
      if (fgets(line, LINELEN, fp) == NULL) break;
      if (strncmp(line, deviceName, nameLen) != 0) {
	fputs(line, tmpfp);
      }
    }

    fclose(fp);
  }

  for(int i=0;i<KERNELIDMAX;i++) {
    if (!optimizedPlan[i].valid) continue;

    fprintf(tmpfp, "%s%d : %lld : %lld : %lld : %lld\n", deviceName, optimizedPlan[i].kernelID,
	    (long long int)optimizedPlan[i].lws[0], (long long int)optimizedPlan[i].lws[1], (long long int)optimizedPlan[i].lws[2], 
	    (long long int)optimizedPlan[i].time);
  }

  fp = fopen(fn, "w");
  if (fp == NULL) exitf(-1, "Couldn't open file %s for writing\n", fn);

  fseek(tmpfp, 0, SEEK_SET);

  char buf[1024];

  for(;;) {
    size_t s = fread(buf, 1, 1024, tmpfp);
    if (s == 0) break;
    fwrite(buf, 1, s, fp);
  }

  fclose(fp);

#ifdef _MSC_VER
  fclose(tmpfp);
  remove("baumtmp.txt");
  free(line);
  free(deviceName);
#endif
  free(dn);
}

void startProfiling(size_t ws1, size_t ws2, size_t ws3) {
  planBufLen = 1024;
  planBuf = (plan_t *)malloc(planBufLen * sizeof(plan_t));
  profiling_lws[0] = ws1;
  profiling_lws[1] = ws2;
  profiling_lws[2] = ws3;
  profilingOngoing = 1;
}

static int planCmp(const void *p1, const void *p2) {
  plan_t *q1 = (plan_t *)p1, *q2 = (plan_t *)p2;

  if (!q1->valid &&  q2->valid) return  1;
  if ( q1->valid && !q2->valid) return -1;
  if (q1->kernelID > q2->kernelID) return  1;
  if (q1->kernelID < q2->kernelID) return -1;
  if (q1->lws[0] > q2->lws[0]) return  1;
  if (q1->lws[0] < q2->lws[0]) return -1;
  if (q1->lws[1] > q2->lws[1]) return  1;
  if (q1->lws[1] < q2->lws[1]) return -1;
  if (q1->lws[2] > q2->lws[2]) return  1;
  if (q1->lws[2] < q2->lws[2]) return -1;

  return 0;
}

void finishProfiling() {
  profilingOngoing = 0;

  qsort(planBuf, nPlanData, sizeof(plan_t), planCmp);

  plan_t cum;
  cum.valid = 0;
  cum.kernelID = -1;

  for(int i=0;i<nPlanData;i++) {
    if (!planBuf[i].valid) break;

    if (planBuf[i].event != NULL) {
      cl_ulong tstart, tend;
      clGetEventProfilingInfo(planBuf[i].event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tstart, NULL);
      clGetEventProfilingInfo(planBuf[i].event, CL_PROFILING_COMMAND_END  , sizeof(cl_ulong), &tend  , NULL);
      planBuf[i].time = tend - tstart;
      planBuf[i].nitems = 1;
      checkError(clReleaseEvent(planBuf[i].event), "finishProfiling");
      planBuf[i].event = NULL;
    }

    if (planCmp(&cum, &planBuf[i]) != 0) {
      if (cum.kernelID != -1) {
	cum.time /= cum.nitems;
	cum.nitems = 1;
	if (!optimizedPlan[cum.kernelID].valid ||
	    cum.time < optimizedPlan[cum.kernelID].time) {
	  optimizedPlan[cum.kernelID] = cum;
	}
      }
      cum = planBuf[i];
    } else {
      cum.nitems += planBuf[i].nitems;
      cum.time += planBuf[i].time;
    }
  }

  cum.time /= cum.nitems;
  cum.nitems = 1;

  if (cum.kernelID != -1 &&
      (!optimizedPlan[cum.kernelID].valid ||
       cum.time < optimizedPlan[cum.kernelID].time)) {
    optimizedPlan[cum.kernelID] = cum;
  }

  nPlanData = planBufLen = 0;
  free(planBuf);
  planBuf = NULL;
}

//

cl_event runKernel1D(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, int nev, ...) {
  if (kernelID < 0 || kernelID >= KERNELIDMAX) exitf(-1, "runKernel1D : Invalid kernel ID %d\n", kernelID);

#ifdef _MSC_VER
  cl_event *events = (cl_event *)malloc(nev * sizeof(cl_event));
#else
  cl_event *events = (cl_event *)alloca(nev * sizeof(cl_event));
#endif
  
  va_list valist;
  va_start(valist, nev);
  for (int i = 0;i < nev;i++) {
    events[i] = va_arg(valist, cl_event);
  }
  va_end(valist);

  cl_int ret;
  cl_event evret = NULL, *pev;
  pev = nev <= 0 ? NULL : events;

  size_t lws[3];
  ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &ws1, getLocalWorkSize(lws, kernelID), nev < 0 ? 0 : nev, pev, (profilingOngoing || nev >= 0) ? &evret : NULL);
  if (ret == CL_INVALID_WORK_GROUP_SIZE || ret == CL_OUT_OF_RESOURCES || ret == 1) {
    lws[0] = lws[1] = lws[2] = 0;
    checkError2(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &ws1, NULL, nev < 0 ? 0 : nev, pev, &evret), kernel, "runKernel1D");
  } else if (ret == CL_SUCCESS) {
    lws[1] = lws[2] = 0;
    addProfileEvent(kernelID, kernel, lws, evret);
  } else {
    checkError2(ret, kernel, "runKernel1D");
  }

  if (nev < 0 && evret != NULL) {
    checkError2(clReleaseEvent(evret), kernel, "runKernel1D");
    evret = NULL;
  }

#ifdef DEBUGKERNEL
  {
    char kn[1030];
    kn[0] = '\0';
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024, kn, NULL);
    printf("at %s\n", kn);
    fflush(stdout);
  }
  clFinish(queue);
#endif

#ifdef _MSC_VER
  free(events);
#endif

  return evret;
}

cl_event runKernel2D(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, size_t ws2, int nev, ...) {
  if (kernelID < 0 || kernelID >= KERNELIDMAX) exitf(-1, "runKernel2D : Invalid kernel ID %d\n", kernelID);

#ifdef _MSC_VER
  cl_event *events = (cl_event *)malloc(nev * sizeof(cl_event));
#else
  cl_event *events = (cl_event *)alloca(nev * sizeof(cl_event));
#endif

  va_list valist;
  va_start(valist, nev);
  for (int i = 0;i < nev;i++) {
    events[i] = va_arg(valist, cl_event);
  }
  va_end(valist);

  cl_int ret;
  cl_event evret = NULL, *pev;
  pev = nev <= 0 ? NULL : events;

  size_t gws[2] = {ws1, ws2}, lws[3];
  ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, getLocalWorkSize(lws, kernelID), nev < 0 ? 0 : nev, pev, (profilingOngoing || nev >= 0) ? &evret : NULL);
  if (ret == CL_INVALID_WORK_GROUP_SIZE || ret == CL_OUT_OF_RESOURCES || ret == 1) {
    lws[0] = lws[1] = lws[2] = 0;
    checkError2(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, NULL, nev < 0 ? 0 : nev, pev, &evret), kernel, "runKernel2D");
  } else if (ret == CL_SUCCESS) {
    lws[2] = 0;
    addProfileEvent(kernelID, kernel, lws, evret);
  } else {
    checkError2(ret, kernel, "runKernel2D");
  }

  if (nev < 0 && evret != NULL) {
    checkError2(clReleaseEvent(evret), kernel, "runKernel2D");
    evret = NULL;
  }

#ifdef DEBUGKERNEL
  {
    char kn[1030];
    kn[0] = '\0';
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024, kn, NULL);
    printf("at %s\n", kn);
    fflush(stdout);
  }
  clFinish(queue);
#endif

#ifdef _MSC_VER
  free(events);
#endif

  return evret;
}

cl_event runKernel1Dx(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, const cl_event *events) {
  if (kernelID < 0 || kernelID >= KERNELIDMAX) exitf(-1, "runKernel1D : Invalid kernel ID %d\n", kernelID);

  cl_int ret;
  cl_event evret = NULL;

  int nev = 0;
  if (events != NULL) for(;events[nev] != NULL;nev++) ;

  size_t lws[3];
  ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &ws1, getLocalWorkSize(lws, kernelID), nev, events, (profilingOngoing || events != NULL) ? &evret : NULL);
  if (ret == CL_INVALID_WORK_GROUP_SIZE || ret == CL_OUT_OF_RESOURCES || ret == 1) {
    lws[0] = lws[1] = lws[2] = 0;
    checkError2(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &ws1, NULL, nev, events, &evret), kernel, "runKernel1D");
  } else if (ret == CL_SUCCESS) {
    lws[1] = lws[2] = 0;
    addProfileEvent(kernelID, kernel, lws, evret);
  } else {
    checkError2(ret, kernel, "runKernel1D");
  }

  if (events == NULL && evret != NULL) {
    checkError2(clReleaseEvent(evret), kernel, "runKernel1D");
    evret = NULL;
  }

#ifdef DEBUGKERNEL
  {
    char kn[1030];
    kn[0] = '\0';
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024, kn, NULL);
    printf("at %s\n", kn);
    fflush(stdout);
  }
  clFinish(queue);
#endif

  return evret;
}

cl_event runKernel2Dx(cl_command_queue queue, cl_kernel kernel, int kernelID, size_t ws1, size_t ws2, const cl_event *events) {
  if (kernelID < 0 || kernelID >= KERNELIDMAX) exitf(-1, "runKernel2D : Invalid kernel ID %d\n", kernelID);

  cl_int ret;
  cl_event evret = NULL;

  int nev = 0;
  if (events != NULL) for(;events[nev] != NULL;nev++) ;

  size_t gws[2] = {ws1, ws2}, lws[3];
  ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, getLocalWorkSize(lws, kernelID), nev, events, (profilingOngoing || events != NULL) ? &evret : NULL);
  if (ret == CL_INVALID_WORK_GROUP_SIZE || ret == CL_OUT_OF_RESOURCES || ret == 1) {
    lws[0] = lws[1] = lws[2] = 0;
    checkError2(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, NULL, nev, events, &evret), kernel, "runKernel2D");
  } else if (ret == CL_SUCCESS) {
    lws[2] = 0;
    addProfileEvent(kernelID, kernel, lws, evret);
  } else {
    checkError2(ret, kernel, "runKernel2D");
  }

  if (events == NULL && evret != NULL) {
    checkError2(clReleaseEvent(evret), kernel, "runKernel2D");
    evret = NULL;
  }

#ifdef DEBUGKERNEL
  {
    char kn[1030];
    kn[0] = '\0';
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024, kn, NULL);
    printf("at %s\n", kn);
    fflush(stdout);
  }
  clFinish(queue);
#endif

  return evret;
}

void waitForEvent(cl_event ev) {
#ifdef DEBUGKERNEL
  printf("Entering waitForEvent\n");
  fflush(stdout);
#endif

  for(;;) {
    cl_int ret;
    ce(clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &ret, NULL));
    if (ret == CL_COMPLETE) break;

    sleepMillis(15);
  }

#ifdef DEBUGKERNEL
  printf("Leaving waitForEvent\n");
  fflush(stdout);
#endif
}

//

static int nextKernelID = 0;

int getNextKernelID() { return nextKernelID++; }

//

#define MAGIC_APM 0x2f47592b

typedef struct AllocatedPinnedMemory {
  uint32_t magic;
  cl_mem mem;
  void *ptr;
} AllocatedPinnedMemory;

ArrayMap *pinnedMemMap = NULL;

void *allocatePinnedMemory(size_t z, cl_context context, cl_command_queue queue) {
  if (pinnedMemMap == NULL) {
    pinnedMemMap = initArrayMap();
  }

  AllocatedPinnedMemory *m = (AllocatedPinnedMemory *)calloc(1, sizeof(AllocatedPinnedMemory));
  m->magic = MAGIC_APM;
  cl_int ret;
  m->mem = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, z, NULL, &ret); checkError(ret, "clCreateBuffer in allocatePinnedMemory");
  m->ptr = clEnqueueMapBuffer(queue, m->mem, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, z, 0, NULL, NULL, &ret); checkError(ret, "clEnqueueMapBuffer in allocatePinnedMemory");

  ArrayMap_put(pinnedMemMap, (uint64_t)(m->ptr), (void *)m);

  return m->ptr;
}

void freePinnedMemory(void *p, cl_context context, cl_command_queue queue) {
  AllocatedPinnedMemory *m = (AllocatedPinnedMemory *)ArrayMap_remove(pinnedMemMap, (uint64_t)p);
  assert(m != NULL && m->magic == MAGIC_APM);

  clEnqueueUnmapMemObject(queue, m->mem, m->ptr, 0, NULL, NULL);
  clFinish(queue);

  ce(clReleaseMemObject(m->mem));
  
  m->magic = 0;
  free(m);
}
