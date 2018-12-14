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
#include <sys/types.h>
#include <sys/timeb.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#include "helper.h"

void exitf(int code, const char *mes, ...) {
  va_list ap;
  va_start(ap, mes);
  vfprintf(stderr, mes, ap);
  va_end(ap);
  fflush(stderr);
  exit(code);
}

char *readFileAsStr(const char *fn, int maxSize) {
  FILE *fp = fopen(fn, "r");
  if (fp == NULL) exitf(-1, "Couldn't open file %s\n", fn);

  long size;

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  if (size > maxSize) exitf(-1, "readFileAsStr : file too large (%d bytes)\n", size);

  char *buf = (char *)malloc(size+10);

  if (buf == NULL) exitf(-1, "readFileAsStr : malloc failed\n");

  size = fread(buf, 1, size, fp);
  buf[size] = '\0';

  fclose(fp);

  return buf;
}

char *readFileAsStrN(const char **fn) {
  long size = 0, index = 0;
  char *buf = NULL;

  for(int i=0;fn[i] != NULL;i++) {
    FILE *fp = fopen(fn[i], "r");
    if (fp == NULL) exitf(-1, "Couldn't open file %s\n", fn[i]);

    fseek(fp, 0, SEEK_END);
    size += ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (size > 1000000) exitf(-1, "readFileAsStrN : total file size %d bytes is too large\n", (int)size);

    buf = (char *)realloc(buf, size + 10);

  if (buf == NULL) exitf(-1, "readFileAsStrN : realloc failed\n");

    index += fread(buf + index, 1, size, fp);
    buf[size] = '\0';

    fclose(fp);
  }

  return buf;
}

void String_trim(char *str) {
  char *dst = str, *src = str, *pterm = src;

  while(*src != '\0' && isspace(*src)) src++;

  for(;*src != '\0';src++) {
    *dst++ = *src;
    if (!isspace(*src)) pterm = dst;
  }

  *pterm = '\0';
}

int64_t currentTimeMillis() {
#ifdef _MSC_VER
  struct _timeb timebuffer;
  _ftime64_s( &timebuffer );
  return timebuffer.time * (int64_t)1000 + timebuffer.millitm;
#else
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * (int64_t)1000 + tp.tv_usec / 1000;
#endif
}

void sleepMillis(int ms) {
#ifdef _MSC_VER
  Sleep(ms);
#else
  usleep(ms * 1000);
#endif
}

#define MAGIC_ARRAYMAPNODE 0xf73130fa
#define MAGIC_ARRAYMAP 0x8693bd21
#define LOGNBUCKETS 10
#define NBUCKETS (1 << LOGNBUCKETS)

static int hash(uint64_t key) {
  return (key ^ (key >> LOGNBUCKETS) ^ (key >> (LOGNBUCKETS*2)) ^ (key >> (LOGNBUCKETS*3))) & (NBUCKETS-1);
}

typedef struct ArrayMapNode {
  uint32_t magic;
  uint64_t key;
  void *value;
} ArrayMapNode;

typedef struct ArrayMap {
  uint32_t magic;
  ArrayMapNode *array[NBUCKETS];
  int size[NBUCKETS], capacity[NBUCKETS], totalSize;
} ArrayMap;

ArrayMap *initArrayMap() {
  ArrayMap *thiz = (ArrayMap *)calloc(1, sizeof(ArrayMap));
  thiz->magic = MAGIC_ARRAYMAP;

  for(int i=0;i<NBUCKETS;i++) {
    thiz->capacity[i] = 8;
    thiz->array[i] = (ArrayMapNode *)malloc(thiz->capacity[i] * sizeof(ArrayMapNode));
    thiz->size[i] = 0;
  }

  thiz->totalSize = 0;
  return thiz;
}

void ArrayMap_dispose(ArrayMap *thiz) {
  assert(thiz != NULL && thiz->magic == MAGIC_ARRAYMAP);

  for(int j=0;j<NBUCKETS;j++) {
    for(int i=0;i<thiz->size[j];i++) {
      assert(thiz->array[j][i].magic == MAGIC_ARRAYMAPNODE);
      thiz->array[j][i].magic = 0;
    }
    free(thiz->array[j]);
  }

  thiz->magic = 0;
  free(thiz);
}

int ArrayMap_size(ArrayMap *thiz) {
  assert(thiz != NULL && thiz->magic == MAGIC_ARRAYMAP);
  return thiz->totalSize;
}

uint64_t *ArrayMap_keyArray(ArrayMap *thiz) {
  assert(thiz != NULL && thiz->magic == MAGIC_ARRAYMAP);
  uint64_t *a = (uint64_t *)malloc(sizeof(uint64_t) * thiz->totalSize);
  int p = 0;
  for(int j=0;j<NBUCKETS;j++) {
    for(int i=0;i<thiz->size[j];i++) {
      assert(thiz->array[j][i].magic == MAGIC_ARRAYMAPNODE);
      a[p++] = thiz->array[j][i].key;
    }
  }
  return a;
}

void **ArrayMap_valueArray(ArrayMap *thiz) {
  assert(thiz != NULL && thiz->magic == MAGIC_ARRAYMAP);
  void **a = (void **)malloc(sizeof(void *) * thiz->totalSize);
  int p = 0;
  for(int j=0;j<NBUCKETS;j++) {
    for(int i=0;i<thiz->size[j];i++) {
      assert(thiz->array[j][i].magic == MAGIC_ARRAYMAPNODE);
      a[p++] = thiz->array[j][i].value;
    }
  }
  return a;
}

void *ArrayMap_remove(ArrayMap *thiz, uint64_t key) {
  assert(thiz != NULL && thiz->magic == MAGIC_ARRAYMAP);

  int h = hash(key);
  for(int i=0;i<thiz->size[h];i++) {
    assert(thiz->array[h][i].magic == MAGIC_ARRAYMAPNODE);
    if (thiz->array[h][i].key == key) {
      void *old = thiz->array[h][i].value;
      thiz->array[h][i].key   = thiz->array[h][thiz->size[h]-1].key;
      thiz->array[h][i].value = thiz->array[h][thiz->size[h]-1].value;
      thiz->array[h][thiz->size[h]-1].magic = 0;
      thiz->size[h]--;
      thiz->totalSize--;
      return old;
    }
  }

  return NULL;
}

void *ArrayMap_put(ArrayMap *thiz, uint64_t key, void *value) {
  if (value == NULL) return ArrayMap_remove(thiz, key);

  assert(thiz != NULL && thiz->magic == MAGIC_ARRAYMAP);

  int h = hash(key);
  for(int i=0;i<thiz->size[h];i++) {
    assert(thiz->array[h][i].magic == MAGIC_ARRAYMAPNODE);
    if (thiz->array[h][i].key == key) {
      void *old = thiz->array[h][i].value;
      thiz->array[h][i].value = value;
      return old;
    }
  }

  if (thiz->size[h] >= thiz->capacity[h]) {
    thiz->capacity[h] *= 2;
    thiz->array[h] = (ArrayMapNode *)realloc(thiz->array[h], thiz->capacity[h] * sizeof(ArrayMapNode));
  }

  ArrayMapNode *n = &(thiz->array[h][thiz->size[h]++]);
  n->magic = MAGIC_ARRAYMAPNODE;
  n->key = key;
  n->value = value;

  thiz->totalSize++;

  return NULL;
}

void *ArrayMap_get(ArrayMap *thiz, uint64_t key) {
  assert(thiz != NULL && thiz->magic == MAGIC_ARRAYMAP);

  int h = hash(key);
  for(int i=0;i<thiz->size[h];i++) {
    assert(thiz->array[h][i].magic == MAGIC_ARRAYMAPNODE);
    if (thiz->array[h][i].key == key) {
      return thiz->array[h][i].value;
    }
  }

  return NULL;
}
