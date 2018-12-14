// Written by Naoki Shibata shibatch.sf.net@gmail.com 
// http://ito-lab.naist.jp/~n-sibata/

// This software is in public domain. You can use and modify this code
// for any purpose without any obligation.


#if defined(__cplusplus)
extern "C" {
#endif

  void exitf(int code, const char *mes, ...);
  char *readFileAsStr(const char *fn, int maxSize);
  char *readFileAsStrN(const char **fn);
  int64_t currentTimeMillis();
  void sleepMillis(int ms);
  void String_trim(char *str);
  
  typedef struct ArrayMap ArrayMap;

  ArrayMap *initArrayMap();
  void ArrayMap_dispose(ArrayMap *thiz);
  int ArrayMap_size(ArrayMap *thiz);
  void *ArrayMap_remove(ArrayMap *thiz, uint64_t key);
  void *ArrayMap_put(ArrayMap *thiz, uint64_t key, void *value);
  void *ArrayMap_get(ArrayMap *thiz, uint64_t key);

  uint64_t *ArrayMap_keyArray(ArrayMap *thiz);
  void **ArrayMap_valueArray(ArrayMap *thiz);
  uint64_t ArrayMap_getKey(ArrayMap *thiz, int idx);
  void *ArrayMap_getValue(ArrayMap *thiz, int idx);
#if defined(__cplusplus)
}
#endif
