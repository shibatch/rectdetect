#if defined(__cplusplus)
extern "C" {
#endif

#define MAGIC_EGBUF 0xfc6f7b49

typedef struct {
  uint32_t magic;
  void *ptr;
  int size, nalloc, sizeMember;
} EGBuf;

static inline EGBuf *EGBuf_init(int sizeMember) {
  EGBuf *b = (EGBuf *)calloc(1, sizeof(EGBuf));

  b->magic = MAGIC_EGBUF;
  b->sizeMember = sizeMember;
  b->nalloc = 16;
  b->ptr = malloc(b->nalloc * b->sizeMember);

  return b;
}

static inline void EGBuf_dispose(EGBuf *b) {
  assert(b->magic == MAGIC_EGBUF);

  free(b->ptr);
  b->ptr = NULL;
  b->magic = 0;
  free(b);
}

static inline void EGBuf_add(EGBuf *b, void *nm) {
  assert(b->magic == MAGIC_EGBUF);

  if (b->nalloc == b->size) {
    b->nalloc *= 2;
    b->ptr = realloc(b->ptr, b->nalloc * b->sizeMember);
  }

  memcpy((uint8_t *)b->ptr + b->sizeMember * b->size, nm, b->sizeMember);
  b->size++;
}

static inline void EGBuf_remove(EGBuf *b, int index) {
  assert(b->magic == MAGIC_EGBUF && 0 <= index && index < b->size);

  memmove((uint8_t *)b->ptr + b->sizeMember * index,
	  (uint8_t *)b->ptr + b->sizeMember * (index + 1),
	  b->sizeMember * (b->size - index - 1));
  b->size--;
}

#if defined(__cplusplus)
}
#endif
