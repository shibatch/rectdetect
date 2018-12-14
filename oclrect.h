#if defined(__cplusplus)
extern "C" {
#endif

typedef struct rect_t {
  union {
    struct {
      vec2 c2[4];
      vec3 c3[4];
      double value;
      uint32_t status;
    };
    int nItems;
  };
} rect_t;

struct oclrect_t *init_oclrect(struct oclimgutil_t *oclimgutil, struct oclpolyline_t *oclpolyline, cl_device_id device, cl_context context, cl_command_queue queue, int iw, int ih);
void dispose_oclrect(struct oclrect_t *thiz);

rect_t *oclrect_executeOnce(struct oclrect_t *thiz, uint8_t *imgData, int ws, const double tanAOV);

void oclrect_enqueueTask(struct oclrect_t *thiz, uint8_t *imgData, int ws);
rect_t *oclrect_pollTask(struct oclrect_t *thiz, const double tanAOV);

#if defined(__cplusplus)
}
#endif
