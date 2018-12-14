#if defined(__cplusplus)
extern "C" {
#endif

typedef struct {
  double a[2];
} vec2;

static inline vec2 cvec2(double d0, double d1) {
  vec2 v; v.a[0] = d0; v.a[1] = d1; return v;
}

static inline vec2 plus2(vec2 a, vec2 b) {
  vec2 v;
  for(int i=0;i<2;i++) v.a[i] = a.a[i] + b.a[i];
  return v;
}

static inline vec2 minus2(vec2 a, vec2 b) {
  vec2 v;
  for(int i=0;i<2;i++) v.a[i] = a.a[i] - b.a[i];
  return v;
}

static inline double vdot2(vec2 a, vec2 b) {
  double sum = 0;
  for(int i=0;i<2;i++) sum += a.a[i] * b.a[i];
  return sum;
}

static inline vec2 dot2(vec2 a, double d) {
  vec2 v;
  for(int i=0;i<2;i++) v.a[i] = a.a[i] * d;
  return v;
}

static inline vec2 normalize2(vec2 v) {
  double sum = 0;
  for(int i=0;i<2;i++) sum += v.a[i] * v.a[i];
  
  return dot2(v, 1.0 / (sqrt(sum) + 1e-20));
}

static inline double lengthSqu2(vec2 v) {
  double sum = 0;
  for(int i=0;i<2;i++) sum += v.a[i] * v.a[i];
  return sum;
}

static inline double distanceSqu2(vec2 v, vec2 w) {
  return lengthSqu2(minus2(v, w));
}

static inline double distance2(vec2 v, vec2 w) {
  return sqrt(distanceSqu2(v, w));
}

static inline vec2 midpoint2(vec2 p0, vec2 p1) {
  return dot2(plus2(p0, p1), 0.5);
}

//

typedef struct {
  double a[3];
} vec3;

static inline vec3 cvec3(double d0, double d1, double d2) {
  vec3 v; v.a[0] = d0; v.a[1] = d1; v.a[2] = d2; return v;
}

static inline vec3 plus3(vec3 a, vec3 b) {
  vec3 v;
  for(int i=0;i<3;i++) v.a[i] = a.a[i] + b.a[i];
  return v;
}

static inline vec3 minus3(vec3 a, vec3 b) {
  vec3 v;
  for(int i=0;i<3;i++) v.a[i] = a.a[i] - b.a[i];
  return v;
}

static inline double vdot3(vec3 a, vec3 b) {
  double sum = 0;
  for(int i=0;i<3;i++) sum += a.a[i] * b.a[i];
  return sum;
}

static inline vec3 dot3(vec3 a, double d) {
  vec3 v;
  for(int i=0;i<3;i++) v.a[i] = a.a[i] * d;
  return v;
}

static inline vec3 normalize3(vec3 v) {
  double sum = 0;
  for(int i=0;i<3;i++) sum += v.a[i] * v.a[i];
  
  return dot3(v, 1.0 / (sqrt(sum) + 1e-20));
}

static inline double lengthSqu3(vec3 v) {
  double sum = 0;
  for(int i=0;i<3;i++) sum += v.a[i] * v.a[i];
  return sum;
}

static inline double distanceSqu3(vec3 v, vec3 w) {
  return lengthSqu3(minus3(v, w));
}

static inline double distance3(vec3 v, vec3 w) {
  return sqrt(distanceSqu3(v, w));
}

static inline vec3 midpoint3(vec3 p0, vec3 p1) {
  return dot3(plus3(p0, p1), 0.5);
}

//

typedef struct {
  double a[4];
} vec4;

static inline vec4 cvec4(double d0, double d1, double d2, double d3) {
  vec4 v; v.a[0] = d0; v.a[1] = d1; v.a[2] = d2; v.a[3] = d3; return v;
}

static inline vec4 plus4(vec4 a, vec4 b) {
  vec4 v;
  for(int i=0;i<4;i++) v.a[i] = a.a[i] + b.a[i];
  return v;
}

static inline vec4 minus4(vec4 a, vec4 b) {
  vec4 v;
  for(int i=0;i<4;i++) v.a[i] = a.a[i] - b.a[i];
  return v;
}

static inline double vdot4(vec4 a, vec4 b) {
  double sum = 0;
  for(int i=0;i<4;i++) sum += a.a[i] * b.a[i];
  return sum;
}

static inline vec4 dot4(vec4 a, double d) {
  vec4 v;
  for(int i=0;i<4;i++) v.a[i] = a.a[i] * d;
  return v;
}

static inline vec4 normalize4(vec4 v) {
  double sum = 0;
  for(int i=0;i<4;i++) sum += v.a[i] * v.a[i];
  
  return dot4(v, 1.0 / (sqrt(sum) + 1e-20));
}

static inline double lengthSqu4(vec4 v) {
  double sum = 0;
  for(int i=0;i<4;i++) sum += v.a[i] * v.a[i];
  return sum;
}

static inline double distanceSqu4(vec4 v, vec4 w) {
  return lengthSqu4(minus4(v, w));
}

static inline double distance4(vec4 v, vec4 w) {
  return sqrt(distanceSqu4(v, w));
}

static inline vec4 midpoint4(vec4 p0, vec4 p1) {
  return dot4(plus4(p0, p1), 0.5);
}

#if defined(__cplusplus)
}
#endif
