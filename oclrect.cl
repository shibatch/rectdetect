// Copyright Naoki Shibata 2018. Distributed under the MIT License.

typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

__constant int rx[] = { 1,  1,  0, -1, -1, -1, 0, 1 };
__constant int ry[] = { 0, -1, -1, -1,  0,  1, 1, 1 };

// --------------------------------

int squ(int x) { return x * x; }

int mirror(int x, int y, int iw, int ih) {
  int2 xy = (int2)(x, y);
  xy = clamp(xy, -xy, (int2)(iw, ih) * 2 - 2 - xy);
  return xy.x + xy.y * iw;
}

uint32_t packlab(float3 clab) {
  uint32_t ret = 0;
  ret = clamp(convert_uint_rtn(clab.z * 1024), 0U, 1023U);
  ret = (ret << 10) | clamp(convert_uint_rtn(clab.y * 1024), 0U, 1023U);
  ret = (ret << 12) | clamp(convert_uint_rtn(clab.x * 4096), 0U, 4095U);
  return ret;
}

float3 unpacklab(uint32_t plab) {
  float3 f = convert_float3((int3) (plab, plab >> 12, plab >> 22) & (int3) (4095, 1023, 1023));
  return f * (float3) (1.0f/4096, 1.0f/1024, 1.0f/1024) + (float3)(0.5f/4096, 0.5f/1024, 0.5f/1024);
}

uint32_t packlabbl(int3 clab) {
  uint32_t ret = 0;
  ret = (uint32_t)clamp(clab.z, 0, 1023);
  ret = (ret << 10) | (uint32_t)clamp(clab.y, 0, 1023);
  ret = (ret << 12) | (uint32_t)clamp(clab.x, 0, 4095);
  return ret;
}

int3 unpacklabbl(uint32_t plab) {
  return (int3) (plab, plab >> 12, plab >> 22) & (int3) (4095, 1023, 1023);
}

float bicubicSub(float p0, float p1, float p2, float p3, float x) {
  float u, v, w;
  v = p1 - p2;
  w = p3 - p0;
  u = v * 3.0f + w;
  u = u * x   + (-4.0f * v + (p0 - p1 - w));
  u = u * x   + (p2 - p0);
  u = u * x * 0.5f + p1;
  return u;
}

float bicubic(global float *p, float x, float y, int iw, int ih) {
  const int ix = (int)x, iy = (int)y;

  return bicubicSub(bicubicSub(p[mirror(ix-1, iy-1,iw,ih)], p[mirror(ix  , iy-1,iw,ih)], p[mirror(ix+1, iy-1,iw,ih)], p[mirror(ix+2, iy-1,iw,ih)], x-ix),
		    bicubicSub(p[mirror(ix-1, iy  ,iw,ih)], p[mirror(ix  , iy  ,iw,ih)], p[mirror(ix+1, iy  ,iw,ih)], p[mirror(ix+2, iy  ,iw,ih)], x-ix),
		    bicubicSub(p[mirror(ix-1, iy+1,iw,ih)], p[mirror(ix  , iy+1,iw,ih)], p[mirror(ix+1, iy+1,iw,ih)], p[mirror(ix+2, iy+1,iw,ih)], x-ix),
		    bicubicSub(p[mirror(ix-1, iy+2,iw,ih)], p[mirror(ix  , iy+2,iw,ih)], p[mirror(ix+1, iy+2,iw,ih)], p[mirror(ix+2, iy+2,iw,ih)], x-ix), y-iy);
}

// --------------------------------

#define BLBLURSIZE 4

__kernel void simpleJunction(global int *out, global int *in, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;
  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) out[p0] = 0;
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;

  int c = in[p0] > 0;

  if (c == 0) {
    out[p0] = 0;
    return;
  }

  int count = 1;

  for(int i=0;i<8;i++) {
    int p1 = p0 + rx[i] + ry[i] * iw;
    if (in[p1] > 0) count++;
  }

  out[p0] = count == 1 ? 0 : count;
}

__kernel void simpleConnect(global int *out, global int *in, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;
  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) out[p0] = 0;
  if (x <= 1 || y <= 1 || x >= (iw-2) || y >= (ih-2)) return;

  out[p0] = in[p0] != 0 ? 1 : 0;

  if (in[p0] != 0) return;

  //

  if (in[p0-   1] == 2 && in[p0+   1] != 0) out[p0] = 1;
  if (in[p0-   1] != 0 && in[p0+   1] == 2) out[p0] = 1;
  if (in[p0-iw  ] == 2 && in[p0+iw  ] != 0) out[p0] = 1;
  if (in[p0-iw  ] != 0 && in[p0+iw  ] == 2) out[p0] = 1;

  if (in[p0-iw-1] == 2 && in[p0+iw+1] == 2) out[p0] = 1;
  if (in[p0-iw+1] == 2 && in[p0+iw-1] == 2) out[p0] = 1;

  if (in[p0   +1] == 2 && in[p0+iw-1] == 2) out[p0] = 1;
  if (in[p0   -1] == 2 && in[p0+iw+1] == 2) out[p0] = 1;
  if (in[p0-iw+1] == 2 && in[p0+iw  ] == 2) out[p0] = 1;
  if (in[p0-iw-1] == 2 && in[p0+iw  ] == 2) out[p0] = 1;
}

__kernel void stringify(global int *out, global int *in, int mod2, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;
  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) out[p0] = in[p0];
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;

  if (((x + y) & 1) != mod2) return;

  if (in[p0-iw] != 0 && in[p0-1] != 0) out[p0] = 0;
  if (in[p0-iw] != 0 && in[p0+1] != 0) out[p0] = 0;
  if (in[p0+iw] != 0 && in[p0-1] != 0) out[p0] = 0;
  if (in[p0+iw] != 0 && in[p0+1] != 0) out[p0] = 0;
}

__kernel void calcStrength(global int *out, global float *edge, global int *label, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;
  const int p0 = y * iw + x;

  if (label[p0] <= 0) return;

  atomic_add(&out[label[p0]], (int)(edge[p0] * edge[p0] * 10000.0f));
}

__kernel void filterStrength(global int *labelinout, global int *str, int thre, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;
  const int p0 = y * iw + x;

  if (labelinout[p0] <= 0 || str[labelinout[p0]] < thre) labelinout[p0] = -1;
}

__kernel void blblur0(global uint32_t *out, global int8_t *edge, global uint32_t *in, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;

  int wsum = 0, oe = edge[y*iw+x] != 0;
  int3 csum = 0;

  for(int xx=0;xx>=-BLBLURSIZE;xx--) {
    if (x + xx < 0) break;
    if (x + xx > 0 && edge[y*iw+x+xx] != 0 && edge[y*iw+x+xx-1] == 0) break;
    if (x + xx > 0 && y < ih-1 && edge[y*iw+x+xx] == 0 && edge[y*iw+x+xx-1] != 0 && edge[(y+1)*iw+x+xx] != 0) break;
    wsum++;
    csum += unpacklabbl(in[y*iw+x+xx]);
  }

  for(int xx=0;xx<=BLBLURSIZE;xx++) {
    if (x + xx > iw-1) break;
    if (x + xx < iw-1 && edge[y*iw+x+xx] == 0 && edge[y*iw+x+xx+1] != 0) break;
    if (oe && edge[y*iw+x+xx] == 0) break;
    wsum++;
    csum += unpacklabbl(in[y*iw+x+xx]);
  }

  out[y*iw+x] = wsum == 0 ? in[y*iw+x] : packlabbl(csum / wsum);
}

__kernel void blblur1(global uint32_t *out, global int8_t *edge, global uint32_t *in, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;

  int wsum = 0, oe = edge[y*iw+x] != 0;
  int3 csum = 0;

  for(int yy=0;yy>=-BLBLURSIZE;yy--) {
    if (y + yy < 0) break;
    if (y + yy > 0 && edge[(y+yy)*iw+x] != 0 && edge[(y+yy-1)*iw+x] == 0) break;
    if (y + yy > 0 && x < iw-1 && edge[(y+yy)*iw+x] == 0 && edge[(y+yy-1)*iw+x] != 0 && edge[(y+yy)*iw+x+1] != 0) break;
    wsum++;
    csum += unpacklabbl(in[(y+yy)*iw+x]);
  }

  for(int yy=0;yy<=BLBLURSIZE;yy++) {
    if (y + yy > ih-1) break;
    if (y + yy < ih-1 && edge[(y+yy)*iw+x] == 0 && edge[(y+yy+1)*iw+x] != 0) break;
    if (oe && edge[(y+yy)*iw+x] == 0) break;
    wsum++;
    csum += unpacklabbl(in[(y+yy)*iw+x]);
  }

  out[y*iw+x] = wsum == 0 ? in[y*iw+x] : packlabbl(csum / wsum);
}

__kernel void quantize(global uint32_t *out, global uint32_t *in, int n0, int n1, int n2, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;

  float3 v = unpacklab(in[y * iw + x]);

  out[y * iw + x] = packlab((float3) (round(v.x * n0) / (float)n0,
				      round(v.y * n1) / (float)n1,
				      round(v.z * n2) / (float)n2));
}

__kernel void despeckle(global uint32_t *out, global uint32_t *in, global float *edge, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;

  const int p0 = y * iw + x;

  out[p0] = in[p0];
  if (edge[p0] < 1e-6f) return;

  float dist = 1e+10f;
  float3 labp0 = unpacklab(in[p0]);
  
  for(int yy=-1;yy<=1;yy++) {
    for(int xx=-1;xx<=1;xx++) {
      if (0 <=  x + xx &&  x + xx < iw && 0 <=  y + yy &&  y + yy < ih) {
	const int p1 = (y + yy) * iw + x + xx;
	if (edge[p1] >= 1e-6f) continue;

	float d = distance(unpacklab(in[p1]), labp0);
	if (d < dist) {
	  out[p0] = in[p1];
	  dist = d;
	}
      }
    }
  }
}

__kernel void mkMergeMask0(global int *out, global int *junctionIn, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  if (junctionIn[p0] != 0) {
    for(int yy=y-6;yy<=y+6;yy++) {
      for(int xx=x-6;xx<=x+6;xx++) {
	if (xx < 0 || iw <= xx || yy < 0 || ih <= yy) continue;
	int dsqu = squ(yy-y) + squ(xx-x);
	int p1 = yy * iw + xx;
	if (16 <= dsqu && dsqu < 36) out[p1] = 1;
      }
    }
  }
}

__kernel void mkMergeMask1(global int *inout, global int *junctionIn, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  if (junctionIn[p0] == 2) {
    for(int yy=y-8;yy<=y+8;yy++) {
      for(int xx=x-8;xx<=x+8;xx++) {
	if (xx < 0 || iw <= xx || yy < 0 || ih <= yy) continue;
	int dsqu = squ(yy-y) + squ(xx-x);
	int p1 = yy * iw + xx;
	if (dsqu < 64) inout[p1] = 0;
      }
    }
  } else if (junctionIn[p0] != 0) {
    for(int yy=y-4;yy<=y+4;yy++) {
      for(int xx=x-4;xx<=x+4;xx++) {
	if (xx < 0 || iw <= xx || yy < 0 || ih <= yy) continue;
	int dsqu = squ(yy-y) + squ(xx-x);
	int p1 = yy * iw + xx;
	if (dsqu < 16) inout[p1] = 0;
      }
    }
  }
}

__kernel void labelxPreprocess(global int *label, global int *pixin, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  //if (pixin[p0] <= 0) { label[p0] = -1; return; }
  if (y > 0 && pixin[p0] == pixin[p0-iw]) { label[p0] = p0-iw; return; }
  if (x > 0 && pixin[p0] == pixin[p0- 1]) { label[p0] = p0- 1; return; }
  label[p0] = p0;
}

__kernel void labelMergeMain(global int *label, global int *pixin, global int *maskin, global int *edgein, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;
  const int p0 = y * iw + x;

  int g = label[p0], og = g, p1, s;
  if (g == -1) return;

  if (y > 0) {
    p1 = p0 - iw; s = label[p1];
    if (s < g && (pixin[p0] == pixin[p1] || maskin[p0] != 0) && edgein[p0] <= 0) g = s;
  }

  if (x > 0) {
    p1 = p0 - 1; s = label[p1];
    if (s < g && (pixin[p0] == pixin[p1] || maskin[p0] != 0) && edgein[p0] <= 0) g = s;
  }

  if (x < iw-1) {
    p1 = p0 + 1; s = label[p1];
    if (s < g && (pixin[p0] == pixin[p1] || maskin[p0] != 0) && edgein[p1] <= 0) g = s;
  }

  if (y < ih-1) {
    p1 = p0 + iw; s = label[p1];
    if (s < g && (pixin[p0] == pixin[p1] || maskin[p0] != 0) && edgein[p1] <= 0) g = s;
  }

  for(int j=0;j<8;j++) g = label[g];

  if (g != og) {
    atomic_min(&label[og], g);
    atomic_min(&label[p0], g);
  }
}

__kernel void calcSize(global int *out,
		       global int *label, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  int l = label[p0];
  if (l != -1) {
    atomic_inc(&out[l]);
  }
}

__kernel void despeckle2(global uint32_t *labelinout, global int *sizein, int thre, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;

  const int p0 = y * iw + x;

  if (sizein[labelinout[p0]] > thre) return;

  int maxSize = 0, maxLabel = labelinout[p0];
  
  for(int yy=-1;yy<=1;yy++) {
    for(int xx=-1;xx<=1;xx++) {
      if (0 <= x + xx &&  x + xx < iw && 0 <= y + yy && y + yy < ih) {
	const int p1 = (y + yy) * iw + x + xx;
	if (sizein[labelinout[p1]] > maxSize) {
	  maxSize = sizein[labelinout[p1]];
	  maxLabel = labelinout[p1];
	}
      }
    }
  }

  labelinout[p0] = maxLabel;
}

__kernel void markBoundary(global int *out, global int *in, global int *edge, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;
  if (x >= iw || y >= ih) return;
  if (x <= 1 || y <= 1 || x >= iw-2 || y >= ih-2) { out[p0] = -1; return; }

  int nearEdge = 0;
  int c0 = in[p0];

  for(int yy=-2;yy<=2;yy++) {
    for(int xx=-2;xx<=2;xx++) {
      int p1 = p0 + yy * iw + xx;
      if (in[p1] != c0) nearEdge = 1;
    }
  }

  out[p0] = nearEdge ? in[p0] : -1;
}

__kernel void colorReassign_pass0(global int *countOut, global int *lout, global int *aout, global int *bout, global uint32_t *labIn, global int *labelin, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;

  int p0 = y * iw + x;

  int lb = labelin[p0];
  if (lb == -1) return;

  float3 v = unpacklab(labIn[p0]);

  atomic_add(&lout[lb], (int)(v.x * 128));
  atomic_add(&aout[lb], (int)(v.y * 128));
  atomic_add(&bout[lb], (int)(v.z * 128));
  atomic_inc(&countOut[lb]);
}

__kernel void colorReassign_pass1(global uint32_t *labOut, global int *countIn, global int *lin, global int *ain, global int *bin, global int *labelin, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;

  int p0 = y * iw + x;

  int lb = labelin[p0];
  if (lb == -1) return;

  float3 v;
  v.x = (float)lin[lb] / (countIn[lb] * 128);
  v.y = (float)ain[lb] / (countIn[lb] * 128);
  v.z = (float)bin[lb] / (countIn[lb] * 128);
  
  labOut[p0] = packlab(v);
}

// nentry = iw * ih * 2 / 5
__kernel void reduceLS(global int *out, global int *boundaryin, global int *lsidin, int iw, int ih, int nentry) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;
  const int p0 = y * iw + x, lsid = lsidin[p0];

  if (lsid <= 0) return;

  int bid = 0, hash = 0;

  for(int yy=-3;yy<=3;yy++) {
    if (y+yy < 0 || ih <= y+yy) continue;
    for(int xx=-3;xx<=3;xx++) {
      if (x+xx < 0 || iw <= x+xx) continue;

      const int p1 = (y + yy) * iw + x + xx;
      if (boundaryin[p1] <= 0) continue;

      if (bid != boundaryin[p1]) {
	bid = boundaryin[p1];
	hash = (((unsigned int)lsid * (unsigned int)bid) & 0x7fffffff) % (unsigned int)nentry;
      }

      int oldid = out[hash*5+0];

      if (oldid == 0) {
	oldid = atomic_cmpxchg(&out[hash*5+0], 0, lsid);
	if (oldid != 0 && oldid != lsid) continue;
      }

      if (oldid != lsid) continue;

      atomic_max(&out[hash*5+1], iw-x);
      atomic_max(&out[hash*5+2],    x);
      atomic_max(&out[hash*5+3], ih-y);
      atomic_max(&out[hash*5+4],    y);
    }
  }
}
