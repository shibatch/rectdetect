// Copyright Naoki Shibata 2018. Distributed under the MIT License.

#ifdef ENABLE_ATOMICS64
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#ifndef NULL
#define NULL ((global void *)0)
#endif

#define MINEDGELEN 1 // 2
#define MINNINDEX 4

#ifndef NDEBUG
#define ASSERT(x, str) { if (!(x)) { printf((constant char *) "Assertion failure : %s\n", str); } }
#else
#define ASSERT(x, str) {}
#endif

typedef struct LS_t { // 56 bytes
  float2 startCoords, endCoords;
  int startIndex, endIndex;
  int leftPtr, rightPtr;

  int startCount, endCount;
  int maxDist;
  int polyid;
  int npix;
  int level;
} LS_t;

typedef struct LSX_t { // 56 bytes
  int64_t mx00, mx01, mx11, my0, my1;
  short2 dirSE, vDirSE;
  int distSquSE, padding;
} LSX_t;

float distanceSqu(float vx, float vy, float wx, float wy) {
  return (vx - wx) * (vx - wx) + (vy - wy) * (vy - wy);
}

float2 closestPoint(float vx, float vy, float wx, float wy, float px, float py) {
  float l2 = distanceSqu(vx, vy, wx, wy);
  if (l2 <= 1e-4f) return (float2)(vx, vy);
  float t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2;
  if (t < 0.0f) return (float2)(vx, vy);
  if (t > 1.0f) return (float2)(wx, wy);

  return (float2) (vx + t * (wx - vx), vy + t * (wy - vy));
}

//

__constant int rx[] = { 1,  1,  0, -1, -1, -1, 0, 1 };
__constant int ry[] = { 0, -1, -1, -1,  0,  1, 1, 1 };

__kernel void simpleJunction(global int *out, global int *in, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;
  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) out[p0] = 0;
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;

  int c = in[p0] != 0;

  if (c == 0) {
    out[p0] = 0;
    return;
  }

  int count = 1;

  for(int i=0;i<8;i++) {
    int p1 = p0 + rx[i] + ry[i] * iw;
    if (in[p1] != 0) count++;
  }

  out[p0] = count == 1 ? 0 : count;
}

__kernel void simpleConnect(global int *out, global int *in, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x <= 1 || y <= 1 || x >= (iw-2) || y >= (ih-2)) return;
  const int p0 = y * iw + x;

  out[p0] = in[p0] != 0 ? 1 : 0;

  if (in[p0] != 0) return;

  //

  if (in[p0-     2] != 0 && in[p0-   1] == 2 && in[p0+   1] == 2 && in[p0+     2] != 0) out[p0] = 1;
  if (in[p0-iw*2  ] != 0 && in[p0-iw  ] == 2 && in[p0+iw  ] == 2 && in[p0+iw*2  ] != 0) out[p0] = 1;

  if (in[p0-iw*2-2] != 0 && in[p0-iw-1] == 2 && in[p0+iw+1] == 2 && in[p0+iw*2+2] != 0) out[p0] = 1;
  if (in[p0-iw*2+2] != 0 && in[p0-iw+1] == 2 && in[p0+iw-1] == 2 && in[p0+iw*2-2] != 0) out[p0] = 1;

  if (in[p0     +2] != 0 && in[p0   +1] == 2 && in[p0+iw-1] == 2 && in[p0+iw  -2] != 0) out[p0] = 1;
  if (in[p0     -2] != 0 && in[p0   -1] == 2 && in[p0+iw+1] == 2 && in[p0+iw  +2] != 0) out[p0] = 1;
  if (in[p0-iw*2+1] != 0 && in[p0-iw+1] == 2 && in[p0+iw  ] == 2 && in[p0+iw*2  ] != 0) out[p0] = 1;
  if (in[p0-iw*2-1] != 0 && in[p0-iw-1] == 2 && in[p0+iw  ] == 2 && in[p0+iw*2  ] != 0) out[p0] = 1;
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

__kernel void removeBranch(global int *out, global int *in, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;
  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) out[p0] = 0;
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;

  int c = in[p0];

  if (c == 0) {
    out[p0] = 0;
    return;
  }

  int count = 0;

  for(int i=0;i<8;i++) {
    int p1 = p0 + rx[i] + ry[i] * iw;
    if (in[p1] != 0) count++;
  }

  out[p0] = count <= 2 ? 1 : 0;
}

__kernel void countEnds(global int *out, global int *junction, global int *label, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;
  const int p0 = y * iw + x;

  if (junction[p0] == 2) out[label[p0]]++;
}

__kernel void breakLoops(global int *edgeinout, global int *labelinout, global int *nEnds, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;
  const int p0 = y * iw + x;

  if (labelinout[p0] != p0) return;
  if (nEnds[p0] == 0) {
    edgeinout[p0] = 0;
    labelinout[p0] = -1;
  }
}

int2 getnp(global int *labelin, int p0, int iw) {
  int l = labelin[p0];

  int2 ret;

  int i;

  for(i=0;i<8;i++) {
    int p1 = p0 + rx[i] + ry[i] * iw;
    if (labelin[p1] == l) break;
  }

  ret.x = i < 8 ? (p0 + rx[i] + ry[i] * iw) : p0;

  for(i++;i<8;i++) {
    int p1 = p0 + rx[i] + ry[i] * iw;
    if (labelin[p1] == l) break;
  }

  ret.y = i < 8 ? (p0 + rx[i] + ry[i] * iw) : p0;

  return ret;
}

__kernel void findEnds0(global int *nextout, global int *prevout, global int *flagout, global int *labelin, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;

  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) {
    nextout[p0] = prevout[p0] = flagout[p0] = -1;
  }
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1) || labelin[p0] == -1) return;

  int2 np = getnp(labelin, p0, iw);

  nextout[p0] = np.x;
  prevout[p0] = np.y;

  int flag = 0;

  if (np.x != p0) {
    int2 nnp = getnp(labelin, np.x, iw);
    if (nnp.x == p0) flag |= 1;
  }

  if (np.y != p0) {
    int2 pnp = getnp(labelin, np.y, iw);
    if (pnp.y == p0) flag |= 2;
  }

  flagout[p0] = flag;
}

__kernel void findEnds1(global int *nextout, global int *prevout, global int *flaginout, global int *nextin, global int *previn, global int *labelin, int page, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;

  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) {
    nextout[p0] = prevout[p0] = -1;
  }
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1) || labelin[p0] == -1) return;

  bool revn = page == 0 ? ((flaginout[p0] & 1) != 0) : ((flaginout[p0] & 4) != 0);
  bool revp = page == 0 ? ((flaginout[p0] & 2) != 0) : ((flaginout[p0] & 8) != 0);
  int nn = nextin[p0], pp = previn[p0];

  for(int i=0;i<8;i++) {
    int nn2 = revn ? previn[nn] : nextin[nn];
    int pp2 = revp ? nextin[pp] : previn[pp];

    int nflag = flaginout[nn];
    int pflag = flaginout[pp];

    if (page != 0) { nflag >>= 2; pflag >>= 2; }

    revn = revn ? ((nflag & 2) == 0) : ((nflag & 1) != 0);
    revp = revp ? ((pflag & 1) == 0) : ((pflag & 2) != 0);

    nn = nn2;
    pp = pp2;
  }

  nextout[p0] = nn;
  prevout[p0] = pp;

  int f = flaginout[p0];

  if (page == 0) {
    f &= 3;
    f |= revn ? 4 : 0;
    f |= revp ? 8 : 0;
  } else {
    f &= (3 << 2);
    f |= revn ? 1 : 0;
    f |= revp ? 2 : 0;
  }

  flaginout[p0] = f;
}

__kernel void findEnds2(global int *numout, global int *linkout, global int *nextin, global int *previn, global int *labelin, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;
  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) { numout[p0] = 0; linkout[p0] = -1; }
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;

  if (labelin[p0] == -1) {
    linkout[p0] = -1;
    numout[p0] = 0;
    return;
  }

  int2 np = getnp(labelin, p0, iw);

  linkout[p0] = nextin[p0] < previn[p0] ? np.x : np.y;
  numout[p0] = linkout[p0] == p0 ? 0 : 1;
}

__kernel void number(global int *numout, global int *linkout, global int *numin, global int *linkin, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;
  if (x >= 0 && y >= 0 && x <= (iw-1) && y <= (ih-1)) { numout[p0] = 0; linkout[p0] = -1; }
  if (x <= 0 || y <= 0 || x >= (iw-1) || y >= (ih-1)) return;

  if (linkin[p0] == -1) {
    numout[p0] = numin[p0];
    linkout[p0] = linkin[p0];
    return;
  }

  int no = numin[p0], lo = linkin[p0];

  for(int i=0;i<32;i++) {
    if (!(0 < lo && lo < (iw*ih))) return;

    no += numin[lo];
    lo = linkin[lo];
  }

  numout[p0] = no;
  linkout[p0] = lo;
}

__kernel void labelpl_preprocess(global int *label, global int *pixinout, global int *flags, int maxpass, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  label[p0] = pixinout[p0] == 0 ? 0 : p0;
  pixinout[p0] = pixinout[p0] == 0 ? 0 : pixinout[p0]+1;

  if (y == 0 && x < maxpass+1) {
    flags[x] = x == 0 ? 1 : 0;
  }
}

__kernel void labelpl_main(global int *label, global int *pix, global int *flags, int pass, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x == 0 || y == 0 || x >= iw-1 || y >= ih-1) return;
  const int p0 = y * iw + x;

  if (flags[pass-1] == 0 || pix[p0] == 0) return;

  int g = label[p0], og = g, s;

  for(int i=0;i<8;i++) {
    int p1 = p0 + rx[i] + ry[i] * iw;
    s = label[p1];
    if (s < g && abs_diff(pix[p0], pix[p1]) <= 1) g = s;
  }

  for(int j=0;j<8;j++) {
    ASSERT(0 <= g && g < (iw*ih), "labelpl_main g");

    int s0 = label[g];
    if (s0 >= g) break;
    g = s0;
  }

  if (g != og) {
    ASSERT(0 <= og && og < (iw*ih), "labelpl_main og");

    atomic_min(&label[og], g);
    atomic_min(&label[p0], g);
    flags[pass] = 1;
  }
}

__kernel void calcSize(global int *out,
		       global int *label, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  int b = label[p0];
  if (b != 0) atomic_inc(&out[b]);
}

__kernel void filterSize(global int *out,
			 global int *labelin,
			 global int *sizein,
			 int sizethre, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  int b = labelin[p0];

  out[p0] = sizein[b] > sizethre ? b : 0;
}

__kernel void relabel_pass0(global int *table, global int *labelin, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x == 0 || y == 0 || x >= iw-1 || y >= ih-1) return;
  const int p0 = y * iw + x;

  int g = labelin[p0];

  if (g == 0 || p0 != g) {
    return;
  }

  ASSERT(0 < g && (g+1) < (iw*ih), "relabel_pass0 g");

  int rl = table[g+1];
  if (rl == 0) {
    int rl = atomic_inc(&table[0]) + 1;
    table[g+1] = rl;
  }
}

__kernel void relabel_pass1(global int *labelinout, global int *tablein, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;

  if (x < 0 || y < 0 || x > iw-1 || y > ih-1) return;

  if (x == 0 || y == 0 || x >= iw-1 || y >= ih-1) {
    labelinout[p0] = 0;
    return;
  }

  int g = labelinout[p0];

  if (g == 0) {
    return;
  }

  ASSERT(0 < g && (g+1) < (iw*ih), "relabel_pass1 g");

  labelinout[p0] = tablein[g+1];
}

global LS_t *leftLS(global LS_t *g, global LS_t *lsList) {
  if (g->leftPtr == 0) return NULL;
  global LS_t *p = &lsList[g->leftPtr];
  if (p->polyid != g->polyid) return NULL;
  return p;
}

global LS_t *rightLS(global LS_t *g, global LS_t *lsList) {
  if (g->leftPtr == 0) return NULL;
  global LS_t *p = &lsList[g->rightPtr];
  if (p->polyid != g->polyid) return NULL;
  return p;
}

//

// pass 0a : Construct the original lsList, find endIndex
__kernel void mkpl_pass0a(global void *lsList, int lsListSize, global int *numberin, global int *labelin, global int *flags, int maxIter, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);

  if (y == 0 && x < maxIter+1) {
    flags[x] = x == 0 ? 1 : 0;
  }

  if (x == 0 || y == 0 || x >= iw-1 || y >= ih-1) return;
  const int p0 = y * iw + x;

  int g = labelin[p0];
  int n = numberin[p0];

  if (g == 0) return;

  global LS_t *gp = lsList;

  if (g < 0 || lsListSize <= (g+1)*sizeof(LS_t)) {
#ifndef NDEBUG
    printf((constant char *)"Too many linesegemnts at mkpl_pass0a g=%d\n", g);
#endif
    return;
  }
  
  if (n == 1) {
    gp[g].startCoords = (float2) (x, y);
    gp[g].level = 0;
    atomic_inc(&gp[g].startCount);
  }

  atomic_inc(&gp[g].npix);
  atomic_max(&gp[g].endIndex, n);
  atomic_max((global int *)lsList, g);
}

// pass 0b : Construct the original lsList(continued)
__kernel void mkpl_pass0b(global void *lsList, int lsListSize, global int *numberin, global int *labelin, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x == 0 || y == 0 || x >= iw-1 || y >= ih-1) return;
  const int p0 = y * iw + x;

  int g = labelin[p0];
  int n = numberin[p0];

  if (g == 0) return;

  global LS_t *gp = lsList;

  if (g < 0 || lsListSize <= (g+1)*sizeof(LS_t)) {
#ifndef NDEBUG
    printf((constant char *)"Too many linesegemnts at mkpl_pass0b\n");
#endif
    return;
  }

  int endIndex = gp[g].endIndex;

  if (n == endIndex) {
    if (gp[g].startCount == 1 && gp[g].npix >= 2) { // ??
      if (atomic_inc(&gp[g].endCount) == 0) {
	gp[g].endCoords = (float2)(x, y);
	gp[g].polyid = labelin[p0];
      }
    } else {
      gp[g].polyid = 0;
    }
  }
}

// pass 1 : Find the max distance
__kernel void mkpl_pass1(global void *lsList, int lsListSize, global int *tmp, global int *numberin, global int *labelin, global int *randin, global int *flags, int nIter, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  if (flags[nIter-1] == 0) return;

  int g = labelin[p0];

  if (g == 0) return;

  global LS_t *gp = lsList;

  if (g < 0 || lsListSize <= (g+1)*sizeof(LS_t)) {
#ifndef NDEBUG
    printf((constant char *)"Too many linesegemnts at mkpl_pass1\n");
#endif
    return;
  }

  if (gp[g].polyid == 0) return;

  int x0 = gp[g].startCoords.x, y0 = gp[g].startCoords.y;
  int x1 = gp[g].endCoords.x, y1 = gp[g].endCoords.y;

  float2 cp = closestPoint(x0, y0, x1, y1, x, y);
  int dist = (int)(hypot(cp.x - x, cp.y - y) * 65536);
  dist ^= (randin[p0] & 0x1fff);

  tmp[p0] = dist;
  atomic_max(&gp[g].maxDist, dist);
}

// pass 2 : Subdivide
__kernel void mkpl_pass2(global void *newlsList, global void *lsList, int lsListSize, global int *tmp, global int *numberin, global int *labelin, global int *randin, global int *flags, int nIter, float minerror, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  if (flags[nIter-1] == 0) return;

  int g = labelin[p0];
  int n = numberin[p0];

  if (g == 0) return;

  global LS_t *gp = lsList;

  if (g < 0 || lsListSize <= (g+1)*sizeof(LS_t)) {
#ifndef NDEBUG
    printf((constant char *)"Too many linesegemnts at mkpl_pass2 (a)\n");
#endif
    return;
  }

  if (gp[g].polyid == 0) return;

  if (gp[g].endIndex - gp[g].startIndex < MINNINDEX-1) return;
  if (gp[g].startCount > 1 || gp[g].endCount > 1) return;

  int maxDist = gp[g].maxDist;

  if (tmp[p0] != maxDist) return;
  if (maxDist < ((int)(minerror * 65536))) return;

  if (maxDist < (minerror * 3 * 65536) && (float)maxDist*maxDist / distanceSqu(gp[g].startCoords.x, gp[g].startCoords.y, gp[g].endCoords.x, gp[g].endCoords.y) < 100000.0f) return;

  if (distanceSqu(x, y, gp[g].startCoords.x, gp[g].startCoords.y) < (MINEDGELEN * MINEDGELEN)) return;
  if (distanceSqu(x, y, gp[g].endCoords.x  , gp[g].endCoords.y  ) < (MINEDGELEN * MINEDGELEN)) return;

  int gr = gp[g].rightPtr;

  ASSERT(0 <= gr && gr < lsListSize-1, "mkpl_pass2 gr");

  //

  int gn = atomic_inc((global int *)newlsList) + 1;

  global LS_t *ngp = newlsList;

  if (gn < 0 || lsListSize <= (gn+1)*sizeof(LS_t)) {
#ifndef NDEBUG
    printf((constant char *)"Too many linesegemnts at mkpl_pass2 (b)\n");
#endif
    return;
  }

  ngp[gn].startIndex = n;
  ngp[gn].endIndex = gp[g].endIndex;
  ngp[gn].startCoords.x = x;
  ngp[gn].startCoords.y = y;
  ngp[gn].endCoords.x = gp[g].endCoords.x;
  ngp[gn].endCoords.y = gp[g].endCoords.y;
  ngp[gn].leftPtr = g;
  ngp[gn].rightPtr = gp[g].rightPtr;
  ngp[gn].maxDist = 0;
  ngp[gn].polyid = gp[g].polyid;
  ngp[gn].level = maxDist;
  
  ngp[g].endIndex = n;
  ngp[g].endCoords.x = x;
  ngp[g].endCoords.y = y;
  ngp[g].rightPtr = gn;
  ngp[g].maxDist = 0;

  if (gr != 0) ngp[gr].leftPtr = gn;
}

// pass 3 : Update labels
__kernel void mkpl_pass3(global void *lsList, int lsListSize, global int *numberin, global int *labelinout, global int *flags, int nIter, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  if (flags[nIter-1] == 0) return;

  int g = labelinout[p0];

  if (g == 0) return;

  global LS_t *gp = lsList;

  if (g < 0 || lsListSize <= (g+1)*sizeof(LS_t)) {
#ifndef NDEBUG
    printf((constant char *)"Too many linesegemnts at mkpl_pass3\n");
#endif
    return;
  }

  if (gp[g].polyid == 0) return;

  int n = numberin[p0];

  if (gp[g].endIndex < n) {
    labelinout[p0] = gp[g].rightPtr;
    flags[nIter] = 1;
  }
}

__kernel void mkpl_pass4(global void *lsList, int lsListSize, global int *numberout, global int *labelin, float minerror, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  int g = labelin[p0];

  if (g == 0) return;

  global LS_t *gp = lsList;

  if (g < 0 || lsListSize <= (g+1)*sizeof(LS_t)) {
#ifndef NDEBUG
    printf((constant char *)"Too many linesegemnts at mkpl_pass3\n");
#endif
    return;
  }

  if (gp[g].polyid == 0) return;

  int x0 = gp[g].startCoords.x, y0 = gp[g].startCoords.y;
  int x1 = gp[g].endCoords.x, y1 = gp[g].endCoords.y;

  float2 cp = closestPoint(x0, y0, x1, y1, x, y);
  float dist = hypot(cp.x - x, cp.y - y);

  if (dist > minerror * 4) {
    numberout[p0] = 0;
    return;
  }
}

__kernel void refine_pass0(global LSX_t *lsxListOut, global LS_t *lsListIn) {
  const int g = get_global_id(0)+1;

  //if (g == 1) printf("sizeof LSX_t = %d\n", sizeof(LSX_t));
  
  int lsListSize = *(global int *)lsListIn;
  if (g > lsListSize) return;

  if (lsListIn[g].polyid == 0) return;

  lsxListOut[g].dirSE       = convert_short2(lsListIn[g].endCoords - lsListIn[g].startCoords);
  lsxListOut[g].vDirSE.x    = -lsxListOut[g].dirSE.y;
  lsxListOut[g].vDirSE.y    =  lsxListOut[g].dirSE.x;
  lsxListOut[g].mx00        = 0;
  lsxListOut[g].mx01        = 0;
  lsxListOut[g].mx11        = 0;
  lsxListOut[g].my0         = 0;
  lsxListOut[g].my1         = 0;
  lsxListOut[g].distSquSE   = lsxListOut[g].dirSE.x * lsxListOut[g].dirSE.x + lsxListOut[g].dirSE.y * lsxListOut[g].dirSE.y;
  lsxListOut[g].padding     = 0;
}

int doti2(int2 x, int2 y) { return x.x * y.x + x.y * y.y; }

#ifndef ENABLE_ATOMICS64
void xatom_add(volatile __global long *p, long val) {
  uint32_t l = val & (long)0xffffffff;
  int32_t h = val >> 32;

  uint32_t o = atomic_add((volatile __global unsigned int *)p, l);
  if (o + l < o) h++;
  atomic_add(1 + (volatile __global int *)p, h);
}
#endif

__kernel void refine_pass1(global LSX_t *lsxListIO, global LS_t *lsListIn, global int *lsIdIn, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  int lsListSize = *(global int *)lsListIn;

  int g = lsIdIn[p0];
  if (g == 0) return;

  if (g < 0 || lsListSize < g) {
#ifndef NDEBUG
    printf((constant char *)"Too many linesegemnts at refine_pass1\n");
#endif
    return;
  }

  int2 v = (int2)(x, y) - convert_int2_rte(lsListIn[g].startCoords);
  int ay  = doti2(v, convert_int2(lsxListIO[g].vDirSE));
  int ax0 = doti2(v, convert_int2(lsxListIO[g].dirSE));
  int ax1 = lsxListIO[g].distSquSE;

#ifdef ENABLE_ATOMICS64
  atom_add(&(lsxListIO[g].mx00), convert_long_rte((float)ax0 * ax0));
  atom_add(&(lsxListIO[g].mx01), convert_long_rte((float)ax0 * ax1));
  atom_add(&(lsxListIO[g].mx11), convert_long_rte((float)ax1 * ax1));
  atom_add(&(lsxListIO[g].my0 ), convert_long_rte((float)ax0 * ay ));
  atom_add(&(lsxListIO[g].my1 ), convert_long_rte((float)ax1 * ay ));
#else
  xatom_add(&(lsxListIO[g].mx00), convert_long_rte((float)ax0 * ax0));
  xatom_add(&(lsxListIO[g].mx01), convert_long_rte((float)ax0 * ax1));
  xatom_add(&(lsxListIO[g].mx11), convert_long_rte((float)ax1 * ax1));
  xatom_add(&(lsxListIO[g].my0 ), convert_long_rte((float)ax0 * ay ));
  xatom_add(&(lsxListIO[g].my1 ), convert_long_rte((float)ax1 * ay ));
#endif
}

__kernel void refine_pass2(global LSX_t *lsxListIn, global LS_t *lsListIO) {
  const int g = get_global_id(0)+1;
  
  int lsListSize = *(global int *)lsListIO;
  if (g > lsListSize) return;

  if (lsListIO[g].polyid == 0) return;
  
  float rdet = (float)lsxListIn[g].mx00 * lsxListIn[g].mx11 - (float)lsxListIn[g].mx01 * lsxListIn[g].mx01;

  if (rdet == 0) return;

  rdet = 1.0 / rdet;
  float as0 = ((float)lsxListIn[g].mx11 * lsxListIn[g].my0 - (float)lsxListIn[g].mx01 * lsxListIn[g].my1) * rdet;
  float as1 = ((float)lsxListIn[g].mx00 * lsxListIn[g].my1 - (float)lsxListIn[g].mx01 * lsxListIn[g].my0) * rdet;

  lsListIO[g].startCoords += convert_float2(lsxListIn[g].vDirSE) * as1;
  lsListIO[g].endCoords   += convert_float2(lsxListIn[g].vDirSE) * (as0 + as1);  
}

__kernel void refine_pass3(global LS_t *lsListIO) {
  const int g = get_global_id(0)+1;
  
  int lsListSize = *(global int *)lsListIO;
  if (g > lsListSize) return;

  if (lsListIO[g].polyid == 0) return;

  const int h = lsListIO[g].rightPtr;
  if (h == 0) return;

  float v0 = lsListIO[g].startCoords.x, v1 = lsListIO[g].startCoords.y;
  float v2 = lsListIO[g].endCoords.x  , v3 = lsListIO[g].endCoords.y  ;

  float u0 = lsListIO[h].startCoords.x, u1 = lsListIO[h].startCoords.y;
  float u2 = lsListIO[h].endCoords.x  , u3 = lsListIO[h].endCoords.y  ;

  float d = (v2 - v0) * (u3 - u1) - (v3 - v1) * (u2 - u0);

  if (fabs(d) < 1e-6) {
    lsListIO[g].endCoords = lsListIO[h].startCoords =
      (lsListIO[g].endCoords + lsListIO[h].startCoords) * 0.5f;
    return;
  }
  
  float n = (v1 - u1) * (u2 - u0) - (v0 - u0) * (u3 - u1);
  float q = n / d;

  float2 w = (float2)(v0 + q * (v2 - v0), v1 + q * (v3 - v1));

  if (distance(w, lsListIO[g].endCoords) > 10 && distance(w, lsListIO[h].startCoords) > 10) {
    lsListIO[g].endCoords = lsListIO[h].startCoords =
      (lsListIO[g].endCoords + lsListIO[h].startCoords) * 0.5f;
    return;
  }

  lsListIO[g].endCoords = lsListIO[h].startCoords = w;
}

__kernel void labelxPreprocess_int_int(global int *label, global int *pix, global int *flags, int maxpass, int bgc, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  const int p0 = y * iw + x;

  if (y == 0 && x < maxpass+1) {
    flags[x] = x == 0 ? 1 : 0;
  }

  if (x >= iw || y >= ih) return;

  if (pix[p0] == bgc) { label[p0] = -1; return; }
  if (y > 0 && pix[p0] == pix[p0-iw]) { label[p0] = p0-iw; return; }
  if (x > 0 && pix[p0] == pix[p0- 1]) { label[p0] = p0- 1; return; }
  label[p0] = p0;
}

__kernel void label8xMain_int_int(global int *label, global int *pix, global int *flags, int pass, int iw, int ih) {
  const int x = get_global_id(0), y = get_global_id(1);
  if (x >= iw || y >= ih) return;
  const int p0 = y * iw + x;

  if (flags[pass-1] == 0) return;

  int g = label[p0], og = g;

  if (g == -1) return;

  for(int yy=-1;yy<=1;yy++) {
    for(int xx=-1;xx<=1;xx++) {
      if (0 <=  x + xx &&  x + xx < iw && 0 <=  y + yy &&  y + yy < ih) {
	const int p1 = (y + yy) * iw + x + xx, s = label[p1];
	if (s != -1 && s < g && pix[p0] == pix[p1]) g = s;
      }
    }
  }

  for(int j=0;j<6;j++) g = label[g];

  if (g != og) {
    atomic_min(&label[og], g);
    atomic_min(&label[p0], g);
    flags[pass] = 1;
  }
}

__kernel void clear(global int *out, int size) {
  const int x = get_global_id(0);
  if (x >= size) return;

  out[x] = 0;
}

__kernel void copy(global int *out, global int *in, int size) {
  const int x = get_global_id(0);
  if (x >= size) return;

  out[x] = in[x];
}

ulong xrandom(ulong s) {
  int n;
  ulong t = s;
  n = (s >> 24) & 63; t = ((t << n) | (t >> (64 - n))); t ^= 0xf3dd0fb7820fde37UL;
  n = (s >>  6) & 63; t = ((t << n) | (t >> (64 - n))); t ^= 0xe6c6ac2c59e52811UL;
  n = (s >> 18) & 63; t = ((t << n) | (t >> (64 - n))); t ^= 0x2fc7871fff7c5b45UL;
  n = (s >> 48) & 63; t = ((t << n) | (t >> (64 - n))); t ^= 0x47c7e1f70aa4f7c5UL;
  n = (s >>  0) & 63; t = ((t << n) | (t >> (64 - n))); t ^= 0x094f02b7fb9ba895UL;
  n = (s >> 12) & 63; t = ((t << n) | (t >> (64 - n))); t ^= 0x89afda817e744570UL;
  n = (s >> 36) & 63; t = ((t << n) | (t >> (64 - n))); t ^= 0xc7277d052c7bf14bUL;
  return t;
}

__kernel void rand(global int *out, ulong seed, int size) {
  const int x = get_global_id(0);
  if (x >= size) return;

  out[x] = (int)xrandom((x    ^ 0xb21c2cb635b48285UL) * 0x9b923b9cec745401UL +
			(seed ^ 0x7bb93d75a79d2f15UL) * 0x22cab58ada573a29UL);
}
