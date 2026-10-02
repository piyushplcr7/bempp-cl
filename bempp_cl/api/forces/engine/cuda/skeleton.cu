// Pairwise panel integration kernel skeleton for the bempp-cl forces engine.
//
// Hand-written device code.  The Python side (cuda_backend.py) only prepends a
// handful of #defines and splices the per-term side functions and the per-point
// term body at the @@GENERATED@@ marker.  Everything numerical lives here.
//
// Design
//   * One thread per panel pair, grid-stride loop over pairs.
//   * Separate launches per adjacency class (far / singular classes); far pairs are
//     enumerated as (test block) x (all trial elements) and adjacent pairs skipped.
//   * Element geometry is precomputed on the host in SoA layout; every per-pair
//     quantity is derived from it in registers; no dynamic memory.
//   * MODE_EVAL: the form is contracted with coefficient vectors; accumulators are
//     FIELD_CHUNK registers per thread (velocity fields processed in chunks),
//     reduced warp -> block -> per-block partials (deterministic host sum).
//   * MODE_ASSEMBLE: NSH_T x NSH_R local block per pair, scattered by atomicAdd.
//   * Velocity families evaluated through per-point cos/sin tables (cos family),
//     so a field costs a few multiplies instead of transcendental calls.

#include <cuda_runtime.h>

#ifndef REAL
#define REAL double
#endif
#ifndef FIELD_CHUNK
#define FIELD_CHUNK 27
#endif
#ifndef KMAX
#define KMAX 3
#endif
#ifndef BLOCK
#define BLOCK 128
#endif

#define INV4PI 0.07957747154594767

// ----------------------------------------------------------------------------
// small vector helpers
// ----------------------------------------------------------------------------
struct R3 {
    REAL x, y, z;
};
__device__ __forceinline__ R3 r3(REAL x, REAL y, REAL z) {
    R3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}
__device__ __forceinline__ R3 operator+(R3 a, R3 b) { return r3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ R3 operator-(R3 a, R3 b) { return r3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ R3 operator*(REAL s, R3 a) { return r3(s * a.x, s * a.y, s * a.z); }
__device__ __forceinline__ R3 operator*(R3 a, REAL s) { return r3(s * a.x, s * a.y, s * a.z); }
__device__ __forceinline__ R3 operator-(R3 a) { return r3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ R3& operator+=(R3& a, R3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__device__ __forceinline__ REAL dot(R3 a, R3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ R3 cross(R3 a, R3 b) {
    return r3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
// 3x3 matrix, row major: m[3*i+j] = ∂V_i/∂x_j
struct M33 {
    REAL m[9];
};
__device__ __forceinline__ R3 matvec(const M33& A, R3 v) {
    return r3(A.m[0] * v.x + A.m[1] * v.y + A.m[2] * v.z, A.m[3] * v.x + A.m[4] * v.y + A.m[5] * v.z,
              A.m[6] * v.x + A.m[7] * v.y + A.m[8] * v.z);
}

// ----------------------------------------------------------------------------
// contractions (selected by overload; see terms.py for the rules)
// ----------------------------------------------------------------------------
__device__ __forceinline__ REAL contract(REAL K, REAL a, REAL b) { return K * a * b; }
__device__ __forceinline__ REAL contract(REAL K, R3 a, R3 b) { return K * dot(a, b); }
__device__ __forceinline__ REAL contract(R3 K, REAL a, R3 b) { return a * dot(K, b); }
__device__ __forceinline__ REAL contract(R3 K, R3 a, REAL b) { return dot(a, K) * b; }
__device__ __forceinline__ REAL contract(R3 K, R3 a, R3 b) { return dot(a, cross(K, b)); }

// ----------------------------------------------------------------------------
// geometry and space data (SoA, precomputed on host)
// ----------------------------------------------------------------------------
struct Geo {
    const REAL* corners;   // [ne][3 vertex][3]
    const REAL* normals;   // [ne][3]
    const REAL* intelem;   // [ne]
    const REAL* p1grad;    // [ne][3 shape][3]  tangential gradients of hat functions
    const int* elements;   // [3][ne]  vertex indices (bempp layout)
    int ne;
};
struct SpaceD {
    const REAL* mult;      // [ne][nshape] local multipliers (signs)
    const REAL* nmult;     // [ne] normal multipliers
    const REAL* rwg;       // [ne][3] mult * edge_length / intelem (RWG only)
    const int* l2g;        // [ne][nshape]
    int nshape;
};

__device__ __forceinline__ R3 corner(const Geo& G, int e, int k) {
    const REAL* p = G.corners + 9 * e + 3 * k;
    return r3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
}
__device__ __forceinline__ R3 normal(const Geo& G, const SpaceD& S, int e) {
    const REAL* p = G.normals + 3 * e;
    REAL s = __ldg(S.nmult + e);
    return r3(s * __ldg(p), s * __ldg(p + 1), s * __ldg(p + 2));
}
__device__ __forceinline__ R3 map_point(const Geo& G, int e, REAL s, REAL t) {
    R3 v0 = corner(G, e, 0), v1 = corner(G, e, 1), v2 = corner(G, e, 2);
    return v0 + s * (v1 - v0) + t * (v2 - v0);
}
__device__ __forceinline__ bool adjacent(const Geo& G, int a, int b) {
    int a0 = __ldg(G.elements + a), a1 = __ldg(G.elements + G.ne + a), a2 = __ldg(G.elements + 2 * G.ne + a);
    int b0 = __ldg(G.elements + b), b1 = __ldg(G.elements + G.ne + b), b2 = __ldg(G.elements + 2 * G.ne + b);
    return a0 == b0 || a0 == b1 || a0 == b2 || a1 == b0 || a1 == b1 || a1 == b2 || a2 == b0 || a2 == b1 || a2 == b2;
}

// ----------------------------------------------------------------------------
// basis functions on an element (k = local shape function index)
// ----------------------------------------------------------------------------
__device__ __forceinline__ REAL basis_P0(const SpaceD& S, int e, int k) { return __ldg(S.mult + e); }
__device__ __forceinline__ REAL hat(int k, REAL s, REAL t) { return k == 0 ? (REAL)1 - s - t : (k == 1 ? s : t); }
__device__ __forceinline__ REAL basis_P1(const SpaceD& S, int e, int k, REAL s, REAL t) {
    return hat(k, s, t) * __ldg(S.mult + 3 * e + k);
}
__device__ __forceinline__ R3 basis_P1_grad(const Geo& G, const SpaceD& S, int e, int k) {
    const REAL* p = G.p1grad + 9 * e + 3 * k;
    REAL m = __ldg(S.mult + 3 * e + k);
    return r3(m * __ldg(p), m * __ldg(p + 1), m * __ldg(p + 2));
}
__device__ __forceinline__ R3 basis_P1_nxgrad(const Geo& G, const SpaceD& S, int e, int k) {
    return cross(normal(G, S, e), basis_P1_grad(G, S, e, k));
}
__device__ __forceinline__ R3 basis_P1_n(const Geo& G, const SpaceD& S, int e, int k, REAL s, REAL t) {
    return basis_P1(S, e, k, s, t) * normal(G, S, e);
}
__device__ __forceinline__ R3 basis_RWG(const Geo& G, const SpaceD& S, int e, int k, R3 x) {
    REAL c = __ldg(S.rwg + 3 * e + k);
    return c * (x - corner(G, e, 2 - k));
}
__device__ __forceinline__ REAL basis_RWG_div(const SpaceD& S, int e, int k) { return (REAL)2 * __ldg(S.rwg + 3 * e + k); }

// ----------------------------------------------------------------------------
// velocity families
//   kind 0: cos   V = e_alpha cos(a x) cos(b y) cos(c z), params (a, b, c, alpha)
//   kind 1: rot   V = e_d x (x - cg),                    params (d, cg)
//   kind 2: trans V = e_alpha,                            params (alpha)
// ----------------------------------------------------------------------------
struct VelTab {
    REAL c[KMAX][3];  // c[a][i] = cos(a * x_i)
    REAL s[KMAX][3];  // s[a][i] = sin(a * x_i)
};
__device__ __forceinline__ void vel_table(int kind, R3 x, VelTab& T) {
    if (kind == 0) {
        REAL xi[3] = {x.x, x.y, x.z};
#pragma unroll
        for (int a = 0; a < KMAX; ++a)
#pragma unroll
            for (int i = 0; i < 3; ++i) sincos((REAL)a * xi[i], &T.s[a][i], &T.c[a][i]);
    }
}
__device__ __forceinline__ void vel_eval(int kind, const REAL* prm, R3 x, const VelTab& T, R3& V, M33& DV) {
#pragma unroll
    for (int i = 0; i < 9; ++i) DV.m[i] = 0;
    if (kind == 0) {
        int a = (int)prm[0], b = (int)prm[1], c = (int)prm[2], al = (int)prm[3];
        REAL ca = T.c[a][0], cb = T.c[b][1], cc = T.c[c][2];
        REAL sa = T.s[a][0], sb = T.s[b][1], sc = T.s[c][2];
        REAL amp = ca * cb * cc;
        V = r3(0, 0, 0);
        REAL* Vp = &V.x;
        Vp[al] = amp;
        DV.m[3 * al + 0] = -(REAL)a * sa * cb * cc;
        DV.m[3 * al + 1] = -(REAL)b * ca * sb * cc;
        DV.m[3 * al + 2] = -(REAL)c * ca * cb * sc;
    } else if (kind == 1) {
        int d = (int)prm[0];
        R3 ax = r3(d == 0, d == 1, d == 2);
        R3 rel = x - r3(prm[1], prm[2], prm[3]);
        V = cross(ax, rel);
        DV.m[1] = -ax.z; DV.m[2] = ax.y;
        DV.m[3] = ax.z;  DV.m[5] = -ax.x;
        DV.m[6] = -ax.y; DV.m[7] = ax.x;
    } else {
        int al = (int)prm[0];
        V = r3(al == 0, al == 1, al == 2);
    }
}

// ----------------------------------------------------------------------------
// kernels (z = y - x)
// ----------------------------------------------------------------------------
struct Ctx {
    R3 x, y, z;
    REAL r, rinv;
    R3 Vx, Vy;
    M33 DVx, DVy;
};
__device__ __forceinline__ REAL kernel_SL(const Ctx& c) { return INV4PI * c.rinv; }
__device__ __forceinline__ R3 kernel_DL(const Ctx& c) { return (-INV4PI * c.rinv * c.rinv * c.rinv) * c.z; }
__device__ __forceinline__ R3 kernel_ADL(const Ctx& c) { return (INV4PI * c.rinv * c.rinv * c.rinv) * c.z; }
__device__ __forceinline__ REAL kernel_A1(const Ctx& c) {
    return INV4PI * dot(c.z, c.Vx - c.Vy) * c.rinv * c.rinv * c.rinv;
}
__device__ __forceinline__ R3 kernel_C3(const Ctx& c) {
    R3 dv = c.Vy - c.Vx;
    REAL r3i = c.rinv * c.rinv * c.rinv;
    REAL zdv = dot(c.z, dv);
    return INV4PI * ((-3 * zdv * r3i * c.rinv * c.rinv) * c.z + r3i * dv);
}
__device__ __forceinline__ R3 kernel_INTEGRABLE(const Ctx& c) {
    R3 dv = c.Vy - c.Vx;
    REAL r5i = c.rinv * c.rinv * c.rinv * c.rinv * c.rinv;
    return (3 * INV4PI * dot(c.z, dv) * r5i) * c.z;
}
__device__ __forceinline__ R3 kernel_COMB(const Ctx& c) {
    REAL r3i = c.rinv * c.rinv * c.rinv;
    return (INV4PI * r3i) * (-matvec(c.DVy, c.z) + c.Vy - c.Vx);
}

// ----------------------------------------------------------------------------
// reductions
// ----------------------------------------------------------------------------
__device__ __forceinline__ REAL warp_sum(REAL v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// @@GENERATED@@
// The Python side inserts here:
//   #define MODE_EVAL / MODE_ASSEMBLE, NSH_T, NSH_R, NFIELDS_USED, VEL_KIND, NPTR ...
//   struct Args { ... pointers unpacked from the pointer table ... };
//   __device__ void point_contrib(const Args& A, const Geo& G, int te, int tr,
//                                 REAL st, REAL tt, REAL sr, REAL trr, REAL w,
//                                 const REAL* velparams, int f0, int nf,
//                                 REAL* acc /* FIELD_CHUNK */  or  REAL* block /* NSH_T*NSH_R */);
// @@END_GENERATED@@

// ----------------------------------------------------------------------------
// pair loop
// ----------------------------------------------------------------------------
__device__ __forceinline__ void integrate_pair(const Args& A, const Geo& G, int te, int tr, const REAL* pt,
                                               const REAL* pr, const REAL* pw, int npts, REAL jac,
                                               const REAL* velparams, int f0, int nf, REAL* out) {
    for (int q = 0; q < npts; ++q) {
        REAL st = __ldg(pt + 2 * q), tt = __ldg(pt + 2 * q + 1);
        REAL sr = __ldg(pr + 2 * q), trr = __ldg(pr + 2 * q + 1);
        REAL w = jac * __ldg(pw + q);
        point_contrib(A, G, te, tr, st, tt, sr, trr, w, velparams, f0, nf, out);
    }
}

#ifdef MODE_EVAL
__device__ __forceinline__ void block_reduce_store(REAL* acc, int nf, REAL* partials) {
    // Slots 0..nf-1 hold the velocity-field accumulators, slot FIELD_CHUNK the
    // velocity-independent part; partials has FIELD_CHUNK + 1 columns per block.
    __shared__ REAL sh[FIELD_CHUNK + 1][BLOCK / 32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int f = 0; f < nf; ++f) {
        REAL v = warp_sum(acc[f]);
        if (lane == 0) sh[f][wid] = v;
    }
    {
        REAL v = warp_sum(acc[FIELD_CHUNK]);
        if (lane == 0) sh[FIELD_CHUNK][wid] = v;
    }
    __syncthreads();
    if (threadIdx.x < nf || threadIdx.x == FIELD_CHUNK) {
        REAL v = 0;
        for (int w = 0; w < BLOCK / 32; ++w) v += sh[threadIdx.x][w];
        partials[(size_t)blockIdx.x * (FIELD_CHUNK + 1) + threadIdx.x] = v;
    }
}
#endif

// Far pairs: p = i * nr + j over test block x all trial elements; adjacent pairs skipped.
extern "C" __global__ void far_kernel(const unsigned long long* ptrs, const int* iprm, int f0, int nf) {
    Args A = unpack(ptrs);
    const Geo& G = A.G;
    const int* test_el = A.test_el;
    const int* trial_el = A.trial_el;
    int nt = iprm[0], nr = iprm[1], nq = iprm[2];
    long long npairs = (long long)nt * nr;
#ifdef MODE_EVAL
    REAL acc[FIELD_CHUNK + 1];
    for (int f = 0; f <= FIELD_CHUNK; ++f) acc[f] = 0;
#endif
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < npairs;
         p += (long long)gridDim.x * blockDim.x) {
        int i = (int)(p / nr), j = (int)(p - (long long)i * nr);
        int te = __ldg(test_el + i), tr = __ldg(trial_el + j);
        if (adjacent(G, te, tr)) continue;
        REAL jac = __ldg(G.intelem + te) * __ldg(G.intelem + tr);
#ifdef MODE_EVAL
        integrate_pair(A, G, te, tr, A.reg_pt, A.reg_pr, A.reg_w, nq, jac, A.velparams, f0, nf, acc);
#else
        REAL block[NSH_T * NSH_R];
        for (int k = 0; k < NSH_T * NSH_R; ++k) block[k] = 0;
        integrate_pair(A, G, te, tr, A.reg_pt, A.reg_pr, A.reg_w, nq, jac, A.velparams, f0, nf, block);
        scatter_block(A, te, tr, block);
#endif
    }
#ifdef MODE_EVAL
    block_reduce_store(acc, nf, A.partials);
#endif
}

// Singular pairs: explicit pair list with per-pair point offsets into the Sauter-Schwab tables.
extern "C" __global__ void singular_kernel(const unsigned long long* ptrs, const int* iprm, int f0, int nf) {
    Args A = unpack(ptrs);
    const Geo& G = A.G;
    int npairs = iprm[0];
#ifdef MODE_EVAL
    REAL acc[FIELD_CHUNK + 1];
    for (int f = 0; f <= FIELD_CHUNK; ++f) acc[f] = 0;
#endif
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < npairs; p += gridDim.x * blockDim.x) {
        int te = __ldg(A.s_te + p), tr = __ldg(A.s_tr + p);
        int ot = __ldg(A.s_ot + p), orr = __ldg(A.s_or + p), ow = __ldg(A.s_ow + p), np_ = __ldg(A.s_np + p);
        REAL jac = __ldg(G.intelem + te) * __ldg(G.intelem + tr);
#ifdef MODE_EVAL
        integrate_pair(A, G, te, tr, A.s_pt + 2 * ot, A.s_pr + 2 * orr, A.s_w + ow, np_, jac, A.velparams, f0, nf,
                       acc);
#else
        REAL block[NSH_T * NSH_R];
        for (int k = 0; k < NSH_T * NSH_R; ++k) block[k] = 0;
        integrate_pair(A, G, te, tr, A.s_pt + 2 * ot, A.s_pr + 2 * orr, A.s_w + ow, np_, jac, A.velparams, f0, nf,
                       block);
        scatter_block(A, te, tr, block);
#endif
    }
#ifdef MODE_EVAL
    block_reduce_store(acc, nf, A.partials);
#endif
}
