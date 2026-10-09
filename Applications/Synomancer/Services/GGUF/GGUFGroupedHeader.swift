// Adapted from llama.cpp 8a1a9b5 Metal grouped GEMM; MLX owns dispatch.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
enum GGUFGroupedHeader {
    static let source = GGUFMetalHeader.source + #"""
typedef struct {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm_id;
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4, typename Counts>
inline void gguf_grouped_mm(
        thread const ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        Counts htpe,
        device const char * hids,
        device       char * dst,
        device const char * amax,
        threadgroup  char * shmem ,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

#ifdef GGML_METAL_HAS_TENSOR
    threadgroup float * sc = (threadgroup float *)(shmem);
#endif

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z; // expert
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    auto tpe_u32 = htpe;
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    // skip the upper half of the token tile when the expert did not fill it
    constexpr short NR1H = NR1/2;

    const bool has_hi = nr1 > NR1H;

    // power-of-two rescaling
    const float s1_inv   = false ? ((device const float *) amax)[0] : 1.0f;
    const float s1_scale = false ? ((device const float *) amax)[1] : 1.0f;

#ifndef GGML_METAL_HAS_TENSOR
    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // simdgroups 2,3 own rows NR1H..NR1-1
    const bool sg_active = has_hi || sgitg < 2;
#else
    auto tA  = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));

    // sb is [NR1][NK] row-major
    auto tB0 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb,             dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + NR1H*NK,   dextents<int32_t, 2>(NK, NR1H));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT0 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto cT1 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
#ifndef GGML_METAL_HAS_TENSOR
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && false) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (false) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) (*((device T1 *) y + i) * (T1) s1_inv) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)((*((device T1_2x4 *) y)) * (T1) s1_inv);
        }
#else
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && false) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (false) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                //const short lx = (tiitg/NL1)%8;
                //const short ly = i;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) (*((device T1 *) y + i) * (T1) s1_inv) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            //const short lx = i;
            const short ly = (tiitg/NL1)%8;
            //const short lx = (tiitg/NL1)%8;
            //const short ly = i;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)((*((device T1_2x4 *) y)) * (T1) s1_inv);
        }
#endif

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifndef GGML_METAL_HAS_TENSOR
        if (sg_active) {
            // load matrices from threadgroup memory and conduct outer products
            threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
            threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*64;
                lsmb += 4*64;
            }
        }
#else
        auto sA  = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);

        mm.run(sB0, sA, cT0);

        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);

            mm.run(sB1, sA, cT1);
        }
#endif
    }

    // block is smaller than 64x32, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef GGML_METAL_HAS_TENSOR
    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc,             dextents<int32_t, 2>(NR0, NR1H));
    cT0.store(tC0);

    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + NR1H*NR0, dextents<int32_t, 2>(NR0, NR1H));
        cT1.store(tC1);
    }
#else
    if (sg_active) {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
    }
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];

        const short ide = id % args.ne20;
        const short idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i) * s1_scale;
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i) * s1_scale;
        }
    }
}

"""#
}
