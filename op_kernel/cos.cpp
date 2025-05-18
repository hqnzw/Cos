/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file cos.cpp
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <class T, class ComputeStrategy>
class KernelCos
{
public:
    __aicore__ inline KernelCos() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t bigCoreDataNum,
                                uint32_t smallCoreDataNum,
                                uint32_t tileDataNum,
                                uint32_t bigCoreNum,
                                AscendC::TPipe* pipe);
    __aicore__ inline void Process();

private:
    template <class U>
    static __aicore__ inline const U& min(const U& a, const U& b) { return (b < a) ? b : a; }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t processDataNum);
    __aicore__ inline void Compute(uint32_t processDataNum);
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t processDataNum);

    __aicore__ inline AscendC::LocalTensor<float> PreDeQueCastX(uint32_t processDataNum);
    __aicore__ inline AscendC::LocalTensor<float> PreAllocateY();
    __aicore__ inline void PostReleaseCastEnQue(AscendC::LocalTensor<float>& xLocal,
                                                AscendC::LocalTensor<float>& yLocal,
                                                uint32_t processDataNum);

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xBuf, yBuf;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;

    uint32_t coreDataNum;
    uint32_t tileDataNum;

    ComputeStrategy strategy;
};

template <class T, class ComputeStrategy>
__aicore__ inline void KernelCos<T, ComputeStrategy>::Init(GM_ADDR x, GM_ADDR y,
                                                           uint32_t bigCoreDataNum,
                                                           uint32_t smallCoreDataNum,
                                                           uint32_t tileDataNum,
                                                           uint32_t bigCoreNum,
                                                           AscendC::TPipe* pipe)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
    if (AscendC::GetBlockIdx() <= bigCoreNum) {
        this->coreDataNum = bigCoreDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - bigCoreNum);
    }
    this->tileDataNum = tileDataNum;

    xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
    pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    if constexpr (!std::is_same_v<T, float>) {
        pipe->InitBuffer(xBuf, this->tileDataNum * sizeof(float));
        pipe->InitBuffer(yBuf, this->tileDataNum * sizeof(float));
    }
    strategy.InitBufImpl(pipe, this->tileDataNum);
}

template <class T, class ComputeStrategy>
__aicore__ inline void KernelCos<T, ComputeStrategy>::Process()
{
    uint64_t coreDataNum = this->coreDataNum;
    uint64_t tileDataNum = this->tileDataNum;
    for (uint64_t i = 0; i < coreDataNum; i += tileDataNum) {
        uint32_t processDataNum = min(tileDataNum, coreDataNum - i);
        CopyIn(i, processDataNum);
        Compute(processDataNum);
        CopyOut(i, processDataNum);
    }
}

template <class T, class ComputeStrategy>
__aicore__ inline void KernelCos<T, ComputeStrategy>::CopyIn(uint64_t offset, uint32_t processDataNum)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[offset], processDataNum);
    inQueueX.EnQue(xLocal);
}

template <class T, class ComputeStrategy>
__aicore__ inline void KernelCos<T, ComputeStrategy>::Compute(uint32_t processDataNum)
{
    AscendC::LocalTensor<float> xLocal = PreDeQueCastX(processDataNum);
    AscendC::LocalTensor<float> yLocal = PreAllocateY();

    strategy.ComputeImpl(xLocal, yLocal, processDataNum);

    PostReleaseCastEnQue(xLocal, yLocal, processDataNum);
}

template <class T, class ComputeStrategy>
__aicore__ inline void KernelCos<T, ComputeStrategy>::CopyOut(uint64_t offset, uint32_t processDataNum)
{
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    AscendC::DataCopy(yGm[offset], yLocal, processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <class T, class ComputeStrategy>
__aicore__ inline AscendC::LocalTensor<float> KernelCos<T, ComputeStrategy>::PreDeQueCastX(uint32_t processDataNum)
{
    if constexpr (std::is_same_v<T, float>) {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        return xLocal;
    } else {
        AscendC::LocalTensor<float> xLocal = xBuf.Get<float>();
        AscendC::LocalTensor<T> xOrigin = inQueueX.DeQue<T>();
        AscendC::Cast(xLocal, xOrigin, AscendC::RoundMode::CAST_NONE, processDataNum);
        inQueueX.FreeTensor(xOrigin);
        return xLocal;
    }
}

template <class T, class ComputeStrategy>
__aicore__ inline AscendC::LocalTensor<float> KernelCos<T, ComputeStrategy>::PreAllocateY()
{
    if constexpr (std::is_same_v<T, float>) {
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        return yLocal;
    } else {
        AscendC::LocalTensor<float> yLocal = yBuf.Get<float>();
        return yLocal;
    }
}

template <class T, class ComputeStrategy>
__aicore__ inline void KernelCos<T, ComputeStrategy>::PostReleaseCastEnQue(AscendC::LocalTensor<float>& xLocal,
                                                                           AscendC::LocalTensor<float>& yLocal,
                                                                           uint32_t processDataNum)
{
    if constexpr (std::is_same_v<T, float>) {
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    } else {
        AscendC::LocalTensor<T> yTarget = outQueueY.AllocTensor<T>();
    #if __CCE_AICORE__ == 200
        AscendC::Cast(yTarget, yLocal, AscendC::RoundMode::CAST_NONE, processDataNum);
    #else
        AscendC::Cast(yTarget, yLocal, AscendC::RoundMode::CAST_RINT, processDataNum);
    #endif
        outQueueY.EnQue(yTarget);
    }
}

class RefStrategy
{
public:
    __aicore__ inline RefStrategy() {}
    __aicore__ inline void InitBufImpl(AscendC::TPipe* pipe, uint32_t tileDataNum);
    __aicore__ inline void ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                       AscendC::LocalTensor<float>& yLocal,
                                       uint32_t processDataNum);

private:
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1, tmpBuf2;
};

__aicore__ inline void RefStrategy::InitBufImpl(AscendC::TPipe* pipe, uint32_t tileDataNum)
{
    pipe->InitBuffer(tmpBuf1, tileDataNum * sizeof(float));
    pipe->InitBuffer(tmpBuf2, tileDataNum * sizeof(float));
}

constexpr float TWO_PI = 2 * 3.14159265358979;
constexpr float REF_COEF_2 = -1.0 / (2 * 1);
constexpr float REF_COEF_4 = -1.0 / (4 * 3);
constexpr float REF_COEF_6 = -1.0 / (6 * 5);
constexpr float REF_COEF_8 = -1.0 / (8 * 7);
constexpr float REF_COEF_10 = -1.0 / (10 * 9);
constexpr float REF_COEF_12 = -1.0 / (12 * 11);
constexpr float REF_COEF_14 = -1.0 / (14 * 13);

__aicore__ inline void RefStrategy::ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                                AscendC::LocalTensor<float>& yLocal,
                                                uint32_t processDataNum)
{
    AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();

    const AscendC::LocalTensor<float>& input_x = xLocal;
    const AscendC::LocalTensor<float>& vmu_ = tmpTensor1;
    const AscendC::LocalTensor<int32_t>& round_fp = yLocal.ReinterpretCast<int32_t>();
    const AscendC::LocalTensor<float>& round_fp32 = tmpTensor1;
    const AscendC::LocalTensor<float>& t = tmpTensor2;
    const AscendC::LocalTensor<float>& input_x_round = yLocal;
    const AscendC::LocalTensor<float>& res = tmpTensor1;
    const AscendC::LocalTensor<float>& input_x_power = xLocal;
    const AscendC::LocalTensor<float>& iter_value = tmpTensor2;
    const AscendC::LocalTensor<float>& res_1 = yLocal;
    const AscendC::LocalTensor<float>& t_1 = tmpTensor1;
    const AscendC::LocalTensor<float>& iter_value_1 = tmpTensor2;
    const AscendC::LocalTensor<float>& res_2 = tmpTensor1;
    const AscendC::LocalTensor<float>& t_2 = yLocal;
    const AscendC::LocalTensor<float>& iter_value_2 = tmpTensor2;
    const AscendC::LocalTensor<float>& res_3 = yLocal;
    const AscendC::LocalTensor<float>& t_3 = tmpTensor1;
    const AscendC::LocalTensor<float>& iter_value_3 = tmpTensor2;
    const AscendC::LocalTensor<float>& res_4 = tmpTensor1;
    const AscendC::LocalTensor<float>& t_4 = yLocal;
    const AscendC::LocalTensor<float>& iter_value_4 = tmpTensor2;
    const AscendC::LocalTensor<float>& res_5 = yLocal;
    const AscendC::LocalTensor<float>& t_5 = tmpTensor1;
    const AscendC::LocalTensor<float>& iter_value_5 = tmpTensor2;
    const AscendC::LocalTensor<float>& res_6 = tmpTensor1;
    const AscendC::LocalTensor<float>& t_6 = yLocal;
    const AscendC::LocalTensor<float>& iter_value_6 = xLocal;
    const AscendC::LocalTensor<float>& res_7 = yLocal;

    /// vmu_ = tbe.vmuls(input_x, 1.0 / Constant.TWO_PI)
    AscendC::Muls(vmu_, input_x, 1.0f / TWO_PI, processDataNum); // 倒数精度ok
    /// round_fp = tbe.round(vmu_)
    AscendC::Cast(round_fp, vmu_, AscendC::RoundMode::CAST_RINT, processDataNum);
    /// round_fp32 = tbe.cast_to(round_fp, dtype)
    AscendC::Cast(round_fp32, round_fp, AscendC::RoundMode::CAST_NONE, processDataNum);
    /// input_x_round = tbe.vsub(input_x, tbe.vmuls(round_fp32, Constant.TWO_PI))
    AscendC::Muls(t, round_fp32, TWO_PI, processDataNum);
    AscendC::Sub(input_x_round, input_x, t, processDataNum);

    /// res = tbe.broadcast(const_res, input_x.shape)
    AscendC::Duplicate(res, 1.0f, processDataNum);

    /// input_x_power = tbe.vmul(input_x_round, input_x_round)
    AscendC::Mul(input_x_power, input_x_round, input_x_round, processDataNum);
    /// iter_value = tbe.vmuls(input_x_power, -1.0 / 2.0)
    AscendC::Muls(iter_value, input_x_power, REF_COEF_2, processDataNum);
    /// res = tbe.vadd(res, iter_value)
    AscendC::Add(res_1, res, iter_value, processDataNum);

    /// iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
    AscendC::Mul(t_1, input_x_power, iter_value, processDataNum);
    AscendC::Muls(iter_value_1, t_1, REF_COEF_4, processDataNum);
    /// res = tbe.vadd(res, iter_value)
    AscendC::Add(res_2, res_1, iter_value_1, processDataNum);

    /// iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
    AscendC::Mul(t_2, input_x_power, iter_value_1, processDataNum);
    AscendC::Muls(iter_value_2, t_2, REF_COEF_6, processDataNum);
    /// res = tbe.vadd(res, iter_value)
    AscendC::Add(res_3, res_2, iter_value_2, processDataNum);

    /// iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
    AscendC::Mul(t_3, input_x_power, iter_value_2, processDataNum);
    AscendC::Muls(iter_value_3, t_3, REF_COEF_8, processDataNum);
    /// res = tbe.vadd(res, iter_value)
    AscendC::Add(res_4, res_3, iter_value_3, processDataNum);

    /// iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
    AscendC::Mul(t_4, input_x_power, iter_value_3, processDataNum);
    AscendC::Muls(iter_value_4, t_4, REF_COEF_10, processDataNum);
    /// res = tbe.vadd(res, iter_value)
    AscendC::Add(res_5, res_4, iter_value_4, processDataNum);

    /// iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
    AscendC::Mul(t_5, input_x_power, iter_value_4, processDataNum);
    AscendC::Muls(iter_value_5, t_5, REF_COEF_12, processDataNum);
    /// res = tbe.vadd(res, iter_value)
    AscendC::Add(res_6, res_5, iter_value_5, processDataNum);

    /// iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
    AscendC::Mul(t_6, input_x_power, iter_value_5, processDataNum);
    AscendC::Muls(iter_value_6, t_6, REF_COEF_14, processDataNum);
    /// res = tbe.vadd(res, iter_value)
    AscendC::Add(res_7, res_6, iter_value_6, processDataNum);
}

class HighPerfStrategy
{
public:
    __aicore__ inline HighPerfStrategy() {}
    __aicore__ inline void InitBufImpl(AscendC::TPipe* pipe, uint32_t tileDataNum);
    __aicore__ inline void ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                       AscendC::LocalTensor<float>& yLocal,
                                       uint32_t processDataNum);

private:
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1, tmpBuf2, tmpBuf3, tmpBuf4;
};

__aicore__ inline void HighPerfStrategy::InitBufImpl(AscendC::TPipe* pipe, uint32_t tileDataNum)
{
    pipe->InitBuffer(tmpBuf1, tileDataNum * sizeof(float));
    pipe->InitBuffer(tmpBuf2, tileDataNum * sizeof(float));
    pipe->InitBuffer(tmpBuf3, tileDataNum * sizeof(float));
    pipe->InitBuffer(tmpBuf4, tileDataNum * sizeof(float));
}

constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375;

constexpr float PI_DOWN = 1.57079637050628662109375;
constexpr float PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375;

constexpr float COS_RES_MULIT_SCA = 2.604926501e-6;
constexpr float COS_RES_ADDICT_UP = -0.0001980894471;
constexpr float COS_2ADDS = 0.008333049340;
constexpr float COS_3ADDS = -0.1666665792;

constexpr float pi_0 = 3.14160156;
constexpr float pi_1 = -8.9071691e-06;
constexpr float pi_2 = -1.74122761e-09;
constexpr float pi_3 = 1.24467439e-13;

__aicore__ inline void HighPerfStrategy::ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                                     AscendC::LocalTensor<float>& yLocal,
                                                     uint32_t processDataNum)
{
    AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();
    AscendC::LocalTensor<float> tmpTensor3 = tmpBuf3.Get<float>();
    AscendC::LocalTensor<float> tmpTensor4 = tmpBuf4.Get<float>();

    const AscendC::LocalTensor<float>& input_x = xLocal;
    const AscendC::LocalTensor<float>& x_vmul = tmpTensor1;
    const AscendC::LocalTensor<float>& x_vmul1 = tmpTensor2;
    const AscendC::LocalTensor<float>& x_vmul0 = yLocal;
    const AscendC::LocalTensor<float>& round_pi_div = tmpTensor1;
    const AscendC::LocalTensor<float>& round_pi_div0 = tmpTensor3;
    const AscendC::LocalTensor<float>& round_pi_div0_1 = tmpTensor2;
    const AscendC::LocalTensor<float>& round_pi_div1 = yLocal;
    const AscendC::LocalTensor<float>& fix = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_1 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_1 = xLocal;
    const AscendC::LocalTensor<float>& fix_2 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_2 = tmpTensor3;
    const AscendC::LocalTensor<float>& x_fixed_3 = xLocal;
    const AscendC::LocalTensor<float>& fix_3 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_4 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_4 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_5 = xLocal;
    const AscendC::LocalTensor<float>& fix_5 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fixed_6 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_6 = xLocal;
    const AscendC::LocalTensor<float>& x_fixed_7 = tmpTensor2;
    const AscendC::LocalTensor<float>& fix_7 = xLocal;
    const AscendC::LocalTensor<float>& x_fixed_8 = tmpTensor3;
    const AscendC::LocalTensor<float>& x_fixed_9 = yLocal;
    const AscendC::LocalTensor<float>& x_pow = tmpTensor2;
    const AscendC::LocalTensor<float>& kover2 = xLocal;
    const AscendC::LocalTensor<float>& kover2floor = tmpTensor3;
    const AscendC::LocalTensor<float>& kover2floorm4 = xLocal;
    const AscendC::LocalTensor<float>& k2 = tmpTensor3;
    const AscendC::LocalTensor<float>& sign = tmpTensor4;
    const AscendC::LocalTensor<float>& sign_1 = tmpTensor1;
    const AscendC::LocalTensor<float>& res_up = tmpTensor3;
    const AscendC::LocalTensor<float>& res_up_1 = xLocal;
    const AscendC::LocalTensor<float>& res_up_2 = tmpTensor3;
    const AscendC::LocalTensor<float>& res_up_3 = xLocal;
    const AscendC::LocalTensor<float>& res_up_4 = tmpTensor3;
    const AscendC::LocalTensor<float>& res_up_5 = xLocal;
    const AscendC::LocalTensor<float>& res_up_6 = tmpTensor3;
    const AscendC::LocalTensor<float>& res_up_7 = tmpTensor2;
    const AscendC::LocalTensor<float>& res_up_8 = xLocal;
    const AscendC::LocalTensor<float>& res_sign = yLocal;
    const AscendC::LocalTensor<float>& res_mins = tmpTensor1;
    const AscendC::LocalTensor<float>& res_maxs = yLocal;

    /// x_vmul = tbe.vmuls(input_x, tvm.const(Constant.PI_FOR_X_TODIV, dtype=dtype))
    AscendC::Muls(x_vmul, input_x, PI_FOR_X_TODIV, processDataNum);
    /// x_vmul1 = tbe.vadds(x_vmul, tvm.const(0.5, dtype=dtype))
    AscendC::Adds(x_vmul1, x_vmul, 0.5f, processDataNum);
    /// x_vmul0 = tbe.vmuls(x_vmul, tvm.const(Constant.ONE_OVER_2048, dtype=dtype))
    AscendC::Muls(x_vmul0, x_vmul, 1.0f / 2048.0f, processDataNum);
    /// round_pi_div = tbe.round_half_up(x_vmul1, "float32")
    AscendC::Cast(round_pi_div, x_vmul1, AscendC::RoundMode::CAST_ROUND, processDataNum);
    /// round_pi_div0 = tbe.round_half_up(x_vmul0, "float32")
    AscendC::Cast(round_pi_div0, x_vmul0, AscendC::RoundMode::CAST_ROUND, processDataNum);
    /// round_pi_div0 = tbe.vmuls(round_pi_div0, tvm.const(2048.0, dtype=dtype))
    AscendC::Muls(round_pi_div0_1, round_pi_div0, 2048.0f, processDataNum);
    /// round_pi_div1 = tbe.vsub(round_pi_div, round_pi_div0)
    AscendC::Sub(round_pi_div1, round_pi_div, round_pi_div0_1, processDataNum);

    /// fix = tbe.vmuls(round_pi_div0, tvm.const(Constant.pi_0, dtype=dtype))
    AscendC::Muls(fix, round_pi_div0_1, pi_0, processDataNum);
    /// x_fixed = tbe.vsub(input_x, fix)
    AscendC::Sub(x_fixed, input_x, fix, processDataNum);
    /// fix = tbe.vmuls(round_pi_div1, tvm.const(Constant.pi_0, dtype=dtype))
    AscendC::Muls(fix_1, round_pi_div1, pi_0, processDataNum);
    /// x_fixed = tbe.vsub(x_fixed, fix)
    AscendC::Sub(x_fixed_1, x_fixed, fix_1, processDataNum);
    /// fix = tbe.vmuls(round_pi_div0, tvm.const(Constant.pi_1, dtype=dtype))
    AscendC::Muls(fix_2, round_pi_div0_1, pi_1, processDataNum);
    /// x_fixed = tbe.vsub(x_fixed, fix)
    AscendC::Sub(x_fixed_2, x_fixed_1, fix_2, processDataNum);

    /// x_fixed = tbe.vadds(x_fixed, tvm.const(Constant.PI_DOWN, dtype=dtype))
    AscendC::Adds(x_fixed_3, x_fixed_2, PI_DOWN, processDataNum);

    /// fix = tbe.vmuls(round_pi_div1, tvm.const(Constant.pi_1, dtype=dtype))
    AscendC::Muls(fix_3, round_pi_div1, pi_1, processDataNum);
    /// x_fixed = tbe.vsub(x_fixed, fix)
    AscendC::Sub(x_fixed_4, x_fixed_3, fix_3, processDataNum);
    /// fix = tbe.vmuls(round_pi_div0, tvm.const(Constant.pi_2, dtype=dtype))
    AscendC::Muls(fix_4, round_pi_div0_1, pi_2, processDataNum);
    /// x_fixed = tbe.vsub(x_fixed, fix)
    AscendC::Sub(x_fixed_5, x_fixed_4, fix_4, processDataNum);
    /// fix = tbe.vmuls(round_pi_div1, tvm.const(Constant.pi_2, dtype=dtype))
    AscendC::Muls(fix_5, round_pi_div1, pi_2, processDataNum);
    /// x_fixed = tbe.vsub(x_fixed, fix)
    AscendC::Sub(x_fixed_6, x_fixed_5, fix_5, processDataNum);
    /// fix = tbe.vmuls(round_pi_div0, tvm.const(Constant.pi_3, dtype=dtype))
    AscendC::Muls(fix_6, round_pi_div0_1, pi_3, processDataNum);
    /// x_fixed = tbe.vsub(x_fixed, fix)
    AscendC::Sub(x_fixed_7, x_fixed_6, fix_6, processDataNum);
    /// fix = tbe.vmuls(round_pi_div1, tvm.const(Constant.pi_3, dtype=dtype))
    AscendC::Muls(fix_7, round_pi_div1, pi_3, processDataNum);
    /// x_fixed = tbe.vsub(x_fixed, fix)
    AscendC::Sub(x_fixed_8, x_fixed_7, fix_7, processDataNum);
    /// x_fixed = tbe.vadds(x_fixed, tvm.const(Constant.PI_RESDOWN_ADDS_NEG, dtype=dtype))
    AscendC::Adds(x_fixed_9, x_fixed_8, PI_RESDOWN_ADDS_NEG, processDataNum);

    /// x_pow = tbe.vmul(x_fixed, x_fixed)
    AscendC::Mul(x_pow, x_fixed_9, x_fixed_9, processDataNum);
    /// kover2 = tbe.vmuls(round_pi_div, tvm.const(0.5, dtype=dtype))
    AscendC::Muls(kover2, round_pi_div, 0.5f, processDataNum);
    /// kover2floor = tbe.floor(kover2, "float32")
    AscendC::Cast(kover2floor, kover2, AscendC::RoundMode::CAST_FLOOR, processDataNum);
    /// kover2floorm4 = tbe.vmuls(kover2floor, tvm.const(4.0, dtype=dtype))
    AscendC::Muls(kover2floorm4, kover2floor, 4.0f, processDataNum);
    /// k2 = tbe.vmuls(round_pi_div, tvm.const(-2.0, dtype=dtype))
    AscendC::Muls(k2, round_pi_div, -2.0f, processDataNum);
    /// sign = tbe.vadd(kover2floorm4, k2)
    AscendC::Add(sign, kover2floorm4, k2, processDataNum);
    /// sign = tbe.vadds(sign, tvm.const(1.0, dtype=dtype))
    AscendC::Adds(sign_1, sign, 1.0f, processDataNum);

    /// res_up = tbe.vmuls(x_pow, tvm.const(Constant.COS_RES_MULIT_SCA, dtype=dtype))
    AscendC::Muls(res_up, x_pow, COS_RES_MULIT_SCA, processDataNum);
    /// res_up = tbe.vadds(res_up, tvm.const(Constant.COS_RES_ADDICT_UP, dtype=dtype))
    AscendC::Adds(res_up_1, res_up, COS_RES_ADDICT_UP, processDataNum);
    /// res_up = tbe.vmul(res_up, x_pow)
    AscendC::Mul(res_up_2, res_up_1, x_pow, processDataNum);
    /// res_up = tbe.vadds(res_up, tvm.const(Constant.COS_2ADDS, dtype=dtype))
    AscendC::Adds(res_up_3, res_up_2, COS_2ADDS, processDataNum);
    /// res_up = tbe.vmul(res_up, x_pow)
    AscendC::Mul(res_up_4, res_up_3, x_pow, processDataNum);
    /// res_up = tbe.vadds(res_up, tvm.const(Constant.COS_3ADDS, dtype=dtype))
    AscendC::Adds(res_up_5, res_up_4, COS_3ADDS, processDataNum);
    /// res_up = tbe.vmul(res_up, x_pow)
    AscendC::Mul(res_up_6, res_up_5, x_pow, processDataNum);
    /// res_up = tbe.vadds(res_up, tvm.const(1.0, dtype=dtype))
    AscendC::Adds(res_up_7, res_up_6, 1.0f, processDataNum);
    /// res_up = tbe.vmul(res_up, x_fixed)
    AscendC::Mul(res_up_8, res_up_7, x_fixed_9, processDataNum);
    /// res_sign = tbe.vmul(res_up, sign)
    AscendC::Mul(res_sign, res_up_8, sign_1, processDataNum);

    /// res_mins = tbe.vmins(res_sign, tvm.const(Constant.NUMBER_POS_ONE, dtype=dtype))
    AscendC::Mins(res_mins, res_sign, 1.0f, processDataNum);
    /// res_maxs = tbe.vmaxs(res_mins, tvm.const(Constant.NUMBER_NEG_ONE, dtype=dtype))
    AscendC::Maxs(res_maxs, res_mins, -1.0f, processDataNum);
}

class HighPrecStrategy
{
public:
    __aicore__ inline HighPrecStrategy() {}
    __aicore__ inline void InitBufImpl(AscendC::TPipe* pipe, uint32_t tileDataNum);
    __aicore__ inline void ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                       AscendC::LocalTensor<float>& yLocal,
                                       uint32_t processDataNum);

private:
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1, tmpBuf2, tmpBuf3, tmpBuf4;
};

__aicore__ inline void HighPrecStrategy::InitBufImpl(AscendC::TPipe* pipe, uint32_t tileDataNum)
{
    pipe->InitBuffer(tmpBuf1, tileDataNum * sizeof(float));
    pipe->InitBuffer(tmpBuf2, tileDataNum * sizeof(float));
    pipe->InitBuffer(tmpBuf3, tileDataNum * sizeof(float));
    pipe->InitBuffer(tmpBuf4, tileDataNum * sizeof(float));
}

constexpr float PI_V4_0 = 1.5708008;
constexpr float PI_V4_1 = -0.0000044535846;
constexpr float PI_V4_2 = -8.706138e-10;
constexpr float PI_V4_3 = 1.5703125;
constexpr float PI_12 = 0.0004837513;
constexpr float PI_22 = 0.000000075495336;
constexpr float PI_32 = 2.5579538e-12;
constexpr float PI_42 = 5.389786e-15;
constexpr float PI_52 = 5.166901e-19;
constexpr float PI_62 = 3.281839e-22;

constexpr float INV_HALF_PI = 0.63661975;

constexpr float SCOEF_4 = 0.0000027183114939898219064;
constexpr float SCOEF_3 = -0.000198393348360966317347;
constexpr float SCOEF_2 = 0.0083333293858894631756;
constexpr float SCOEF_1 = -0.166666666416265235595;

constexpr float CCOEF_4 = 0.0000243904487962774090654;
constexpr float CCOEF_3 = -0.00138867637746099294692;
constexpr float CCOEF_2 = 0.0416666233237390631894;
constexpr float CCOEF_1 = -0.499999997251031003120;

__aicore__ inline void HighPrecStrategy::ComputeImpl(AscendC::LocalTensor<float>& xLocal,
                                                     AscendC::LocalTensor<float>& yLocal,
                                                     uint32_t processDataNum)
{
    AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();
    AscendC::LocalTensor<float> tmpTensor3 = tmpBuf3.Get<float>();
    AscendC::LocalTensor<float> tmpTensor4 = tmpBuf4.Get<float>();

    const AscendC::LocalTensor<float>& input_x = xLocal;
    const AscendC::LocalTensor<float>& x_scaled = tmpTensor1;
    const AscendC::LocalTensor<float>& x_overpi = tmpTensor3;
    const AscendC::LocalTensor<float>& n = tmpTensor2;
    const AscendC::LocalTensor<float>& n0 = yLocal;
    const AscendC::LocalTensor<float>& n0_1 = tmpTensor3;
    const AscendC::LocalTensor<float>& n0_2 = yLocal;
    const AscendC::LocalTensor<float>& n1 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix = tmpTensor2;
    const AscendC::LocalTensor<float>& fix_1 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_1 = tmpTensor1;
    const AscendC::LocalTensor<float>& fix_2 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_2 = tmpTensor2;
    const AscendC::LocalTensor<float>& fix_3 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_3 = tmpTensor1;
    const AscendC::LocalTensor<float>& fix_4 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_4 = tmpTensor2;
    const AscendC::LocalTensor<float>& remain_x = tmpTensor1;
    const AscendC::LocalTensor<float>& temp = tmpTensor2;
    const AscendC::LocalTensor<float>& n2 = tmpTensor1;
    const AscendC::LocalTensor<float>& n0_3 = tmpTensor2;
    const AscendC::LocalTensor<float>& n1_1 = yLocal;
    const AscendC::LocalTensor<float>& fix_5 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_5 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_6 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_6 = xLocal;
    const AscendC::LocalTensor<float>& fix_7 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_7 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_8 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_8 = xLocal;
    const AscendC::LocalTensor<float>& fix_9 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_9 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_10 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_10 = xLocal;
    const AscendC::LocalTensor<float>& fix_11 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_11 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_12 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_12 = xLocal;
    const AscendC::LocalTensor<float>& fix_13 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_13 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_14 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_14 = xLocal;
    const AscendC::LocalTensor<float>& fix_15 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_15 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_16 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_16 = xLocal;
    const AscendC::LocalTensor<float>& fix_17 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_17 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_18 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_18 = xLocal;
    const AscendC::LocalTensor<float>& fix_19 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_19 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_20 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_20 = xLocal;
    const AscendC::LocalTensor<float>& fix_21 = tmpTensor4;
    const AscendC::LocalTensor<float>& x_fix_21 = tmpTensor3;
    const AscendC::LocalTensor<float>& fix_22 = xLocal;
    const AscendC::LocalTensor<float>& x_fix_22 = tmpTensor2;
    const AscendC::LocalTensor<float>& fix_23 = tmpTensor3;
    const AscendC::LocalTensor<float>& x_fix_23 = xLocal;
    const AscendC::LocalTensor<float>& fix_24 = tmpTensor2;
    const AscendC::LocalTensor<float>& x_fix_24 = yLocal;
    const AscendC::LocalTensor<float>& fix_25 = tmpTensor2;
    const AscendC::LocalTensor<float>& x_fix_25 = xLocal;
    const AscendC::LocalTensor<float>& x_pow = tmpTensor2;
    const AscendC::LocalTensor<float>& sin_poly = tmpTensor3;
    const AscendC::LocalTensor<float>& sin_poly_1 = yLocal;
    const AscendC::LocalTensor<float>& sin_poly_2 = tmpTensor3;
    const AscendC::LocalTensor<float>& sin_poly_3 = yLocal;
    const AscendC::LocalTensor<float>& sin_poly_4 = tmpTensor3;
    const AscendC::LocalTensor<float>& sin_poly_5 = yLocal;
    const AscendC::LocalTensor<float>& sin_poly_6 = tmpTensor3;
    const AscendC::LocalTensor<float>& sin_poly_7 = tmpTensor4;
    const AscendC::LocalTensor<float>& sin_poly_8 = yLocal;
    const AscendC::LocalTensor<float>& cos_poly = tmpTensor3;
    const AscendC::LocalTensor<float>& cos_poly_1 = xLocal;
    const AscendC::LocalTensor<float>& cos_poly_2 = tmpTensor3;
    const AscendC::LocalTensor<float>& cos_poly_3 = xLocal;
    const AscendC::LocalTensor<float>& cos_poly_4 = tmpTensor3;
    const AscendC::LocalTensor<float>& cos_poly_5 = xLocal;
    const AscendC::LocalTensor<float>& cos_poly_6 = tmpTensor3;
    const AscendC::LocalTensor<float>& cos_poly_7 = tmpTensor2;
    const AscendC::LocalTensor<float>& n2_1 = xLocal;
    const AscendC::LocalTensor<float>& half_n2 = tmpTensor4;
    const AscendC::LocalTensor<float>& half4_n2 = tmpTensor3;
    const AscendC::LocalTensor<float>& n_half2 = tmpTensor1;
    const AscendC::LocalTensor<float>& n_half4 = tmpTensor4;
    const AscendC::LocalTensor<float>& k1 = tmpTensor3;
    const AscendC::LocalTensor<float>& k2 = tmpTensor1;
    const AscendC::LocalTensor<float>& sign = tmpTensor4;
    const AscendC::LocalTensor<float>& sign_1 = tmpTensor1;
    const AscendC::LocalTensor<float>& ifcos = tmpTensor4;
    const AscendC::LocalTensor<float>& ifsin = xLocal;
    const AscendC::LocalTensor<float>& ifsin_1 = tmpTensor3;
    const AscendC::LocalTensor<float>& temp1 = xLocal;
    const AscendC::LocalTensor<float>& cos_poly_8 = yLocal;
    const AscendC::LocalTensor<float>& res = tmpTensor2;
    const AscendC::LocalTensor<float>& res_1 = yLocal;

    /// x_scaled = tbe.vmuls(input_x, one_over_n)
    AscendC::Muls(x_scaled, input_x, 1.0f / 2048.0f, processDataNum);
    /// x_overpi = tbe.vmuls(x_scaled, inv_half_pi)
    AscendC::Muls(x_overpi, x_scaled, INV_HALF_PI, processDataNum);
    /// n = tbe.round(x_overpi, "float32")
    AscendC::Cast(n, x_overpi, AscendC::RoundMode::CAST_RINT, processDataNum);

    /// n0 = tbe.vmuls(x_overpi, one_over_n)
    AscendC::Muls(n0, x_overpi, 1.0f / 2048.0f, processDataNum);
    /// n0 = tbe.round(n0, "float32")
    AscendC::Cast(n0_1, n0, AscendC::RoundMode::CAST_RINT, processDataNum);
    /// n0 = tbe.vmuls(n0, number_2048)
    AscendC::Muls(n0_2, n0_1, 2048.0f, processDataNum);
    /// n1 = tbe.vsub(n, n0)
    AscendC::Sub(n1, n, n0_2, processDataNum);

    /// fix = tbe.vmuls(n0, pi_0)
    AscendC::Muls(fix, n0_2, PI_V4_0, processDataNum);
    /// x_fix = tbe.vsub(x_scaled, fix)
    AscendC::Sub(x_fix, x_scaled, fix, processDataNum);
    /// fix = tbe.vmuls(n1, pi_0)
    AscendC::Muls(fix_1, n1, PI_V4_0, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_1, x_fix, fix_1, processDataNum);
    /// fix = tbe.vmuls(n0, pi_1)
    AscendC::Muls(fix_2, n0_2, PI_V4_1, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_2, x_fix_1, fix_2, processDataNum);
    /// fix = tbe.vmuls(n1, pi_1)
    AscendC::Muls(fix_3, n1, PI_V4_1, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_3, x_fix_2, fix_3, processDataNum);
    /// fix = tbe.vmuls(n0, pi_2)
    AscendC::Muls(fix_4, n0_2, PI_V4_2, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_4, x_fix_3, fix_4, processDataNum);

    /// remain_x = tbe.vmuls(x_fix, number_2048)
    AscendC::Muls(remain_x, x_fix_4, 2048.0f, processDataNum);
    /// temp = tbe.vmuls(remain_x, inv_half_pi)
    AscendC::Muls(temp, remain_x, INV_HALF_PI, processDataNum);
    /// n2 = tbe.round(temp, "float32")
    AscendC::Cast(n2, temp, AscendC::RoundMode::CAST_RINT, processDataNum);
    /// n0 = tbe.vmuls(n0, number_2048)
    AscendC::Muls(n0_3, n0_2, 2048.0f, processDataNum);
    /// n1 = tbe.vmuls(n1, number_2048)
    AscendC::Muls(n1_1, n1, 2048.0f, processDataNum);
    /// fix = tbe.vmuls(n0, pi_02)
    AscendC::Muls(fix_5, n0_3, PI_V4_3, processDataNum);
    /// x_fix = tbe.vsub(input_x, fix)
    AscendC::Sub(x_fix_5, input_x, fix_5, processDataNum);
    /// fix = tbe.vmuls(n1, pi_02)
    AscendC::Muls(fix_6, n1_1, PI_V4_3, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_6, x_fix_5, fix_6, processDataNum);
    /// fix = tbe.vmuls(n0, pi_12)
    AscendC::Muls(fix_7, n0_3, PI_12, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_7, x_fix_6, fix_7, processDataNum);

    /// fix = tbe.vmuls(n2, pi_02)
    AscendC::Muls(fix_8, n2, PI_V4_3, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_8, x_fix_7, fix_8, processDataNum);
    /// fix = tbe.vmuls(n1, pi_12)
    AscendC::Muls(fix_9, n1_1, PI_12, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_9, x_fix_8, fix_9, processDataNum);
    /// fix = tbe.vmuls(n0, pi_22)
    AscendC::Muls(fix_10, n0_3, PI_22, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_10, x_fix_9, fix_10, processDataNum);

    /// fix = tbe.vmuls(n2, pi_12)
    AscendC::Muls(fix_11, n2, PI_12, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_11, x_fix_10, fix_11, processDataNum);
    /// fix = tbe.vmuls(n1, pi_22)
    AscendC::Muls(fix_12, n1_1, PI_22, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_12, x_fix_11, fix_12, processDataNum);
    /// fix = tbe.vmuls(n0, pi_32)
    AscendC::Muls(fix_13, n0_3, PI_32, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_13, x_fix_12, fix_13, processDataNum);

    /// fix = tbe.vmuls(n2, pi_22)
    AscendC::Muls(fix_14, n2, PI_22, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_14, x_fix_13, fix_14, processDataNum);
    /// fix = tbe.vmuls(n1, pi_32)
    AscendC::Muls(fix_15, n1_1, PI_32, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_15, x_fix_14, fix_15, processDataNum);
    /// fix = tbe.vmuls(n0, pi_42)
    AscendC::Muls(fix_16, n0_3, PI_42, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_16, x_fix_15, fix_16, processDataNum);

    /// fix = tbe.vmuls(n2, pi_32)
    AscendC::Muls(fix_17, n2, PI_32, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_17, x_fix_16, fix_17, processDataNum);
    /// fix = tbe.vmuls(n1, pi_42)
    AscendC::Muls(fix_18, n1_1, PI_42, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_18, x_fix_17, fix_18, processDataNum);
    /// fix = tbe.vmuls(n0, pi_52)
    AscendC::Muls(fix_19, n0_3, PI_52, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_19, x_fix_18, fix_19, processDataNum);

    /// fix = tbe.vmuls(n2, pi_42)
    AscendC::Muls(fix_20, n2, PI_42, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_20, x_fix_19, fix_20, processDataNum);
    /// fix = tbe.vmuls(n1, pi_52)
    AscendC::Muls(fix_21, n1_1, PI_52, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_21, x_fix_20, fix_21, processDataNum);
    /// fix = tbe.vmuls(n0, pi_62)
    AscendC::Muls(fix_22, n0_3, PI_62, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_22, x_fix_21, fix_22, processDataNum);

    /// fix = tbe.vmuls(n2, pi_52)
    AscendC::Muls(fix_23, n2, PI_52, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_23, x_fix_22, fix_23, processDataNum);
    /// fix = tbe.vmuls(n1, pi_62)
    AscendC::Muls(fix_24, n1_1, PI_62, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_24, x_fix_23, fix_24, processDataNum);
    /// fix = tbe.vmuls(n2, pi_62)
    AscendC::Muls(fix_25, n2, PI_62, processDataNum);
    /// x_fix = tbe.vsub(x_fix, fix)
    AscendC::Sub(x_fix_25, x_fix_24, fix_25, processDataNum);

    /// x_pow = tbe.vmul(x_fix, x_fix)
    AscendC::Mul(x_pow, x_fix_25, x_fix_25, processDataNum);
    /// sin_poly = tbe.vmuls(x_pow, scoef4)
    AscendC::Muls(sin_poly, x_pow, SCOEF_4, processDataNum);
    /// sin_poly = tbe.vadds(sin_poly, scoef3)
    AscendC::Adds(sin_poly_1, sin_poly, SCOEF_3, processDataNum);
    /// sin_poly = tbe.vmul(x_pow, sin_poly)
    AscendC::Mul(sin_poly_2, x_pow, sin_poly_1, processDataNum);
    /// sin_poly = tbe.vadds(sin_poly, scoef2)
    AscendC::Adds(sin_poly_3, sin_poly_2, SCOEF_2, processDataNum);
    /// sin_poly = tbe.vmul(x_pow, sin_poly)
    AscendC::Mul(sin_poly_4, x_pow, sin_poly_3, processDataNum);
    /// sin_poly = tbe.vadds(sin_poly, scoef1)
    AscendC::Adds(sin_poly_5, sin_poly_4, SCOEF_1, processDataNum);
    /// sin_poly = tbe.vmul(x_pow, sin_poly)
    AscendC::Mul(sin_poly_6, x_pow, sin_poly_5, processDataNum);
    /// sin_poly = tbe.vadds(sin_poly, tvm.const(1.0, dtype=dtype))
    AscendC::Adds(sin_poly_7, sin_poly_6, 1.0f, processDataNum);
    /// sin_poly = tbe.vmul(x_fix, sin_poly)
    AscendC::Mul(sin_poly_8, x_fix_25, sin_poly_7, processDataNum);

    /// cos_poly = tbe.vmuls(x_pow, ccoef4)
    AscendC::Muls(cos_poly, x_pow, CCOEF_4, processDataNum);
    /// cos_poly = tbe.vadds(cos_poly, ccoef3)
    AscendC::Adds(cos_poly_1, cos_poly, CCOEF_3, processDataNum);
    /// cos_poly = tbe.vmul(x_pow, cos_poly)
    AscendC::Mul(cos_poly_2, x_pow, cos_poly_1, processDataNum);
    /// cos_poly = tbe.vadds(cos_poly, ccoef2)
    AscendC::Adds(cos_poly_3, cos_poly_2, CCOEF_2, processDataNum);
    /// cos_poly = tbe.vmul(x_pow, cos_poly)
    AscendC::Mul(cos_poly_4, x_pow, cos_poly_3, processDataNum);
    /// cos_poly = tbe.vadds(cos_poly, ccoef1)
    AscendC::Adds(cos_poly_5, cos_poly_4, CCOEF_1, processDataNum);
    /// cos_poly = tbe.vmul(x_pow, cos_poly)
    AscendC::Mul(cos_poly_6, x_pow, cos_poly_5, processDataNum);
    /// cos_poly = tbe.vadds(cos_poly, tvm.const(1.0, dtype=dtype))
    AscendC::Adds(cos_poly_7, cos_poly_6, 1.0f , processDataNum);

    /// n2 = tbe.vadds(n2, tvm.const(1.0, dtype=dtype))
    AscendC::Adds(n2_1, n2, 1.0f, processDataNum);
    /// half_n2 = tbe.vmuls(n2, tvm.const(0.5, dtype=dtype))
    AscendC::Muls(half_n2, n2_1, 0.5f, processDataNum);
    /// half4_n2 = tbe.vmuls(n2, tvm.const(0.25, dtype=dtype))
    AscendC::Muls(half4_n2, n2_1, 0.25f, processDataNum);
    /// n_half2 = tbe.floor(half_n2, "float32")
    AscendC::Cast(n_half2, half_n2, AscendC::RoundMode::CAST_FLOOR, processDataNum);
    /// n_half4 = tbe.floor(half4_n2, "float32")
    AscendC::Cast(n_half4, half4_n2, AscendC::RoundMode::CAST_FLOOR, processDataNum);
    /// k1 = tbe.vmuls(n_half2, tvm.const(-2.0, dtype=dtype))
    AscendC::Muls(k1, n_half2, -2.0f, processDataNum);
    /// k2 = tbe.vmuls(n_half4, tvm.const(4.0, dtype=dtype))
    AscendC::Muls(k2, n_half4, 4.0f, processDataNum);
    /// sign = tbe.vadd(k1, k2)
    AscendC::Add(sign, k1, k2, processDataNum);
    /// sign = tbe.vadds(sign, tvm.const(1.0, dtype=dtype))
    AscendC::Adds(sign_1, sign, 1.0f, processDataNum);

    /// ifcos = tbe.vadd(n2, k1)
    AscendC::Add(ifcos, n2_1, k1, processDataNum);
    /// ifsin = tbe.vmuls(ifcos, tvm.const(-1.0, dtype=dtype))
    AscendC::Muls(ifsin, ifcos, -1.0f, processDataNum);
    /// ifsin = tbe.vadds(ifsin, tvm.const(1.0, dtype=dtype))
    AscendC::Adds(ifsin_1, ifsin, 1.0f, processDataNum);

    /// temp1 = tbe.vmul(sin_poly, ifsin)
    AscendC::Mul(temp1, sin_poly_8, ifsin_1, processDataNum);
    /// cos_poly = tbe.vmul(cos_poly, ifcos)
    AscendC::Mul(cos_poly_8, cos_poly_7, ifcos, processDataNum);
    /// res = tbe.vadd(temp1, cos_poly)
    AscendC::Add(res, temp1, cos_poly_8, processDataNum);
    /// res = tbe.vmul(res, sign)
    AscendC::Mul(res_1, res, sign_1, processDataNum);
}

extern "C" __global__ __aicore__ void cos(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

#if __CCE_AICORE__ == 200
    using ComputeStrategy = RefStrategy;
#elif defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    using ComputeStrategy = HighPerfStrategy;
#else
    using ComputeStrategy = HighPrecStrategy;
#endif

    KernelCos<DTYPE_X, ComputeStrategy> op;
    AscendC::TPipe pipe;
    op.Init(x, y,
            tiling_data.bigCoreDataNum,
            tiling_data.smallCoreDataNum,
            tiling_data.tileDataNum,
            tiling_data.bigCoreNum,
            &pipe);
    op.Process();
}
