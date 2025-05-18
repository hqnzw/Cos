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
#include "cos_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cmath>

namespace optiling {
constexpr uint32_t BLOCK_SIZE = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CosTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto socVersion = ascendcPlatform.GetSocVersion();
    auto xType = context->GetInputDesc(0)->GetDataType();

    if (socVersion != platform_ascendc::SocVersion::ASCEND910B && xType == ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t xTypeLength = (xType == ge::DT_FLOAT) ? 4 : 2;
    uint32_t blockElemNum = BLOCK_SIZE / xTypeLength;

    uint32_t inputBlockNum = (inputNum / blockElemNum) + (inputNum % blockElemNum != 0);
    coreNum = std::max(std::min(coreNum, (uint32_t)std::sqrt(0.4f * inputBlockNum)), 1u);
    uint32_t smallCoreBlockNum = inputBlockNum / coreNum;
    uint32_t bigCoreNum = inputBlockNum % coreNum;

    uint32_t smallCoreDataNum = smallCoreBlockNum * blockElemNum;
    uint32_t bigCoreDataNum = smallCoreDataNum + blockElemNum;

    uint32_t ubTileNum;
    if (socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        ubTileNum = (xType == ge::DT_FLOAT) ? 6 : 12;
    } else {
        ubTileNum = (xType == ge::DT_FLOAT) ? 8 : 16;
    }
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubTileNum;
    uint32_t tileDataNum = tileBlockNum * blockElemNum;

    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_bigCoreNum(bigCoreNum);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Cos : public OpDef {
public:
    explicit Cos(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");

        OpAICoreConfig config310p;
        config310p.Input("x")
                  .ParamType(REQUIRED)
                  .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                  .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        config310p.Output("y")
                  .ParamType(REQUIRED)
                  .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                  .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore()
            .AddConfig("ascend310p", config310p);
    }
};

OP_ADD(Cos);
}
