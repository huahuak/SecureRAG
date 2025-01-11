#include "mask.h"

#include <algorithm>
#include <optional>
#include <vector>

// MARK: put & get
std::vector<LinearInfo> linearInfoList(0);

std::vector<MaskInfo> maskInfoList(0);

void putLinearInfo(LinearInfo linearinfo) {
    linearInfoList.push_back(linearinfo);
}

std::optional<LinearInfo *> getLinearInfo(int linearid) {
    auto it = std::find_if(linearInfoList.begin(), linearInfoList.end(),
                           [linearid](LinearInfo a) {
                               if (a.linearid == linearid) {
                                   return true;
                               } else {
                                   return false;
                               }
                           });

    if (it != linearInfoList.end()) {
        return &(*it);
    }

    return std::nullopt;
}

void putMaskInfo(MaskInfo maskinfo) { maskInfoList.push_back(maskinfo); }

std::optional<MaskInfo *> getMaskInfo(int maskid) {
    auto it = std::find_if(maskInfoList.begin(), maskInfoList.end(),
                           [maskid](MaskInfo a) {
                               if (a.maskid == maskid) {
                                   return true;
                               } else {
                                   return false;
                               }
                           });

    if (it != maskInfoList.end()) {
        return &(*it);
    }

    return std::nullopt;
}

// MARK: mask & unmask
void mask(float *input, float *output, float *size, int linearid, int *maskid) {
}

void unmask(float *input, float *output, float *size, int maskid) {}