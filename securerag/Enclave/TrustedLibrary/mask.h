#ifndef MASK_H
#define MASK_H
#include <optional>

struct LinearInfo {
    int linearid;
    float *weight;
    float *bias;
    int N;
    int indim;
    int outdim;
};

struct MaskInfo {
    int maskid;
    int linearid;
    float *mask;
    float *unmask;
    int size;
};

// MARK: put & get
void putLinearInfo(LinearInfo linearinfo);

std::optional<LinearInfo *> getLinearInfo(int linearid);

void putMaskInfo(MaskInfo maskinfo);

std::optional<MaskInfo *> getMaskInfo(int maskid);

// MARK: mask & unmask
void mask(float *input, float *output, float *size, int linearid, int *maskid);

void unmask(float *input, float *output, float *size, int maskid);

#endif