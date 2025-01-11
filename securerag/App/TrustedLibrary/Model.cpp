#include <cstdio>

#include "utils.h"

void loadModel(const char *filePath) {
    FILE *modelFile = fopen(filePath, "r");
    if (modelFile == nullptr) {
        err("model file not found.");
    };

    fseek(modelFile, 512 * 4, 0);
    const int paramSize = 768;
    float param[paramSize];
    fread(param, sizeof(float), paramSize, modelFile);
    for (auto it : param) {
        printf("%f\n", it);
    }
}