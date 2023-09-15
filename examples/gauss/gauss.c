#include <flint/flint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    flintInit(FLINT_BACKEND_ONLY_CPU);
    fSetLoggingLevel(F_INFO);

    FGraphNode *img = fload_image("../../flint.png");

    size_t h = img->operation.shape[0];
    size_t w = img->operation.shape[1];
    size_t c = img->operation.shape[2];

    printf("image size: h: %d, w: %d, c: %d\n", h , w, c);

    // channel in first dim
    int transpose[] = {2, 1, 0};
    img = ftranspose(img, transpose);

    float kernelData[1][3][3][1] = {{
        {{1 / 16.0f}, {1 / 8.0f}, {1 / 16.0f}},
        {{1 / 8.0f}, {1 / 4.0f}, {1 / 8.0f}},
        {{1 / 16.0f}, {1 / 8.0f}, {1 / 16.0f}}
    }};
    size_t kernelShape[] = {1, 3, 3, 1};
    FGraphNode *kernel = fCreateGraph(kernelData, 9, F_FLOAT32, kernelShape, 4);

    for (int i = 0; i < 500; i++) {
        size_t shape[] = {c, w + 2, h + 2};
        size_t indices[] = {0, 1, 1};
        img = fextend(img, shape, indices);

        size_t shape2[] = {c, w + 2, h + 2, 1};
        img = freshape(img, shape2,  4);

        uint stride[] = {1, 1, 1};
        img = fconvolve(img, kernel, stride);

        img = fExecuteGraph(img);
    }
    img = ftranspose(img, transpose);

    fstore_image(img, "flint.bmp", F_BMP);
    flintCleanup();
}
