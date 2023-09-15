#include <stdio.h>
#include <stdlib.h>
#include <flint/flint.h>

int main(void) {
    flintInit(FLINT_BACKEND_ONLY_CPU);
    fEnableEagerExecution();
    fSetLoggingLevel(F_VERBOSE);

    size_t img_shape[] = {30, 30, 3};
    FGraphNode *img = fconstant_i(4, &img_shape, sizeof(img_shape) / sizeof(size_t));

    size_t kernel_shape[] = {3,3,3};
    FGraphNode *kernel = fconstant_i(4, &kernel_shape, sizeof(kernel_shape) / sizeof(size_t));

    unsigned int steps[] = { 1, 1, 1 };
    FGraphNode * conv = fconvolve(img, kernel, &steps);

    FGraphNode *res = fCalculateResult(conv);
    int * result = res->result_data->data;
    size_t * result_shape = res->operation.shape;

    // print output
    for (int i = 0; i <  res->result_data->num_entries; i++) {
        printf("res %i: %d\n", i, result[i]);
    }
    // print shape
    printf("shape: ");
    for (int i = 0; i < res->operation.dimensions; i++) {
        printf("%d,", result_shape[i]);
    }
    printf("\n");

    fFreeGraph(res);
    flintCleanup();
    return 0;
}
