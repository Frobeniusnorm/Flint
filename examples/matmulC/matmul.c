#include <stdio.h>
#include <stdlib.h>
#include <flint/flint.h>

int main(void) {
    flintInit(FLINT_BACKEND_ONLY_CPU);
    fEnableEagerExecution();
    fSetLoggingLevel(F_VERBOSE);

    int32_t data1[] = {1, 3, 3, 4};
    int32_t data2[] = {4, 9, 2, 56};
    size_t shape[] = {2, 2};
    FGraphNode *g1 = fCreateGraph(&data1[0], 4, F_INT32, &shape[0], 2);
    FGraphNode *g2 = fCreateGraph(&data2[0], 4, F_INT32, &shape[0], 2);
    FGraphNode *add = fadd_g(g1, g2);
    FGraphNode *res = fCalculateResult(add);
    int32_t *result = res->result_data->data;
    size_t result_shape[] = {res->operation.shape[0], res->operation.shape[1]};

    size_t repr_len;
    // FIXME: why does it segfault when i use rd here?
    char *repr = fserialize(res, &repr_len);
    for (int i = 0; i < 4; i++) printf("res%i: %d\n", i, result[i]);

    puts(repr);
    printf("repr_len: %d\n", repr_len);

    free(repr);
    fFreeGraph(add);
    flintCleanup();
    return 0;
}
