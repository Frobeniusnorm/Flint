#include <stdio.h>
#include <stdlib.h>
#include <flint/flint.h>

int main(void) {
    flintInit(0);

    float data1[] = {1,2,3,4};
    double data2[] = {4,3,2,1};
    double shape[] = {2,2};
    FGraphNode *g1 = fCreateGraph(&data1[0], 4, F_FLOAT32,  &shape[0], 2);
    FGraphNode *g2 = fCreateGraph(&data2[0], 4, F_FLOAT64, &shape[0], 2);
    FGraphNode *mm = fExecuteGraph(fmatmul(g1, g2));
    FResultData *rd = mm->result_data;
    double* result = rd->data;
//    double result_shape = {mm->operation->shape[0], mm->operation->shape[1]};

    fprintf("result: %f", &result);

    return 0;
}
