kernel void vec_calc(global const float* a,
                     global const float* b,
                     global float* result
                     ){
    int id = get_global_id(0);

    float term = sin(a[id]) * cos(b[id]);
    result[id]= exp(term + sign(term) * cos(a[id])*sin(b[id]));
}
