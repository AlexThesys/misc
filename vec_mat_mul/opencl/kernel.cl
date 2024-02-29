#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void vec_mat_mul(const int num_cols, const int num_rows, __global float* result, const __global half *matrix, __global float* vector) {
    const int t_id 			= get_global_id(0);
    const float bound_check = (t_id < num_rows) ? 1.0f : 0.0f;
	
	const int tail			= num_cols & 3;
	const int cols_trunc 	= (num_cols - tail);
	const int cols_trunc_v	= cols_trunc / 4;

	float4 accum_v			= {0.0f, 0.0f, 0.0f, 0.0f};
	const half *m 			= matrix + t_id * num_cols;
	for (int i = 0; i < cols_trunc_v; i++) {
		float4 t_ps			= vload_half4(i, m);
		float4 v_ps 		= vload4(i, vector);
		accum_v				= mad(t_ps, v_ps, accum_v);
	}
	// compute the reminder chunk
	float accum 			= 0.0f;
    for (int i = cols_trunc; i < num_cols; i++) {
        const float t_ps 	= vload_half(i, m);
        accum 				+= t_ps * vector[i];
    }
    accum 					+= (accum_v[0] + accum_v[1] + accum_v[2] + accum_v[3]);

	// horizontal add
    result[t_id] 			= accum * bound_check;
}