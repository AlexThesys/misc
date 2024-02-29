#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void vec_mat_mul(const int num_cols, const int num_rows, __global float* result, const __global half* matrix, __global float* vector) {
    int t_id 				= get_global_id(0);
    const int bound_check 	= (t_id < num_rows) ? 1 : 0;
	t_id 					= t_id % get_global_size(0); // or maybe an early exit?
	
	const int tail			= num_cols & 15;
	const int cols_trunc 	= (num_cols - tail);
	const int cols_trunc_v	= cols_trunc / 16;

	float16 accum_v			= {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	const half *m 			= matrix + t_id * num_cols;
	for (int i = 0; i < cols_trunc_v; i++) {
		float16 t_ph		= vload_half16(i, m);
		float16 v_ph 		= vload16(i, vector);
		accum_v				= mad(t_ph, v_ph, accum_v);
	}
	// compute the reminder chunk
	float accum 			= 0.0f;
    for (int i = cols_trunc; i < num_cols; i++) {
        const float t_ps 	= vload_half(i, m);
        accum 				+= t_ps * vector[i];
    }
	// horizontal add
    accum 					+= 	accum_v[0] + accum_v[1] + accum_v[2] + accum_v[3] + accum_v[4] + accum_v[5] + accum_v[6] + accum_v[7]
							+ 	accum_v[8] + accum_v[9] + accum_v[10] + accum_v[11] + accum_v[12] + accum_v[13] + accum_v[14] + accum_v[15];

	if (bound_check)	// we have to branch at some point
		result[t_id] 		= accum;
}

// __kernel void vec_mat_mul(const int num_cols, const int num_rows, __global float* result, const __global half* matrix, __global float* vector) {
    // int t_id 				= get_global_id(0);
    // const int bound_check 	= (t_id < num_rows) ? 1 : 0;
	// t_id 					= t_id % get_global_size(0); // or maybe an early exit?
	
	// const int tail			= num_cols & 7;
	// const int cols_trunc 	= (num_cols - tail);
	// const int cols_trunc_v	= cols_trunc / 8;

	// float8 accum_v			= {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	// const half *m 			= matrix + t_id * num_cols;
	// for (int i = 0; i < cols_trunc_v; i++) {
		// float8 t_ph			= vload_half8(i, m);
		// float8 v_ph 		= vload8(i, vector);
		// accum_v				= mad(t_ph, v_ph, accum_v);
	// }
	// // compute the reminder chunk
	// float accum 			= 0.0f;
    // for (int i = cols_trunc; i < num_cols; i++) {
        // const float t_ps 	= vload_half(i, m);
        // accum 				+= t_ps * vector[i];
    // }
	// // horizontal add
    // accum 					+= accum_v[0] + accum_v[1] + accum_v[2] + accum_v[3] + accum_v[4] + accum_v[5] + accum_v[6] + accum_v[7];

	// if (bound_check)	// we have to branch at some point
		// result[t_id] 		= accum;
// }
