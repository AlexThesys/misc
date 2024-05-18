void quary_plaform_features(struct platform *features) {
	enum reg_ids {eax, ebx, ecx, edx};
	int	cpuinfo[4] = {0};
	__cpuid	(cpuinfo, 1);
	
	features->_sse2 = (0 != (cpuinfo[edx] & (1 << 26)));
	int os_xsave_support = (0 != (cpuinfo[ecx] & (1<<27)));
	features->_avx = os_xsave_support && (0 != (cpuinfo[ecx] & (1<<28)));
	if (features->_avx) { // both xsave and avx support are required for the rest of the features
		const unsigned long long xcr0_features = _xgetbv(0);
		const unsigned long long sse_avx_states_mask = 0b0110;
		const int sse_avx_states_enabled = (sse_avx_states_mask == (xcr0_features & sse_avx_states_mask));
		features->_avx = sse_avx_states_enabled;
		if (sse_avx_states_enabled) {
			features->_f16c = (0 != (cpuinfo[ecx] & (1<<29)));
			features->_fma3 = (0 != (cpuinfo[ecx] & (1<<12)));
			__cpuidex(cpuinfo, 7, 0);
			features->_avx2 = (0 != (cpuinfo[ebx] & (1<<5)));
		}
	}
}
