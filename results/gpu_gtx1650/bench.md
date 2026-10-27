GPU: NVIDIA GeForce GTX 1650; torch 2.14.1+cu126; frequencies [4.0, 8.0, 10.0, 14.0, 18.0] GHz.

| Design | Unknowns | Assembly CPU (s) | Assembly CUDA (s) | Solver | Precision | Mean per point (s) | Speed vs PARDISO | Max abs(ΔS11) vs PARDISO | Device memory (MB) |
|---|---|---|---|---|---|---|---|---|---|
| class5 | 50,534 | 1.33 | 0.54 | pardiso | complex128 | 0.713 | 1.00× | 0.0e+00 |  |
| class5 | 50,534 | 1.33 | 0.54 | cudss | complex128 | 0.664 | 1.07× | 9.8e-14 | 1442 |
| class5 | 50,534 | 1.33 | 0.54 | cudss32 | complex64 | 0.134 | 5.31× | 7.6e-05 | 1570 |
| class5 | 50,534 | 1.33 | 0.54 | cudss32ir | complex64 + IR (complex128 residual) | 0.154 | 4.63× | 5.3e-12 | 1802 |
| class19 | 83,490 | 1.78 | 0.89 | pardiso | complex128 | 1.563 | 1.00× | 0.0e+00 |  |
| class19 | 83,490 | 1.78 | 0.89 | cudss | complex128 | 1.757 | 0.89× | 7.3e-14 | 2314 |
| class19 | 83,490 | 1.78 | 0.89 | cudss32 | complex64 | 0.312 | 5.02× | 8.7e-06 | 2564 |
| class19 | 83,490 | 1.78 | 0.89 | cudss32ir | complex64 + IR (complex128 residual) | 0.331 | 4.73× | 1.5e-12 | 2978 |
| class8 | 97,218 | 2.62 | 1.00 | pardiso | complex128 | 3.465 | 1.00× | 0.0e+00 |  |
| class8 | 97,218 | 2.62 | 1.00 | cudss | complex128 | 4.196 | 0.83× | 3.5e-14 | 2136 |
| class8 | 97,218 | 2.62 | 1.00 | cudss32 | complex64 | 0.654 | 5.30× | 9.7e-06 | 2118 |
| class8 | 97,218 | 2.62 | 1.00 | cudss32ir | complex64 + IR (complex128 residual) | 0.716 | 4.84× | 5.8e-12 | 2300 |
| class9 | 124,682 | 2.67 | 1.30 | pardiso | complex128 | 10.363 | 1.00× | 0.0e+00 |  |
| class9 | 124,682 | 2.67 | 1.30 | cudss | complex128 | 8.580 | 1.21× | 2.5e-13 | 2884 |
| class9 | 124,682 | 2.67 | 1.30 | cudss32 | complex64 | 0.943 | 10.99× | 2.2e-05 | 2862 |
| class9 | 124,682 | 2.67 | 1.30 | cudss32ir | complex64 + IR (complex128 residual) | 0.983 | 10.54× | 9.4e-12 | 3090 |
