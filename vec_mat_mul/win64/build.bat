@echo off
echo Assembling vec_mat_mul_win64.asm with NASM...
nasm -f win64 vec_mat_mul_win64.asm -o vec_mat_mul_win64.obj

if %errorlevel% neq 0 (
    echo Error assembling vec_mat_mul_win64.asm
    exit /b %errorlevel%
)

echo Creating static library memset.lib...
LIB /OUT:vec_mat_mul_win64.lib vec_mat_mul_win64.obj

if %errorlevel% neq 0 (
    echo Error creating static library vec_mat_mul_win64.lib
    exit /b %errorlevel%
)

echo Build successful!